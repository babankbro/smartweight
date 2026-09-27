"""Train the weight ensemble served by ai-api (notebook V3, cells 35-39).

    fine-tuned backbones (frozen) -> concat features + geometry
        -> SVR | Ridge | multi-task MLP -> average

Writes ai-api/models/predict_weight/ensemble/{svr.joblib, ridge.joblib, mlp.pt,
manifest.json, oof_predictions.csv}.

Run inside the ai-api image so training and serving share one environment:

    docker run --rm -v <repo>/ai-api:/work/ai-api -v <repo>/ml:/work/ml \
        -v <dataset>:/data -w /work bovweightai-ai-api \
        python ml/03_train_weight_ensemble.py --images /data/cow_dataset --cow-csv /data/cow.csv

About the numbers this prints: the four backbones were fine-tuned (with weight as
a target) on fold 0 of an 8-fold GroupKFold, i.e. on ~7/8 of these animals. Their
features for those animals encode memorised labels, so plain cross-validation of
the heads is optimistic. The script rebuilds that 8-fold split, checks it against
the error stored in each checkpoint, and reports the animals the backbones never
saw separately - that subset is the honest estimate.
"""

import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-api"))
from app import weight as W  # noqa: E402

SEED = 42

# Typical error for the MLP's non-weight outputs. Cross-validating them here would
# leak the same way the weight does (the backbones were fine-tuned on these
# targets too), so the leakage-free figures from `Predition_Weight_V3 (1).ipynb`
# cell 14 are used: EfficientNet-B0 multi-task, GroupKFold by animal.
CLEAN_AUX_BASELINE = {
    "Height_y":  {"MAE": 8.91, "R2": 0.357},
    "L_y":       {"MAE": 12.75, "R2": 0.334},
    "age_years": {"MAE": 1.28, "R2": 0.096},
    "ratio_lh":  {"MAE": 0.08, "R2": 0.018},
}
FORMULA = (5.854, -595.98)   # most cow.csv weights are exactly 5.854*L - 595.98


# ---------------------------------------------------------------- data ------
def normalise_id(raw) -> str:
    s = str(raw).strip().upper().split("_")[0].split("-")[0]
    if s.startswith("C"):
        s = s[1:]
    return f"C{int(s):03d}" if s.isdigit() else str(raw).strip().upper()


def parse_age_years(v):
    if pd.isna(v):
        return np.nan
    m = re.match(r"^([\d.]+)\s*([YM]?)$", str(v).strip().upper())
    if not m:
        return np.nan
    n = float(m.group(1))
    return n / 12.0 if m.group(2) == "M" else n


def load_table(images: Path, cow_csv: Path) -> pd.DataFrame:
    cow = pd.read_csv(cow_csv, encoding="utf-8-sig")
    cow = cow[[c for c in cow.columns if not re.fullmatch(r"_.*_", str(c))]]
    cow["animal_id"] = cow["ID"].map(normalise_id)
    for c in ("Weight", "Height", "L"):
        cow[c] = pd.to_numeric(cow[c], errors="coerce")
    cow = cow.dropna(subset=["Weight"]).drop_duplicates("animal_id")
    cow["age_years"] = cow["Age"].map(parse_age_years)
    cow["ratio_lh"] = cow["L"] / cow["Height"]
    cow["label_is_formula"] = (cow["Weight"] - (FORMULA[0] * cow["L"] + FORMULA[1])).abs() < 0.01

    rows = [dict(path=str(p), split=split, animal_id=normalise_id(p.name))
            for split in ("train", "valid", "test")
            for p in sorted((images / split / "images").glob("*"))
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    df = pd.DataFrame(rows).merge(
        cow[["animal_id", "Weight", "Height", "L", "age_years", "ratio_lh",
             "Category", "label_is_formula"]], on="animal_id", how="inner")
    df = df.rename(columns={"Weight": "weight", "Height": "Height_y", "L": "L_y"})
    return df.reset_index(drop=True)


# ------------------------------------------------------------- metrics ------
def report(y, p, label) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    m = dict(n=int(len(y)), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
             MAPE=float(np.abs(e / y).mean() * 100), R2=float(1 - (e ** 2).sum() / ss) if ss else None,
             bias=float(e.mean()), within10=float((np.abs(e / y) <= .10).mean() * 100))
    print(f"  {label:<40} n={m['n']:>4} | MAE {m['MAE']:6.2f} | MAPE {m['MAPE']:5.2f}% | "
          f"R2 {m['R2'] if m['R2'] is None else round(m['R2'], 3)} | <=10% {m['within10']:5.1f}%")
    return m


# ----------------------------------------------------------------- MLP ------
def train_mlp(X, Y, epochs=400, batch_size=32, lr=1e-3, seed=SEED):
    """Notebook cell 33, with one change: the final-epoch weights are kept.
    The notebook picked the best epoch by *validation* MAE, which leaks the
    validation fold into model selection and cannot be done at deployment."""
    torch.manual_seed(seed)
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor((Y - mu) / sd, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)  # BN needs >1
    model = W.FrozenMultitaskMLP(X.shape[1], n_reg=Y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = 0
            for i in range(out.shape[1]):
                s = model.log_var[i]
                loss = loss + torch.exp(-s) * F.smooth_l1_loss(out[:, i], yb[:, i]) + s
            loss.backward()
            opt.step()
        sch.step()
    return model.eval(), mu, sd


def mlp_predict(model, mu, sd, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32))[:, 0].numpy() * sd[0] + mu[0]


def make_svr():
    return make_pipeline(StandardScaler(), SVR(C=100, epsilon=5))


def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 40)))


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path, help="Roboflow export root")
    ap.add_argument("--cow-csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=W.ENSEMBLE_DIR)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/weight_features.npz"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mlp-epochs", type=int, default=400)
    ap.add_argument("--skip-split-check", action="store_true",
                    help="skip re-running the 4 fine-tuned nets to verify the 8-fold split")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    np.random.seed(SEED)

    df = load_table(args.images, args.cow_csv)
    print(f"images {len(df)} | animals {df.animal_id.nunique()} | "
          f"weighed-looking animals {(~df.drop_duplicates('animal_id').label_is_formula).sum()}")

    # ---- features (cached: extraction is the slow part) --------------------
    if args.cache.exists() and list(np.load(args.cache, allow_pickle=True)["paths"]) == list(df.path):
        z = np.load(args.cache, allow_pickle=True)
        feats, geom = z["feats"], pd.DataFrame(list(z["geom"]))
        print("loaded cached features", feats.shape)
    else:
        fx = W.FeatureExtractor()
        t0, feats, geom = time.time(), [], []
        for i in range(0, len(df), 16):
            ims = [W.to_model_input(Image.open(p)) for p in df.path[i:i + 16]]
            geom += [fx.geometry(im) for im in ims]
            feats.append(fx.backbone_features(ims))
            print(f"  features {min(i + 16, len(df))}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
        feats, geom = np.concatenate(feats), pd.DataFrame(geom)
        np.savez(args.cache, feats=feats, geom=np.array(geom.to_dict("records")),
                 paths=np.array(df.path))
    df = pd.concat([df, geom[W.GEOM_COLS + ["found"]]], axis=1)
    print(f"detector found an object on {int(df.found.sum())}/{len(df)} images")

    # Standardised once over all rows, like the notebook (cell 35).
    geom_stats = {c: (float(df[c].mean()), float(df[c].std())) for c in W.GEOM_COLS}
    G = W.scale_geometry(df[W.GEOM_COLS].to_numpy(float), geom_stats)
    X = np.hstack([feats, G])
    y = df.weight.to_numpy(float)
    Y = df[W.REG_TARGETS].to_numpy(float)
    for j in range(Y.shape[1]):
        Y[np.isnan(Y[:, j]), j] = np.nanmean(Y[:, j])
    groups = df.animal_id.to_numpy()
    print("X", X.shape)

    # ---- which animals did the backbones never see? ------------------------
    ft_tr, ft_va = list(GroupKFold(n_splits=8).split(df, groups=groups))[0]
    seen = np.zeros(len(df), bool)
    seen[ft_tr] = True
    split_ok = not args.skip_split_check
    if split_ok:
        print("\nverifying the rebuilt fine-tune split against each checkpoint's stored val MAE:")
    for key in (W.BACKBONE_KEYS if split_ok else []):
        net, ck = W.load_finetuned(key)
        wmu, wsd = ck["norm"]["weight"]
        out = []
        with torch.no_grad():
            for i in range(0, len(df), 32):
                x = torch.stack([W.FEAT_TF(W.to_model_input(Image.open(p)))
                                 for p in df.path[i:i + 32]])
                out.append(net(x, torch.tensor(G[i:i + 32], dtype=torch.float32))[0][:, 0].numpy())
        p = np.concatenate(out) * wsd + wmu
        va_mae, tr_mae = np.abs(p - y)[~seen].mean(), np.abs(p - y)[seen].mean()
        ok = abs(va_mae - ck["val_mae"]) < 3.0 and tr_mae < va_mae
        split_ok &= ok
        print(f"  {key:<11} stored {ck['val_mae']:6.2f} | rebuilt held-out {va_mae:6.2f} | "
              f"fine-tune train {tr_mae:6.2f}  {'ok' if ok else 'MISMATCH'}")
        del net
    print(f"held-out animals: {len(set(groups[~seen]))} ({(~seen).sum()} images) - "
          + ("split verified" if split_ok else "split NOT verified, honest subset unreliable"))
    # Known result (2026-09-27): the rebuilt split does NOT match Colab's -
    # convnext_t 51.8 vs stored 73.6, effv2_s 60.1 vs 73.9. GroupKFold tie order
    # differs between numpy/sklearn builds, so the unseen subset is unknown.

    # ---- cross-validated heads (notebook cell 39) ---------------------------
    oof = {k: np.zeros(len(df)) for k in ("svr", "ridge", "mlp")}
    for k, (tr, va) in enumerate(GroupKFold(n_splits=args.folds).split(X, groups=groups), 1):
        t0 = time.time()
        oof["svr"][va] = make_svr().fit(X[tr], y[tr]).predict(X[va])
        oof["ridge"][va] = make_ridge().fit(X[tr], y[tr]).predict(X[va])
        m, mu, sd = train_mlp(X[tr], Y[tr], epochs=args.mlp_epochs, seed=SEED + k)
        oof["mlp"][va] = mlp_predict(m, mu, sd, X[va])
        print(f"  fold {k}/{args.folds} done ({time.time() - t0:.0f}s)", flush=True)
    oof["ensemble"] = (oof["svr"] + oof["ridge"] + oof["mlp"]) / 3

    metrics = {}
    print("\nout-of-fold, ALL animals (optimistic: backbones saw ~7/8 of them):")
    metrics["all"] = {k: report(y, v, k) for k, v in oof.items()}
    if split_ok:
        print("\nout-of-fold, animals the backbones NEVER saw (honest):")
        metrics["unseen_by_backbones"] = {k: report(y[~seen], v[~seen], k) for k, v in oof.items()}
    weighed = ~df.label_is_formula.to_numpy()
    print("\nensemble by label type:")
    metrics["ensemble_by_label"] = {
        "formula_labels": report(y[~weighed], oof["ensemble"][~weighed], "formula labels (5.854*L-595.98)"),
        "weighed_labels": report(y[weighed], oof["ensemble"][weighed], "weighed-looking labels"),
    }
    report(y, np.full_like(y, y.mean()), "reference: predict-the-mean")

    # ---- final fit on everything, save -------------------------------------
    print("\nfitting final heads on all rows ...")
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(make_svr().fit(X, y), args.out / "svr.joblib")
    joblib.dump(make_ridge().fit(X, y), args.out / "ridge.joblib")
    m, mu, sd = train_mlp(X, Y, epochs=args.mlp_epochs)
    torch.save(dict(state_dict=m.state_dict(), in_features=int(X.shape[1]),
                    y_mu=mu.tolist(), y_sd=sd.tolist(), targets=W.REG_TARGETS),
               args.out / "mlp.pt")

    if split_ok:
        honest, honest_src = metrics["unseen_by_backbones"]["ensemble"]["MAE"], "unseen_by_backbones"
    else:
        # The leaked CV MAE would understate the error shown to users. The best
        # leakage-free figure available is the backbones' own held-out MAE.
        honest = min(float(torch.load(W.WEIGHT_DIR / f"{k}_multitask_fold1.pth", mmap=True, map_location="cpu",
                                      weights_only=True)["val_mae"]) for k in W.BACKBONE_KEYS)
        honest_src = "best fine-tuned backbone, held-out animals (Colab val_mae)"
    now = datetime.now(timezone.utc)
    manifest = dict(
        version=f"feat-ensemble@{now:%Y-%m-%d}",
        created=now.isoformat(timespec="seconds"),
        backbones=W.BACKBONE_KEYS,
        backbone_files=[f"{k}_multitask_fold1.pth" for k in W.BACKBONE_KEYS],
        detector="yolo11n.pt (COCO, largest box of any class)",
        input_size=W.INPUT_SIZE, feat_img=W.FEAT_IMG, n_features=int(X.shape[1]),
        geom_cols=W.GEOM_COLS, geom_stats=geom_stats,
        heads=["svr", "ridge", "mlp"], combine="mean",
        mlp_epochs=args.mlp_epochs, folds=args.folds,
        n_images=int(len(df)), n_animals=int(df.animal_id.nunique()),
        typical_error_kg=round(honest, 1),
        typical_error_source=honest_src,
        aux_typical_error=CLEAN_AUX_BASELINE,
        aux_typical_error_source="V3 notebook cell 14 (B0 multi-task, GroupKFold by animal)",
        metrics=metrics,
        caveats=[
            "Backbones were fine-tuned on ~7/8 of the training animals, so these CV "
            "metrics are optimistic. The Colab 8-fold split could not be rebuilt "
            "(convnext_t 51.8 vs stored 73.6), so no leakage-free subset is known; "
            "the fine-tuned backbones alone scored 73.6-101.6 kg MAE on held-out animals.",
            f"{int(df.drop_duplicates('animal_id').label_is_formula.sum())} of "
            f"{df.animal_id.nunique()} animals have weight == 5.854*L - 595.98 "
            "exactly: labels are mostly a tape formula, not scale weights.",
            "Trained on cattle only (no buffalo).",
        ],
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    pd.DataFrame({"path": [Path(p).name for p in df.path], "animal_id": groups,
                  "actual": y, "seen_by_backbones": seen, **{f"pred_{k}": v for k, v in oof.items()}}
                 ).to_csv(args.out / "oof_predictions.csv", index=False)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
