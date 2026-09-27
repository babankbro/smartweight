"""Generates ml/Frozen_Features_And_Backbones.ipynb.

Follow-up to Multitask_Weight_Experiment.ipynb, whose run produced
MAE 74-80 kg / R2 0.27-0.35 across all seven variants, with the validation
curve still falling when the 40-epoch budget ran out.
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))


# =========================================================================
md(r"""
# Frozen features, backbones, and where the ceiling actually is

Follows the crop/no-crop run, which gave:

| variant | MAE | RMSE | R² | ≤10% |
|---|---|---|---|---|
| D full-frame + geom | **74.01** | 93.43 | 0.347 | 33.4% |
| A full-frame | 74.87 | 94.25 | 0.336 | 34.6% |
| F crop+ctx + geom | 75.05 | 93.89 | 0.341 | 31.5% |
| C crop+20% ctx | 75.10 | 94.25 | 0.336 | 34.8% |
| E tight-crop + geom | 76.60 | 96.53 | 0.303 | 31.7% |
| E- single-task | 76.95 | 97.71 | 0.286 | 34.2% |
| B tight-crop | 79.86 | 98.60 | 0.273 | 31.3% |

Three things that run established:

1. **Cropping costs ~5 kg** (A 74.87 vs B 79.86), and geometry recovers about
   two thirds of it (E 76.60 vs B 79.86). Scale is real and it is measurable.
2. **Multi-task barely moved the headline number** (E 76.60 vs E- 76.95) — but
   look at the curves: the single-task folds stopped at epochs 22/30/28 with
   MAE swinging 98→83 between logs, while multi-task folds trained 32-40 epochs
   smoothly. Multi-task is stabilising training even where it is not yet
   improving the final score.
3. **Every variant was still improving when the budget ran out.** A fold 1 went
   83.54 → 78.73 between epochs 21 and 31 and never triggered early stopping.
   These models are **underfitted, not overfitted.** The original notebook
   needed 147 epochs to reach its best.

So the first thing to fix is not the backbone. It is the budget, plus the
missing train-vs-validation diagnostic that would have made this obvious.

This notebook runs, in order of cost:

| block | question | runtime |
|---|---|---|
| 2-3 | frozen features + shallow heads — is fine-tuning even needed? | ~10 min |
| 4 | learning curve — would more animals help, or is the input saturated? | seconds |
| 5 | backbone sweep on frozen features | ~10 min |
| 6 | fine-tune patch: longer budget, train-MAE logging | ~1 h |

With roughly 220 distinct images, freezing a strong pretrained encoder and
fitting a shallow head is the textbook low-data recipe — and it costs minutes
instead of hours.
""")

# =========================================================================
md("## Block 1 — Setup (rerun blocks 1-5 of the first notebook first)")

code(r"""
# Requires from the previous notebook: df, GEOM_COLS, DATA_FULL, report(), SEED
# If this is a fresh runtime, re-run its Blocks 1-5 before continuing.
!pip -q install timm

import os, copy, itertools, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEEDED = ["df", "GEOM_COLS", "DATA_FULL", "report", "SEED",      # blocks 2-5
          "REG_TARGETS", "CowDataset", "TRAIN_TF", "EVAL_TF",     # block 6 only
          "BATCH_SIZE", "LR", "N_BREEDS", "multitask_loss"]
_missing = [n for n in NEEDED if n not in globals()]
assert not _missing, f"re-run Blocks 1-8 of notebook 1 first; missing: {_missing}"

print(f"images {len(df)} | animals {df['animal_id'].nunique()}")
y_all = df["weight"].to_numpy(float)
groups = df["animal_id"].to_numpy()

# The bar every model here must clear.
print(f"\ntarget: mean {y_all.mean():.1f}  sd {y_all.std():.1f}  "
      f"range {y_all.min():.0f}-{y_all.max():.0f} kg")
report(y_all, np.full_like(y_all, y_all.mean()), "predict-the-mean")
print(f"{'fine-tuned B0 (best so far)':<34} MAE  74.01 | RMSE  93.43 | "
      f"MAPE 22.38% | R2  0.347")
""")

# =========================================================================
md("""
## Block 2 — Extract frozen features

No training here: each encoder runs once in `eval()` mode and its embedding is
cached. Fine-tuning 5M parameters on ~220 distinct images is already a stretch;
a frozen encoder plus a shallow head has far fewer degrees of freedom to waste.

`dinov2` is included because self-supervised features transfer unusually well
to fine-grained shape tasks, which is exactly what body condition is.

Both crop modes are extracted so the frozen results can be compared against the
fine-tuned A-vs-B result on equal terms.
""")

code(r"""
BACKBONES = {
    "dinov2_s":  "vit_small_patch14_dinov2.lvd142m",
    "clip_b16":  "vit_base_patch16_clip_224.openai",
    "convnext_t": "convnext_tiny.fb_in22k",
    "effv2_s":   "tf_efficientnetv2_s.in21k",
}
FEAT_IMG = 224
CROP_MODES = ["none", "context"]     # the two that won in notebook 1

class PlainDS(Dataset):
    def __init__(self, sub, root, crop_mode, tf, ctx=0.20):
        self.df = sub.reset_index(drop=True)
        self.root, self.crop, self.tf, self.ctx = root, crop_mode, tf, ctx
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.loc[i]
        im = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
        if self.crop != "none":
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            if self.crop == "context":
                dw, dh = (x2-x1)*self.ctx/2, (y2-y1)*self.ctx/2
                x1, y1, x2, y2 = x1-dw, y1-dh, x2+dw, y2+dh
            W, H = im.size
            box = (max(0,int(x1)), max(0,int(y1)), min(W,int(x2)), min(H,int(y2)))
            if box[2]-box[0] > 8 and box[3]-box[1] > 8:
                im = im.crop(box)
        return self.tf(im)

class Letterbox:
    def __init__(self, size, fill=114): self.size, self.fill = size, fill
    def __call__(self, im):
        w, h = im.size; s = self.size/max(w, h)
        im = im.resize((max(1,round(w*s)), max(1,round(h*s))), Image.BILINEAR)
        c = Image.new("RGB", (self.size,)*2, (self.fill,)*3)
        c.paste(im, ((self.size-im.size[0])//2, (self.size-im.size[1])//2))
        return c

FEAT_TF = transforms.Compose([
    Letterbox(FEAT_IMG), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract(model_name, crop_mode):
    m = timm.create_model(model_name, pretrained=True, num_classes=0,
                          **({"dynamic_img_size": True} if "dinov2" in model_name
                             or "clip" in model_name else {}))
    m = m.eval().to(DEVICE)
    dl = DataLoader(PlainDS(df, DATA_FULL, crop_mode, FEAT_TF),
                    batch_size=32, shuffle=False, num_workers=2)
    out = []
    for x in dl:
        # flip-average: the same trick as TTA, applied once at extraction
        x = x.to(DEVICE)
        f = (m(x) + m(torch.flip(x, dims=[3]))) / 2
        out.append(f.float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(out)

FEATS = {}
for crop in CROP_MODES:
    for key, name in BACKBONES.items():
        tag = f"{key}|{crop}"
        cache = f"/content/feat_{key}_{crop}.npy"
        if os.path.exists(cache):
            FEATS[tag] = np.load(cache)
        else:
            print("extracting", tag, "...", end=" ", flush=True)
            FEATS[tag] = extract(name, crop)
            np.save(cache, FEATS[tag])
            print(FEATS[tag].shape)

for k, v in FEATS.items():
    print(f"{k:<22} {v.shape}")
""")

# =========================================================================
md("""
## Block 3 — Shallow heads on frozen features

The question the user raised directly: does a random forest on deep features
beat fine-tuning? With this much data it very plausibly does, and it costs
seconds per fit instead of an hour.

`+geom` appends the bounding-box geometry. On cropped features that is the only
channel through which absolute size can reach the model at all.
""")

code(r"""
GEOM = df[GEOM_COLS].to_numpy(float)
GEOM = (GEOM - GEOM.mean(0)) / (GEOM.std(0) + 1e-8)

HEADS = {
    "ridge":  lambda: make_pipeline(StandardScaler(),
                                    RidgeCV(alphas=np.logspace(-2, 5, 40))),
    "pca64+ridge": lambda: make_pipeline(StandardScaler(), PCA(64, random_state=SEED),
                                         RidgeCV(alphas=np.logspace(-2, 5, 40))),
    "rf":     lambda: RandomForestRegressor(600, min_samples_leaf=2,
                                            max_features=0.3, n_jobs=-1,
                                            random_state=SEED),
    "pca64+rf": lambda: make_pipeline(StandardScaler(), PCA(64, random_state=SEED),
                                      RandomForestRegressor(600, min_samples_leaf=2,
                                                            n_jobs=-1, random_state=SEED)),
    "hgb":    lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=400,
                                                    learning_rate=0.05,
                                                    random_state=SEED),
    "svr":    lambda: make_pipeline(StandardScaler(), SVR(C=100, epsilon=5)),
}

gkf = GroupKFold(n_splits=5)
SPLITS = list(gkf.split(df, groups=groups))

def oof_predict(X, make_head):
    p = np.zeros(len(X))
    for tr, va in SPLITS:
        h = make_head().fit(X[tr], y_all[tr])
        p[va] = h.predict(X[va])
    return p

rows = []
for tag, F in FEATS.items():
    for use_geom in (False, True):
        X = np.hstack([F, GEOM]) if use_geom else F
        for hname, make in HEADS.items():
            p = oof_predict(X, make)
            m = report(y_all, p, f"{tag}{'+geom' if use_geom else '':<6} {hname}")
            rows.append(dict(features=tag, geom=use_geom, head=hname,
                             **{k: m[k] for k in ("MAE","RMSE","MAPE","R2","bias","within10")}))

frozen = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
print("\n" + "="*78)
display(frozen.head(15).round(2))
""")

# =========================================================================
md("""
## Block 4 — Learning curve: would more animals help?

This is the experiment that decides where the next month of effort goes.

Train on an increasing number of **animals** (never images — splitting by image
would leak) and watch the validation error.

- still falling at 100% → collecting more cattle pays off
- flat well above target → the input has been exhausted; the missing
  information is **scale**, and no amount of data or compute recovers it from a
  photo with nothing of known size in frame
""")

code(r"""
best = frozen.iloc[0]
Xb = np.hstack([FEATS[best["features"]], GEOM]) if best["geom"] else FEATS[best["features"]]
make_best = HEADS[best["head"]]
print(f"using {best['features']}{'+geom' if best['geom'] else ''} / {best['head']}\n")

rng = np.random.default_rng(SEED)
uniq = df["animal_id"].unique()
fractions = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
curve = []

for frac in fractions:
    maes = []
    for tr, va in SPLITS:
        tr_animals = pd.unique(groups[tr])
        keep = rng.choice(tr_animals, max(4, int(len(tr_animals)*frac)), replace=False)
        sub = tr[np.isin(groups[tr], keep)]
        h = make_best().fit(Xb[sub], y_all[sub])
        maes.append(np.abs(h.predict(Xb[va]) - y_all[va]).mean())
    curve.append(dict(frac=frac, n_animals=int(len(uniq)*0.8*frac),
                      MAE=np.mean(maes), sd=np.std(maes)))
    print(f"  {frac:>4.0%} of train animals (~{curve[-1]['n_animals']:>3}) -> "
          f"MAE {curve[-1]['MAE']:6.2f} +/- {curve[-1]['sd']:.2f}")

curve = pd.DataFrame(curve)
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4.5))
plt.errorbar(curve.n_animals, curve.MAE, yerr=curve.sd, marker="o", capsize=3)
plt.axhline(74.01, color="crimson", ls="--", label="fine-tuned B0")
plt.axhline(np.abs(y_all - y_all.mean()).mean(), color="gray", ls=":", label="predict-the-mean")
plt.xlabel("training animals"); plt.ylabel("val MAE (kg)")
plt.title("learning curve"); plt.legend(); plt.grid(alpha=.3); plt.show()

tail = curve.MAE.iloc[-3:].to_numpy()
slope = (tail[0] - tail[-1]) / max(1, curve.n_animals.iloc[-1] - curve.n_animals.iloc[-3])
print(f"\nslope over the last third: {slope*100:.2f} kg per 100 extra animals")
print("=> more data still pays" if slope*100 > 2 else
      "=> curve has flattened: the limit is the INPUT (scale), not the sample size")
""")

# =========================================================================
md("""
## Block 5 — Where is the headroom?

Compare, on one axis:

1. tape measurements only (Block 4 of notebook 1) — the label-quality ceiling
2. frozen features + shallow head — this notebook
3. fine-tuned EfficientNet-B0 — notebook 1

If (1) is far better than (2) and (3), the images are the weak link and the
`image → H, L → weight` split is worth building. If (1) is no better, the
labels themselves are noisy and no model change will help.
""")

code(r"""
summary = pd.DataFrame([
    dict(approach="predict-the-mean",
         MAE=np.abs(y_all - y_all.mean()).mean(),
         RMSE=float(np.sqrt(((y_all - y_all.mean())**2).mean())), R2=0.0),
    dict(approach="fine-tuned B0 (D full+geom)", MAE=74.01, RMSE=93.43, R2=0.347),
    dict(approach=f"frozen {best['features']} / {best['head']}",
         MAE=best["MAE"], RMSE=best["RMSE"], R2=best["R2"]),
]).sort_values("MAE")
display(summary.round(2))

print("\n>>> Paste the 'Height + L' line from Block 4 of notebook 1 here to "
      "complete the picture.")
print(">>> tape MAE ~25-30  -> images are the weak link; build image -> H,L -> weight")
print(">>> tape MAE ~70     -> the labels are noisy; fix the data, not the model")

# Ensemble of the best few frozen configurations: cheap, and usually worth
# 2-4 kg because the errors are only partly correlated.
top = frozen.head(5)
preds = []
for _, r in top.iterrows():
    X = np.hstack([FEATS[r["features"]], GEOM]) if r["geom"] else FEATS[r["features"]]
    preds.append(oof_predict(X, HEADS[r["head"]]))
report(y_all, np.mean(preds, 0), "ensemble of top-5 frozen")

pa = pd.DataFrame({"a": groups, "y": y_all, "p": np.mean(preds, 0)}).groupby("a").mean()
report(pa["y"], pa["p"], "ensemble, per-animal")
""")

# =========================================================================
md("""
## Block 6 — Fine-tuning, done properly this time

Only worth running once the blocks above say fine-tuning is competitive.

Two changes from notebook 1:

- **150 epochs, patience 30.** Every fold there was still improving at epoch 40,
  and the original notebook needed 147 epochs to reach its best.
- **Train MAE is logged next to validation MAE.** Without it there is no way to
  tell underfitting from overfitting, which is precisely the mistake this run
  is correcting.

Read the gap: train ≈ val and both high → underfitted, train longer or use a
bigger backbone. train ≪ val → overfitted, regularise or shrink the model.
""")

code(r"""
EPOCHS_LONG, PATIENCE_LONG = 150, 30

def run_fold_v2(tr, va, *, crop_mode, feat_cols, multitask=True,
                backbone="efficientnet_b0", root=DATA_FULL,
                epochs=EPOCHS_LONG, seed=SEED, log_every=15):
    torch.manual_seed(seed)
    def _stats(t):
        mu, sd = tr[t].mean(), tr[t].std()
        mu = 0.0 if pd.isna(mu) else float(mu)
        sd = float(sd) if (not pd.isna(sd) and sd > 1e-6) else 1.0
        return mu, sd
    norm = {t: _stats(t) for t in REG_TARGETS}

    mk = lambda s, tf: CowDataset(s, root, tf, crop_mode=crop_mode,
                                  feat_cols=feat_cols, norm=norm)
    dl_tr = DataLoader(mk(tr, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, drop_last=True)
    dl_tr_eval = DataLoader(mk(tr, EVAL_TF), batch_size=BATCH_SIZE, num_workers=2)
    dl_va = DataLoader(mk(va, EVAL_TF), batch_size=BATCH_SIZE, num_workers=2)

    model = MultiTaskNetV2(backbone, len(feat_cols), N_BREEDS,
                           len(REG_TARGETS), multitask=multitask).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    wmu, wsd = norm["weight"]

    def mae_on(dl):
        model.eval(); P, T = [], []
        with torch.no_grad():
            for x, f, yv, m, b in dl:
                ro, _ = model(x.to(DEVICE), f.to(DEVICE))
                P.append(ro[:, 0].cpu().numpy()); T.append(yv[:, 0].numpy())
        return float(np.abs((np.concatenate(P) - np.concatenate(T)) * wsd).mean())

    best, best_wts, stale, hist = 1e9, copy.deepcopy(model.state_dict()), 0, []
    for ep in range(epochs):
        model.train()
        for x, f, yv, m, b in dl_tr:
            x, f, yv, m, b = [t.to(DEVICE) for t in (x, f, yv, m, b)]
            opt.zero_grad()
            ro, bo = model(x, f)
            loss, _ = multitask_loss(ro, bo, yv, m, b, model.log_var, multitask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        v = mae_on(dl_va)
        if v < best - 0.1:
            best, best_wts, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if ep % log_every == 0 or stale >= PATIENCE_LONG:
            t_ = mae_on(dl_tr_eval)          # the diagnostic that was missing
            hist.append((ep, t_, v))
            print(f"      ep {ep+1:>3}  train {t_:6.2f}  val {v:6.2f}  "
                  f"gap {v-t_:+6.2f}  best {best:6.2f}")
        if stale >= PATIENCE_LONG:
            print(f"      early stop at ep {ep+1}")
            break

    model.load_state_dict(best_wts)
    model.eval(); out = []
    with torch.no_grad():
        for x, f, yv, m, b in dl_va:
            x, f = x.to(DEVICE), f.to(DEVICE)
            r1, _ = model(x, f); r2, _ = model(torch.flip(x, dims=[3]), f)
            out.append(((r1 + r2) / 2).cpu().numpy())
    ps = np.concatenate(out)
    return {t: ps[:, i] * norm[t][1] + norm[t][0]
            for i, t in enumerate(REG_TARGETS[:ps.shape[1]])}, hist


class MultiTaskNetV2(nn.Module):
    # Same as notebook 1 but the backbone is swappable via timm.
    def __init__(self, backbone, n_feats=0, n_breeds=2, n_reg=5,
                 dropout=0.4, multitask=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        d_img = self.backbone.num_features
        self.tab = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 64), nn.SiLU(),
            nn.Dropout(dropout/2), nn.Linear(64, 64), nn.SiLU(),
        ) if n_feats else None
        d = d_img + (64 if n_feats else 0)
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 256), nn.SiLU())
        self.reg = nn.Linear(256, n_reg if multitask else 1)
        self.breed = nn.Linear(256, n_breeds) if multitask else None
        nn.init.zeros_(self.reg.weight); nn.init.zeros_(self.reg.bias)
        self.log_var = nn.Parameter(torch.zeros((n_reg if multitask else 1) + 1))

    def forward(self, img, feats):
        z = self.backbone(img)
        if self.tab is not None:
            z = torch.cat([z, self.tab(feats)], 1)
        z = self.trunk(z)
        return self.reg(z), (self.breed(z) if self.breed is not None else None)


# Winner from notebook 1 (D: full frame + geometry), now with a real budget.
dfx = df.copy()
dfx[GEOM_COLS] = (dfx[GEOM_COLS] - dfx[GEOM_COLS].mean()) / (dfx[GEOM_COLS].std() + 1e-8)

oof = np.full(len(dfx), np.nan)
for k, (tr_i, va_i) in enumerate(SPLITS[:3], 1):
    print(f"   fold {k}  train {len(tr_i)} / val {len(va_i)}")
    r, hist = run_fold_v2(dfx.iloc[tr_i], dfx.iloc[va_i],
                          crop_mode="none", feat_cols=GEOM_COLS, seed=SEED+k)
    oof[va_i] = r["weight"]

ok = ~np.isnan(oof)
report(y_all[ok], oof[ok], "B0, 150 epochs, full+geom")
print(f"{'same config, 40 epochs':<34} MAE  74.01 | RMSE  93.43 | R2  0.347")
""")

# =========================================================================
md("""
## How to act on the numbers

| observation | meaning | do this |
|---|---|---|
| Block 6 train ≈ val, both ~70 | underfitted | bigger backbone (`convnext_tiny`, `tf_efficientnetv2_s`), higher resolution |
| Block 6 train ≪ val | overfitted | keep B0, raise dropout/weight-decay, stronger augmentation |
| frozen + RF ≥ fine-tuned | too little data to fine-tune | ship the frozen pipeline; it is faster to train *and* to serve |
| learning curve still falling | sample size binds | collect more cattle |
| learning curve flat ~70 | **input** binds | ArUco marker; no model change will help |
| tape measurements ≫ everything | images are the weak link | build `image → H, L → weight` |
| nothing beats ~70 anywhere | labels are noisy | audit weighing protocol and ID matching first |

One caution on reading Block 3: every row there shares the same five folds, so
picking the single best of ~48 configurations will be optimistic by a few
kilograms. Treat the top group as a tie and confirm the winner on a held-out
set of animals that took no part in this selection.
""")

# =========================================================================
nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
         **({"source": s.splitlines(keepends=True)} if t == "markdown" else
            {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).parent / "Frozen_Features_And_Backbones.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
