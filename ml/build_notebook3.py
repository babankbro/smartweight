"""Generates ml/Final_Training.ipynb - the long run that produces a saved model.

Written after three rounds of evidence:
  round 1  crop/no-crop matrix   -> full frame + geometry wins (MAE 74.01)
  round 2  frozen features       -> best 78.55, loses to fine-tuning by 4.5 kg
  round 3  auxiliary targets     -> weight error = 3x the linear-dimension error
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))


# =========================================================================
md(r"""
# Final training run — long budget, best configuration, saved model

## What the previous three rounds settled

**Round 1 — cropping.** Full frame beat a tight crop by 5 kg (74.87 vs 79.86);
adding bounding-box geometry recovered most of the loss on cropped inputs
(79.86 → 76.60). Best overall: **full frame + geometry, MAE 74.01, R² 0.347**.

**Round 2 — frozen features.** Best frozen configuration was
`convnext_tiny` (full frame) + geometry + SVR at **MAE 78.55, R² 0.30** —
**4.5 kg worse than fine-tuning**. Three further findings:

- **SVR dominates, random forest never reached the top 15.** On 700-dimensional
  embeddings with ~700 samples, a margin-based regressor generalises where tree
  ensembles do not.
- **Geometry added only 0.31 kg** on full-frame features (78.86 → 78.55), versus
  3.3 kg on tight crops. A full frame already encodes scale; the explicit
  numbers are mostly redundant once you stop cropping.
- Full frame beat context-crop again, and `convnext_tiny` beat `dinov2_s`.

So: **fine-tuning wins, the frozen path is closed.**

**Round 3 — auxiliary targets.** Weight, Height and L all landed at R² ≈ 0.34 —
the model has learned a single "how big is this animal" factor, not three
separate skills. And the arithmetic closes exactly:

$$W \propto H^2 L \;\Rightarrow\; \sigma_W \approx 2\sigma_H + \sigma_L
= 2(6.67\%) + 7.70\% = 21.0\% \quad\text{vs measured } 22.38\%$$

Weight error is the linear-dimension error, tripled by the cube law. To reach
8% on weight you need **2.7%** on linear dimensions — currently 7%.

**Round 4 — learning curve.** 16 → 112 animals moved MAE 96.2 → 78.6. Fitting
`a·n^-b + c` gives an exponent of only **0.127** (typical deep-learning curves
sit at 0.3–0.5), and the asymptote is **not identifiable** from seven noisy
points (±253 kg — ignore any specific value). What *is* robust is the short
extrapolation: **doubling the herd to ~224 animals buys about 4.6 kg**, and
reaching 500 animals buys about 9.5 kg. Real, but slow, and nowhere near the
target on its own.

## What this notebook does

| block | purpose | time |
|---|---|---|
| 2 | learning-curve fit, reproducible | seconds |
| 3 | **diagnostic**: train MAE vs val MAE at full budget — settles underfit vs overfit | ~20 min |
| 4 | configuration screen: backbone × augmentation × feature set | ~60 min |
| 5 | **final 5-fold run, saves every fold to Drive** | 2–3 h |
| 6 | final metrics, plots, out-of-fold predictions | minutes |
| 7 | inference helper for `ai-api` | — |

In a hurry: run blocks 1, 3, then 5 with the defaults already set from rounds
1–3, and skip the screen.
""")

# =========================================================================
md("## Block 1 — Setup")

code(r"""
# Needs Blocks 1-8 of Multitask_Weight_Experiment.ipynb to have run.
!pip -q install timm

import os, copy, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm, matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEEDED = ["df", "GEOM_COLS", "DATA_FULL", "report", "SEED", "REG_TARGETS",
          "CowDataset", "N_BREEDS", "multitask_loss", "LetterboxPad", "NORM"]
_missing = [n for n in NEEDED if n not in globals()]
assert not _missing, f"re-run Blocks 1-8 of notebook 1 first; missing: {_missing}"

SAVE_DIR = "/content/drive/MyDrive/KSU/Research/SmartWeight/weight_model"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE   = 320
BATCH_SIZE = 16
LR         = 3e-4
y_all  = df["weight"].to_numpy(float)
groups = df["animal_id"].to_numpy()
SPLITS = list(GroupKFold(n_splits=5).split(df, groups=groups))

print(f"device {DEVICE} | images {len(df)} | animals {df['animal_id'].nunique()}")
print(f"saving to {SAVE_DIR}")
report(y_all, np.full_like(y_all, y_all.mean()), "predict-the-mean")
print(f"{'round 1 best (B0, 40 ep)':<34} MAE  74.01 | RMSE  93.43 | R2  0.347")
print(f"{'round 2 best (frozen convnext)':<34} MAE  78.55 | RMSE  97.09 | R2  0.300")
""")

# =========================================================================
md("""
## Block 2 — Learning curve fit

Reproduces round 4. The point is not the asymptote — seven noisy points cannot
pin that down — but the **exponent** and the short extrapolation.
""")

code(r"""
from scipy.optimize import curve_fit

n   = np.array([16, 33, 50, 67, 84, 101, 112], float)
mae = np.array([96.22, 85.32, 83.52, 84.47, 82.03, 80.48, 78.56])
sd  = np.array([9.84, 8.78, 9.13, 4.41, 7.78, 4.51, 4.04])

f = lambda x, a, b, c: a * x**(-b) + c
p, cov = curve_fit(f, n, mae, p0=[200, .5, 60], sigma=sd, maxfev=200000,
                   bounds=([0, .01, 0], [1e6, 3, 95]))
err = np.sqrt(np.diag(cov))
print(f"MAE(n) = {p[0]:.1f}*n^-{p[1]:.3f} + {p[2]:.1f}")
print(f"exponent {p[1]:.3f}  (deep-learning curves usually 0.3-0.5 -> this is shallow)")
print(f"asymptote {p[2]:.1f} +/- {err[2]:.0f} kg  -> NOT identifiable, do not quote it\n")
for N in (112, 224, 300, 500):
    print(f"  n={N:>4} -> MAE {f(N, *p):6.2f} kg")
print(f"\ndoubling the herd buys {f(112,*p)-f(224,*p):.1f} kg")

plt.figure(figsize=(7, 4.5))
plt.errorbar(n, mae, yerr=sd, fmt="o", capsize=3, label="measured")
xs = np.linspace(16, 500, 200)
plt.plot(xs, f(xs, *p), label=f"fit  n^-{p[1]:.2f}")
plt.axhline(74.01, color="crimson", ls="--", label="fine-tuned B0")
plt.xlabel("training animals"); plt.ylabel("val MAE (kg)")
plt.legend(); plt.grid(alpha=.3); plt.title("returns to more cattle"); plt.show()
""")

# =========================================================================
md("""
## Block 3 — Augmentation, rebuilt

One change matters more than the rest: **`RandomErasing` is removed.**

Erasing random patches is standard for classification, where an object stays
the same class with a chunk missing. Here the answer *is* the animal's size and
shape, so cutting a hole in the body corrupts the very relationship being
learned. It was in round 1's transform and should not have been.

Three strengths are defined so the screen can settle the question rather than
argue it. If Block 4 shows underfitting, lighter augmentation is the right move
— the usual instinct to add more augmentation is backwards in that case.
""")

code(r"""
def build_tf(strength: str, img_size=IMG_SIZE):
    lb = LetterboxPad(img_size)
    if strength == "light":
        aug = [transforms.RandomHorizontalFlip(0.5),
               transforms.RandomAffine(degrees=4, translate=(0.03, 0.03),
                                       scale=(0.97, 1.03)),
               transforms.ColorJitter(0.15, 0.15, 0.1, 0.02)]
    elif strength == "medium":
        aug = [transforms.RandomHorizontalFlip(0.5),
               transforms.RandomAffine(degrees=7, translate=(0.05, 0.05),
                                       scale=(0.95, 1.05)),
               transforms.ColorJitter(0.25, 0.25, 0.2, 0.03)]
    elif strength == "strong":
        # scale jitter widened deliberately: it is the one augmentation that
        # simulates a different shooting distance
        aug = [transforms.RandomHorizontalFlip(0.5),
               transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                       scale=(0.85, 1.15)),
               transforms.ColorJitter(0.35, 0.35, 0.3, 0.05),
               transforms.RandomGrayscale(p=0.05)]
    else:
        raise ValueError(strength)
    return transforms.Compose([lb, *aug, transforms.ToTensor(), NORM])

EVAL_TF_F = transforms.Compose([LetterboxPad(IMG_SIZE), transforms.ToTensor(), NORM])
print("augmentation levels: light / medium / strong   (RandomErasing removed)")
""")

# =========================================================================
md("""
## Block 4 — Model and training loop

Two additions over round 1:

- **Train MAE is logged beside validation MAE.** Round 1 lacked this, which is
  why the underfitting only became visible from the shape of the curves.
- **EMA of the weights.** A running average of recent weights is consistently
  worth a little on small noisy datasets and costs one extra copy of the model.
""")

code(r"""
class WeightNet(nn.Module):
    def __init__(self, backbone, n_feats=0, n_breeds=2, n_reg=5,
                 dropout=0.4, multitask=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        d_img = self.backbone.num_features
        self.tab = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 64), nn.SiLU(),
            nn.Dropout(dropout / 2), nn.Linear(64, 64), nn.SiLU(),
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


def train_one(tr, va, *, backbone="efficientnet_b0", aug="medium",
              feat_cols=(), crop_mode="none", epochs=150, patience=30,
              multitask=True, seed=SEED, log_every=15, use_ema=True, verbose=True):
    torch.manual_seed(seed)
    feat_cols = list(feat_cols)

    def _stats(t):
        mu, sd_ = tr[t].mean(), tr[t].std()
        mu = 0.0 if pd.isna(mu) else float(mu)
        sd_ = float(sd_) if (not pd.isna(sd_) and sd_ > 1e-6) else 1.0
        return mu, sd_
    norm = {t: _stats(t) for t in REG_TARGETS}
    wmu, wsd = norm["weight"]

    mk = lambda s, tf: CowDataset(s, DATA_FULL, tf, crop_mode=crop_mode,
                                  feat_cols=feat_cols, norm=norm)
    dl_tr      = DataLoader(mk(tr, build_tf(aug)), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, drop_last=True)
    dl_tr_eval = DataLoader(mk(tr, EVAL_TF_F), batch_size=BATCH_SIZE, num_workers=2)
    dl_va      = DataLoader(mk(va, EVAL_TF_F), batch_size=BATCH_SIZE, num_workers=2)

    model = WeightNet(backbone, len(feat_cols), N_BREEDS, len(REG_TARGETS),
                      multitask=multitask).to(DEVICE)
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda a, c, _: 0.999 * a + 0.001 * c,
        use_buffers=True,      # without this the EMA keeps BatchNorm stats from init
    ) if use_ema else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def mae_of(net, dl):
        net.eval(); P, T = [], []
        with torch.no_grad():
            for x, f_, yv, m, b in dl:
                ro, _ = net(x.to(DEVICE), f_.to(DEVICE))
                P.append(ro[:, 0].cpu().numpy()); T.append(yv[:, 0].numpy())
        return float(np.abs((np.concatenate(P) - np.concatenate(T)) * wsd).mean())

    best, best_wts, stale, hist = 1e9, copy.deepcopy(model.state_dict()), 0, []
    for ep in range(epochs):
        model.train()
        for x, f_, yv, m, b in dl_tr:
            x, f_, yv, m, b = [t.to(DEVICE) for t in (x, f_, yv, m, b)]
            opt.zero_grad()
            ro, bo = model(x, f_)
            loss, _ = multitask_loss(ro, bo, yv, m, b, model.log_var, multitask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
        sch.step()

        v = mae_of(model, dl_va)
        if v < best - 0.1:
            best, best_wts, stale = v, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if verbose and (ep % log_every == 0 or stale >= patience or ep == epochs - 1):
            t_ = mae_of(model, dl_tr_eval)
            hist.append(dict(epoch=ep + 1, train=t_, val=v))
            print(f"      ep {ep+1:>3}  train {t_:6.2f}  val {v:6.2f}  "
                  f"gap {v-t_:+6.2f}  best {best:6.2f}")
        if stale >= patience:
            print(f"      early stop at ep {ep+1}")
            break

    model.load_state_dict(best_wts)
    if ema is not None:
        v_ema = mae_of(ema.module, dl_va)
        if v_ema < best:
            print(f"      EMA better: {v_ema:.2f} < {best:.2f}, using EMA weights")
            model.load_state_dict(ema.module.state_dict()); best = v_ema

    model.eval(); out = []
    with torch.no_grad():
        for x, f_, yv, m, b in dl_va:                      # flip TTA
            x, f_ = x.to(DEVICE), f_.to(DEVICE)
            r1, _ = model(x, f_); r2, _ = model(torch.flip(x, dims=[3]), f_)
            out.append(((r1 + r2) / 2).cpu().numpy())
    ps = np.concatenate(out)
    preds = {t: ps[:, i] * norm[t][1] + norm[t][0]
             for i, t in enumerate(REG_TARGETS[:ps.shape[1]])}
    return dict(model=model, preds=preds, norm=norm, best=best,
                hist=pd.DataFrame(hist), feat_cols=feat_cols)
""")

# =========================================================================
md("""
## Block 5 — Diagnostic: underfitted or overfitted?

One fold, full budget, with the train/validation gap printed. Everything after
this depends on the answer.

| pattern | reading | what to do |
|---|---|---|
| train ≈ val, both ~70–80 | **underfitted** | bigger backbone, higher resolution, *lighter* augmentation |
| train ≪ val (gap > 25) | **overfitted** | keep B0, raise dropout/weight-decay, *stronger* augmentation |
| train falls, val flat | at the information limit | neither — the input is the constraint |
""")

code(r"""
# Standardise geometry once, as in the earlier rounds.
dfx = df.copy()
dfx[GEOM_COLS] = (dfx[GEOM_COLS] - dfx[GEOM_COLS].mean()) / (dfx[GEOM_COLS].std() + 1e-8)

tr_i, va_i = SPLITS[0]
print("diagnostic: efficientnet_b0, medium aug, full frame + geometry, 150 epochs")
diag = train_one(dfx.iloc[tr_i], dfx.iloc[va_i],
                 backbone="efficientnet_b0", aug="medium",
                 feat_cols=GEOM_COLS, crop_mode="none", epochs=150, log_every=10)

h = diag["hist"]
plt.figure(figsize=(7, 4.5))
plt.plot(h.epoch, h.train, label="train"); plt.plot(h.epoch, h.val, label="val")
plt.xlabel("epoch"); plt.ylabel("MAE (kg)"); plt.legend(); plt.grid(alpha=.3)
plt.title("train vs validation"); plt.show()

gap = h.val.iloc[-1] - h.train.iloc[-1]
print(f"\nfinal gap {gap:+.1f} kg | best val {diag['best']:.2f} "
      f"(round 1, 40 epochs: 78.4 on this fold)")
print("=> UNDERFITTED: go bigger, lighter augmentation" if gap < 15 else
      "=> OVERFITTED: regularise harder, stronger augmentation" if gap > 25 else
      "=> balanced: gains now come from data or inputs, not capacity")
""")

# =========================================================================
md("""
## Block 6 — Configuration screen

One fold each, so the comparison is cheap. `breed` appears here as an *input*,
not only as an auxiliary target: the scan page already asks the user to pick
โคเนื้อ / โคนม / กระบือ, so it is genuinely available at inference time and using
it is not leakage.

Trim `CONFIGS` if the runtime budget is tight — the defaults from rounds 1–3
are already sensible without this step.
""")

code(r"""
BREED_COLS = [c for c in dfx.columns if c.startswith("breed_")
              and pd.api.types.is_numeric_dtype(dfx[c])]
FEATURE_SETS = {
    "geom":        GEOM_COLS,
    "geom+breed":  GEOM_COLS + BREED_COLS,
    "none":        [],
}
print("breed columns available as input:", BREED_COLS)

CONFIGS = [
    dict(backbone="efficientnet_b0",    aug="medium", feats="geom"),
    dict(backbone="efficientnet_b0",    aug="light",  feats="geom"),
    dict(backbone="convnext_tiny",      aug="medium", feats="geom"),
    dict(backbone="tf_efficientnetv2_s", aug="medium", feats="geom"),
    dict(backbone="efficientnet_b0",    aug="medium", feats="geom+breed"),
]

screen = []
for cfg in CONFIGS:
    tag = f"{cfg['backbone']}/{cfg['aug']}/{cfg['feats']}"
    print(f"\n--- {tag} ---")
    r = train_one(dfx.iloc[tr_i], dfx.iloc[va_i],
                  backbone=cfg["backbone"], aug=cfg["aug"],
                  feat_cols=FEATURE_SETS[cfg["feats"]], crop_mode="none",
                  epochs=150, log_every=30)
    screen.append(dict(config=tag, val_MAE=r["best"],
                       gap=float(r["hist"].val.iloc[-1] - r["hist"].train.iloc[-1]),
                       **cfg))
    del r; torch.cuda.empty_cache()

screen_df = pd.DataFrame(screen).sort_values("val_MAE").reset_index(drop=True)
display(screen_df.round(2))
BEST = screen_df.iloc[0]
print(f"\nwinner: {BEST['config']}  (single fold - treat close scores as a tie)")
""")

# =========================================================================
md("""
## Block 7 — Final run, saved to Drive

Five folds at the winning configuration. Each fold's weights are written to
Drive together with **everything needed to reproduce a prediction**: the feature
column order, the standardisation statistics used for those columns, and the
per-fold target normalisation. A checkpoint without those is not reloadable —
the model would silently receive differently-scaled inputs.

The five fold models together form the deployment ensemble.
""")

code(r"""
FINAL = dict(
    backbone = BEST["backbone"] if "BEST" in globals() else "efficientnet_b0",
    aug      = BEST["aug"]      if "BEST" in globals() else "medium",
    feats    = BEST["feats"]    if "BEST" in globals() else "geom",
    crop_mode = "none",
    epochs    = 200,
    patience  = 40,
)
feat_cols = FEATURE_SETS[FINAL["feats"]]
print("final configuration:", FINAL)

# The exact statistics used to standardise the tabular features, so inference
# can reproduce them. Recomputing from a different dataframe would not match.
geom_stats = {c: (float(df[c].mean()), float(df[c].std() + 1e-8)) for c in feat_cols}

oof = np.full(len(dfx), np.nan)
manifest = dict(created=pd.Timestamp.utcnow().isoformat(), img_size=IMG_SIZE,
                reg_targets=REG_TARGETS, n_breeds=int(N_BREEDS),
                feat_cols=feat_cols, geom_stats=geom_stats, folds=[], **FINAL)

for k, (tr_i_, va_i_) in enumerate(SPLITS, 1):
    print(f"\n=== fold {k}/5  train {len(tr_i_)} / val {len(va_i_)} "
          f"({dfx.iloc[va_i_]['animal_id'].nunique()} animals) ===")
    r = train_one(dfx.iloc[tr_i_], dfx.iloc[va_i_],
                  backbone=FINAL["backbone"], aug=FINAL["aug"],
                  feat_cols=feat_cols, crop_mode=FINAL["crop_mode"],
                  epochs=FINAL["epochs"], patience=FINAL["patience"],
                  seed=SEED + k, log_every=25)
    oof[va_i_] = r["preds"]["weight"]

    path = f"{SAVE_DIR}/fold{k}.pth"
    torch.save(dict(state_dict=r["model"].state_dict(), norm=r["norm"],
                    val_mae=r["best"], fold=k), path)
    manifest["folds"].append(dict(fold=k, path=os.path.basename(path),
                                  val_mae=float(r["best"]),
                                  n_val_animals=int(dfx.iloc[va_i_]["animal_id"].nunique())))
    print(f"  saved {path}  (val MAE {r['best']:.2f})")
    del r; torch.cuda.empty_cache()

report(y_all, oof, "FINAL out-of-fold")
manifest["oof"] = {k: float(v) for k, v in report(y_all, oof).items()}

with open(f"{SAVE_DIR}/manifest.json", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=2)
pd.DataFrame({"animal_id": dfx["animal_id"], "path": dfx["path"],
              "actual": y_all, "predicted": oof}).to_csv(
    f"{SAVE_DIR}/oof_predictions.csv", index=False)
print(f"\nmanifest + predictions written to {SAVE_DIR}")
""")

# =========================================================================
md("## Block 8 — Final evaluation")

code(r"""
print(f"{'predict-the-mean':<34} MAE {np.abs(y_all-y_all.mean()).mean():6.2f}")
print(f"{'original notebook CNN':<34} MAE  79.91 | RMSE  92.66 | R2 < 0")
print(f"{'round 1 (40 epochs)':<34} MAE  74.01 | RMSE  93.43 | R2  0.347")
print(f"{'round 2 (frozen features)':<34} MAE  78.55 | RMSE  97.09 | R2  0.300")
report(y_all, oof, "THIS RUN")

pa = pd.DataFrame({"a": groups, "y": y_all, "p": oof}).groupby("a").mean()
report(pa["y"], pa["p"], "per-animal (photos averaged)")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].scatter(y_all, oof, alpha=.35)
lim = [y_all.min(), y_all.max()]; ax[0].plot(lim, lim, "r--")
ax[0].set_xlabel("actual (kg)"); ax[0].set_ylabel("predicted (kg)")
ax[0].set_title("per image")

ax[1].scatter(pa["y"], pa["p"], alpha=.6)
ax[1].plot(lim, lim, "r--"); ax[1].set_xlabel("actual"); ax[1].set_ylabel("predicted")
ax[1].set_title("per animal")

d = oof - y_all; bias, s = d.mean(), d.std(ddof=1)
ax[2].scatter((y_all + oof) / 2, d, alpha=.35)
ax[2].axhline(bias, color="crimson", label=f"bias {bias:+.1f}")
ax[2].axhline(bias + 1.96*s, color="gray", ls="--")
ax[2].axhline(bias - 1.96*s, color="gray", ls="--")
ax[2].set_title(f"Bland-Altman  (LoA {bias-1.96*s:+.0f} .. {bias+1.96*s:+.0f} kg)")
ax[2].legend()
plt.tight_layout(); plt.show()

# Where it fails worst - usually informative about capture conditions.
worst = (pd.DataFrame({"animal": groups, "path": dfx["path"], "actual": y_all,
                       "pred": oof, "err": np.abs(oof - y_all)})
         .sort_values("err", ascending=False).head(10))
display(worst.round(1))
""")

# =========================================================================
md("""
## Block 9 — Loading the ensemble for inference

What `ai-api` will need. Kept in the notebook so the saved artefacts can be
verified here, before anything is wired into the service.
""")

code(r"""
def load_ensemble(save_dir=SAVE_DIR):
    man = json.load(open(f"{save_dir}/manifest.json", encoding="utf-8"))
    nets = []
    for f in man["folds"]:
        ck = torch.load(f"{save_dir}/{f['path']}", map_location=DEVICE,
                        weights_only=False)
        net = WeightNet(man["backbone"], len(man["feat_cols"]), man["n_breeds"],
                        len(man["reg_targets"])).to(DEVICE)
        net.load_state_dict(ck["state_dict"]); net.eval()
        nets.append((net, ck["norm"]))
    return man, nets


@torch.no_grad()
def predict_weight(pil_img, geom: dict, man, nets):
    # geom: raw (unstandardised) values, keyed by man['feat_cols']
    tf = transforms.Compose([LetterboxPad(man["img_size"]),
                             transforms.ToTensor(), NORM])
    x = tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    v = [(geom[c] - man["geom_stats"][c][0]) / man["geom_stats"][c][1]
         for c in man["feat_cols"]]
    fv = torch.tensor([v], dtype=torch.float32).to(DEVICE)

    out = []
    for net, norm in nets:
        r1, _ = net(x, fv); r2, _ = net(torch.flip(x, dims=[3]), fv)
        z = ((r1 + r2) / 2)[0, 0].item()
        out.append(z * norm["weight"][1] + norm["weight"][0])
    # the spread across folds is a usable confidence signal
    return float(np.mean(out)), float(np.std(out)), out


man, nets = load_ensemble()
row = dfx.iloc[0]
img = Image.open(os.path.join(DATA_FULL, row["path"]))
raw_geom = {c: float(df.loc[row.name, c]) for c in man["feat_cols"]}
w, s, each = predict_weight(img, raw_geom, man, nets)
print(f"actual {row['weight']:.1f} kg | ensemble {w:.1f} +/- {s:.1f} kg")
print("per fold:", [round(v, 1) for v in each])
""")

# =========================================================================
md("""
## Honest expectations

Rounds 1–4 bound what this run can achieve:

- Round 1 reached MAE 74.01 at a 40-epoch budget with curves still falling.
  Finishing the schedule, dropping `RandomErasing`, adding EMA and averaging
  five folds should land somewhere around **65–72 kg**.
- Round 4 says another 112 animals would buy roughly 4.6 kg.
- Round 3 says weight error is 3× the linear-dimension error, and 7% linear
  error is what a photo without a scale reference supports.

Stacking those: **MAPE will land near 18–21%, against a target of 8%.** The run
is still worth doing — it produces the saved model the service needs, and a
solid baseline to measure the next change against — but it will not close the
gap.

The gap closes when a scale reference enters the frame. An ArUco marker printed
on A4, held in the plane of the animal's body, takes linear error to well under
1% and weight error with it. Everything in `ml/` is set up to be re-run the day
that data exists.
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
out = Path(__file__).parent / "Final_Training.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
