"""Generates ml/Multitask_Weight_Experiment.ipynb from the cell sources below.

Kept as a script so the notebook stays reviewable in git as plain Python
rather than as a wall of escaped JSON.
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# =========================================================================
md(r"""
# Multi-task weight estimation — crop vs no-crop

Replaces the single-output regression in `Predition_Weight.ipynb`, which scored
**MAE 79.91 kg / RMSE 92.66 kg** — worse than always answering with the mean
weight (R² below zero).

Two changes are tested here, independently:

**1. Multi-task.** Predict `weight`, `height`, `length`, `age`, `breed` and the
scale-free ratio `L/H` from one shared encoder. With ~220 distinct images and a
large backbone, extra supervision per image is the cheapest regularisation
available. Note which targets are actually recoverable from a photo:

| target | recoverable? | why |
|---|---|---|
| breed | yes | colour, markings, hump, horns |
| `ratio_lh` | yes | a **ratio** — survives any crop or resize |
| age | partly | body proportions change with maturity |
| height, length (cm) | **no** | absolute units, same scale problem as weight |

Height and length still earn their place as auxiliary targets: they push the
encoder toward shape rather than memorised texture.

**2. Crop vs no crop.** A full frame carries scale *implicitly* — a heavier cow
fills more of the frame, provided shooting distance is roughly consistent.
Cropping to the bounding box throws that away.

| | no geometry | + geometry (bbox in original px) |
|---|---|---|
| full frame | **A** | **D** |
| tight crop | **B** | **E** |
| crop + 20% context | **C** | **F** |

**If A beats B, the bottleneck is scale, not background clutter.** That single
comparison is the point of this notebook.

Run the cells in order. Each one is self-contained enough to paste into an
existing notebook.
""")

# =========================================================================
md("## Block 1 — Setup")

code(r"""
!pip -q install ultralytics

import os, re, json, copy, math, warnings, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# ---- paths: edit these two if yours differ -------------------------------
DRIVE      = "/content/drive/MyDrive/KSU/Research/SmartWeight"
DATA_FULL  = "/content/cow_dataset"          # UNCROPPED images (needed for A/C/D/F)
COW_CSV    = f"{DRIVE}/cow.csv"
DATASET_CSV = f"{DATA_FULL}/cow_weight_dataset.csv"

# ---- experiment budget ---------------------------------------------------
IMG_SIZE   = 320
FOLDS      = 3      # screening; raise to 5 for the final number
EPOCHS     = 40     # screening; raise to 80 for the final number
BATCH_SIZE = 16
LR         = 3e-4
PATIENCE   = 10
SEED       = 42

torch.manual_seed(SEED); np.random.seed(SEED)
""")

# =========================================================================
md("""
## Block 2 — Mount Drive and unzip the **uncropped** dataset

The no-crop variants need the original images, not `cow_dataset_cropped`.
""")

code(r"""
from google.colab import drive
drive.mount('/content/drive')

import zipfile
if not os.path.exists(DATASET_CSV):
    z = f"{DRIVE}/cow_dataset.zip"
    print("unzipping", z)
    os.makedirs(DATA_FULL, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA_FULL)

print(sorted(os.listdir(DATA_FULL)))
assert os.path.exists(DATASET_CSV), f"missing {DATASET_CSV}"
""")

# =========================================================================
md("""
## Block 3 — Merge labels, group by animal

`normalise_id` strips Roboflow's `-1` augmentation suffix so that every photo of
one cow shares a group id. Without this, augmented copies of the same animal end
up on both sides of a split and every metric is optimistic.
""")

code(r"""
def normalise_id(raw):
    s = str(raw).strip().upper().split("_")[0].split("-")[0]
    if s.startswith("C"):
        s = s[1:]
    return f"C{int(s):03d}" if s.isdigit() else str(raw).strip().upper()

def parse_age_years(v):
    if pd.isna(v): return np.nan
    m = re.match(r"^([\d.]+)\s*([YM]?)$", str(v).strip().upper())
    if not m: return np.nan
    n = float(m.group(1))
    return n / 12.0 if m.group(2) == "M" else n

# --- measurements, one row per animal ---
cow = pd.read_csv(COW_CSV)
cow = cow[[c for c in cow.columns if not re.fullmatch(r"_.*_", str(c))]]
cow["animal_id"] = cow["ID"].map(normalise_id)
cow["age_years"] = cow["Age"].map(parse_age_years)
for c in ("Weight", "Height", "L"):
    cow[c] = pd.to_numeric(cow[c], errors="coerce")
cow = cow.dropna(subset=["Weight"]).drop_duplicates("animal_id")
cow["ratio_lh"] = cow["L"] / cow["Height"]
cow["h2l"] = cow["Height"]**2 * cow["L"] / 10_000

# --- images ---
img = pd.read_csv(DATASET_CSV)
img["animal_id"] = img["ID"].map(normalise_id)

df = img.merge(
    cow[["animal_id", "Height", "L", "age_years", "ratio_lh", "h2l", "Category"]],
    on="animal_id", how="left",
)
df["breed_id"] = LabelEncoder().fit_transform(df["Category"].fillna("unknown"))
N_BREEDS = df["breed_id"].nunique()

print(f"images {len(df)} | animals {df['animal_id'].nunique()} | breeds {N_BREEDS}")
print(f"weight {df['weight'].min():.0f}-{df['weight'].max():.0f} kg "
      f"(mean {df['weight'].mean():.1f}, sd {df['weight'].std():.1f})")
print("\nmissing per aux target:")
print(df[["Height", "L", "age_years", "ratio_lh"]].isna().sum())
print("\nphotos per animal:")
print(df.groupby("animal_id").size().describe()[["min", "50%", "max"]])
""")

# =========================================================================
md("""
## Block 4 — The bar to clear

Before any GPU work: how much of the weight is explained by the tape
measurements alone? Any image model that cannot beat this has learned nothing
useful, and the gap tells you whether the problem is the architecture or the
inputs.
""")

code(r"""
def report(y, p, label=""):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_tot = ((y - y.mean())**2).sum()
    m = dict(n=len(y),
             MAE=np.abs(e).mean(),
             RMSE=np.sqrt((e**2).mean()),
             MAPE=np.abs(e / y).mean() * 100,
             R2=1 - (e**2).sum() / ss_tot if ss_tot else np.nan,
             bias=e.mean(),
             within10=(np.abs(e / y) <= .10).mean() * 100)
    if label:
        print(f"{label:<34} MAE {m['MAE']:6.2f} | RMSE {m['RMSE']:6.2f} | "
              f"MAPE {m['MAPE']:5.2f}% | R2 {m['R2']:6.3f} | "
              f"bias {m['bias']:+6.1f} | <=10% {m['within10']:5.1f}%")
    return m

_a = cow.dropna(subset=["Height", "L"])
_y = _a["Weight"].to_numpy(float)
_cv = KFold(5, shuffle=True, random_state=SEED)
_ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))

print(f"--- tabular ceiling, {len(_a)} animals, 5-fold CV ---")
for name, cols in [("Height only", ["Height"]),
                   ("L only", ["L"]),
                   ("Height + L", ["Height", "L"]),
                   ("Schaeffer h^2*L", ["h2l"]),
                   ("H + L + h2l + age", ["Height", "L", "h2l", "age_years"])]:
    X = _a[cols].to_numpy(float)
    for j in range(X.shape[1]):                       # fill per column, not globally
        X[np.isnan(X[:, j]), j] = np.nanmean(X[:, j])
    report(_y, cross_val_predict(_ridge(), X, _y, cv=_cv), name)

report(_y, np.full_like(_y, _y.mean()), "predict-the-mean")
print(f"{'notebook CNN (image only)':<34} MAE  79.91 | RMSE  92.66 | "
      f"MAPE ~21%  | R2 < 0")

# Fitted physics rule, reused later to turn predicted H,L into kilograms.
_r = _ridge().fit(_a[["h2l"]].to_numpy(float), _y)
PHYS_A = _r[-1].coef_[0] / _a["h2l"].std(ddof=0)
PHYS_B = _r[-1].intercept_ - PHYS_A * _a["h2l"].mean()
print(f"\nfitted rule:  W ~= {PHYS_A:.3f} * (H^2*L/10000) + {PHYS_B:.1f}")
""")

# =========================================================================
md("""
## Block 5 — Cache one bounding box per image

Run the detector **once** and store boxes in original-image pixels. Every crop
variant then reuses the same boxes, so the comparison isolates the cropping
choice and nothing else.

`bbox_area_ratio` — how much of the frame the animal fills — is the scale cue
that cropping destroys. It is kept here so variants D/E/F can hand it back to
the model explicitly.
""")

code(r"""
BBOX_CACHE = "/content/bboxes.csv"

if os.path.exists(BBOX_CACHE):
    boxes = pd.read_csv(BBOX_CACHE)
    print("loaded cached boxes:", len(boxes))
else:
    from ultralytics import YOLO
    import cv2
    det = YOLO("yolo11n.pt")          # COCO; class 19 is 'cow'
    rows = []
    for i, r in df.iterrows():
        p = os.path.join(DATA_FULL, r["path"])
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        res = det.predict(im, verbose=False, conf=0.25)[0]
        x1 = y1 = 0; x2, y2 = W, H; found = False
        if res.boxes is not None and len(res.boxes):
            xy = res.boxes.xyxy.cpu().numpy()
            a = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
            x1, y1, x2, y2 = xy[int(a.argmax())]
            found = True
        rows.append(dict(path=r["path"], img_w=W, img_h=H, found=found,
                         x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")
    boxes = pd.DataFrame(rows)
    boxes.to_csv(BBOX_CACHE, index=False)
    print("detected on", boxes["found"].sum(), "/", len(boxes))

df = df.drop(columns=[c for c in boxes.columns if c != "path"], errors="ignore")
df = df.merge(boxes, on="path", how="left")
df = df.dropna(subset=["img_w"]).reset_index(drop=True)

df["bbox_w"]  = df["x2"] - df["x1"]
df["bbox_h"]  = df["y2"] - df["y1"]
df["bbox_area_ratio"] = (df["bbox_w"] * df["bbox_h"]) / (df["img_w"] * df["img_h"])
df["bbox_aspect"]     = df["bbox_w"] / df["bbox_h"]
df["bbox_w_rel"]      = df["bbox_w"] / df["img_w"]
df["bbox_h_rel"]      = df["bbox_h"] / df["img_h"]

GEOM_COLS = ["bbox_w", "bbox_h", "bbox_area_ratio", "bbox_aspect",
             "bbox_w_rel", "bbox_h_rel", "img_w", "img_h"]

print(f"\nrows {len(df)}")
print(df[["bbox_area_ratio", "bbox_aspect"]].describe().loc[["mean", "std", "min", "max"]])
print("\ncorr(bbox_area_ratio, weight) =",
      round(df["bbox_area_ratio"].corr(df["weight"]), 3),
      " <- if this is far from 0, full-frame images do carry scale")
""")

# =========================================================================
md("""
## Block 6 — Dataset with switchable cropping

`LetterboxPad` resizes the long edge and pads the rest. `Resize((n,n))` squashes
to a square and destroys the length-to-height ratio, which is one of the two
strongest shape cues for weight — that alone is worth several kilograms of MAE.
""")

code(r"""
class LetterboxPad:
    def __init__(self, size, fill=114):
        self.size, self.fill = size, fill
    def __call__(self, im):
        w, h = im.size
        s = self.size / max(w, h)
        im = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.BILINEAR)
        c = Image.new("RGB", (self.size, self.size), (self.fill,)*3)
        c.paste(im, ((self.size-im.size[0])//2, (self.size-im.size[1])//2))
        return c

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    LetterboxPad(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
    transforms.ToTensor(), NORM,
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
])
EVAL_TF = transforms.Compose([LetterboxPad(IMG_SIZE), transforms.ToTensor(), NORM])

REG_TARGETS = ["weight", "Height", "L", "age_years", "ratio_lh"]

class CowDataset(Dataset):
    # crop_mode: 'none' | 'tight' | 'context'
    def __init__(self, sub, root, tf, crop_mode="tight", ctx=0.20,
                 feat_cols=(), norm=None):
        self.df = sub.reset_index(drop=True)
        self.root, self.tf = Path(root), tf
        self.crop_mode, self.ctx = crop_mode, ctx
        self.feat_cols = list(feat_cols)
        self.norm = norm                      # per-target (mean, std) from train fold

    def __len__(self):
        return len(self.df)

    def _crop(self, im, r):
        if self.crop_mode == "none":
            return im
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if self.crop_mode == "context":
            dw, dh = (x2-x1)*self.ctx/2, (y2-y1)*self.ctx/2
            x1, y1, x2, y2 = x1-dw, y1-dh, x2+dw, y2+dh
        W, H = im.size
        box = (max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return im
        return im.crop(box)

    def __getitem__(self, i):
        r = self.df.loc[i]
        im = Image.open(self.root / r["path"]).convert("RGB")
        x = self.tf(self._crop(im, r))

        feats = (torch.tensor(r[self.feat_cols].to_numpy(np.float32))
                 if self.feat_cols else torch.zeros(0))

        y, mask = [], []
        for t in REG_TARGETS:
            v = r[t]
            ok = not pd.isna(v)
            mu, sd = self.norm[t]
            y.append((float(v) - mu) / sd if ok else 0.0)   # standardised units
            mask.append(ok)
        return (x, feats,
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(int(r["breed_id"]), dtype=torch.long))
""")

# =========================================================================
md("""
## Block 7 — Multi-task network

One shared encoder, one head per target. Task losses are balanced by learned
uncertainty (Kendall & Gal): each task gets a trainable `log σ²`, so a target
the model cannot fit — `Height` in centimetres, on cropped images — quietly
down-weights itself instead of poisoning the shared features.

The weight head's bias starts at zero **in standardised units**, i.e. at the
mean weight. The original notebook started it near zero kilograms, which is why
its first epoch reported RMSE 406 — exactly the mean weight.
""")

code(r"""
class MultiTaskNet(nn.Module):
    def __init__(self, n_feats=0, n_breeds=2, n_reg=5, dropout=0.4, multitask=True):
        super().__init__()
        bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        d_img = bb.classifier[1].in_features
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.multitask = multitask

        self.tab = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, 64), nn.SiLU(), nn.Dropout(dropout/2),
            nn.Linear(64, 64), nn.SiLU(),
        ) if n_feats else None

        d = d_img + (64 if n_feats else 0)
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 256), nn.SiLU())
        self.reg   = nn.Linear(256, n_reg if multitask else 1)
        self.breed = nn.Linear(256, n_breeds) if multitask else None

        nn.init.zeros_(self.reg.weight); nn.init.zeros_(self.reg.bias)
        # one log-variance per regression target, plus one for breed
        self.log_var = nn.Parameter(torch.zeros((n_reg if multitask else 1) + 1))

    def forward(self, img, feats):
        z = self.backbone(img)
        if self.tab is not None:
            z = torch.cat([z, self.tab(feats)], 1)
        z = self.trunk(z)
        return self.reg(z), (self.breed(z) if self.breed is not None else None)


def multitask_loss(reg_out, breed_out, y, mask, breed, log_var, multitask):
    # Uncertainty-weighted sum over whatever targets are present.
    total, parts = 0.0, {}
    n = reg_out.shape[1]
    for i in range(n):
        m = mask[:, i]
        if m.sum() == 0:
            continue
        li = F.smooth_l1_loss(reg_out[m, i], y[m, i])
        s = log_var[i]
        total = total + torch.exp(-s) * li + s
        parts[REG_TARGETS[i]] = float(li)
    if multitask and breed_out is not None:
        lb = F.cross_entropy(breed_out, breed)
        s = log_var[-1]
        total = total + torch.exp(-s) * lb + s
        parts["breed"] = float(lb)
    return total, parts
""")

# =========================================================================
md("""
## Block 8 — Train and evaluate one fold

`CosineAnnealingLR` rather than `OneCycleLR`. The original notebook paired
`OneCycleLR(epochs=400)` with early stopping at epoch 187, so the learning rate
was still near its peak (0.00043) and never annealed — the saved weights came
from a phase that had not converged.
""")

code(r"""
def run_fold(tr, va, *, crop_mode, feat_cols, multitask, root=DATA_FULL,
             epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    def _stats(t):
        mu, sd = tr[t].mean(), tr[t].std()
        # nan is truthy, so `float(sd) or 1.0` would let a nan through
        mu = 0.0 if pd.isna(mu) else float(mu)
        sd = float(sd) if (not pd.isna(sd) and sd > 1e-6) else 1.0
        return mu, sd
    norm = {t: _stats(t) for t in REG_TARGETS}

    mk = lambda sub, tf: CowDataset(sub, root, tf, crop_mode=crop_mode,
                                    feat_cols=feat_cols, norm=norm)
    dl_tr = DataLoader(mk(tr, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=2, drop_last=True)
    dl_va = DataLoader(mk(va, EVAL_TF), batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=2)

    model = MultiTaskNet(len(feat_cols), N_BREEDS, len(REG_TARGETS),
                         multitask=multitask).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    wmu, wsd = norm["weight"]
    best, best_wts, stale = 1e9, copy.deepcopy(model.state_dict()), 0

    for ep in range(epochs):
        model.train()
        for x, f, y, m, b in dl_tr:
            x, f, y, m, b = [t.to(DEVICE) for t in (x, f, y, m, b)]
            opt.zero_grad()
            ro, bo = model(x, f)
            loss, _ = multitask_loss(ro, bo, y, m, b, model.log_var, multitask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval(); P, T = [], []
        with torch.no_grad():
            for x, f, y, m, b in dl_va:
                ro, _ = model(x.to(DEVICE), f.to(DEVICE))
                P.append(ro[:, 0].cpu().numpy()); T.append(y[:, 0].numpy())
        mae = float(np.abs((np.concatenate(P) - np.concatenate(T)) * wsd).mean())
        if mae < best - 0.1:
            best, best_wts, stale = mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if ep % 10 == 0:
            print(f"      ep {ep+1:>3}  val MAE {mae:6.2f}  best {best:6.2f}")
        if stale >= PATIENCE:
            print(f"      early stop at ep {ep+1}")
            break

    model.load_state_dict(best_wts)

    # horizontal-flip TTA: free variance reduction on small validation folds
    model.eval(); out = []
    with torch.no_grad():
        for x, f, y, m, b in dl_va:
            x, f = x.to(DEVICE), f.to(DEVICE)
            r1, _ = model(x, f)
            r2, _ = model(torch.flip(x, dims=[3]), f)
            out.append(((r1 + r2) / 2).cpu().numpy())
    pred_std = np.concatenate(out)

    res = {}
    for i, t in enumerate(REG_TARGETS[:pred_std.shape[1]]):
        mu, sd = norm[t]
        res[t] = pred_std[:, i] * sd + mu
    return res
""")

# =========================================================================
md("""
## Block 9 — Run the matrix

Six variants under identical folds, seed and epoch budget, so any difference is
attributable to the cropping and geometry choice alone.

Screening defaults are 3 folds × 40 epochs. Re-run the winner at 5 × 80.
""")

code(r"""
EXPERIMENTS = [
    dict(name="A full-frame",            crop="none",    geom=False, multitask=True),
    dict(name="B tight-crop",            crop="tight",   geom=False, multitask=True),
    dict(name="C crop+20% ctx",          crop="context", geom=False, multitask=True),
    dict(name="D full-frame + geom",     crop="none",    geom=True,  multitask=True),
    dict(name="E tight-crop + geom",     crop="tight",   geom=True,  multitask=True),
    dict(name="F crop+ctx + geom",       crop="context", geom=True,  multitask=True),
    # ablation: same as E but weight is the only target
    dict(name="E- single-task",          crop="tight",   geom=True,  multitask=False),
]

# Standardise geometry once, globally. On ~700 rows the scale drift from doing
# it per fold hurts more than the mild leakage it would avoid.
dfx = df.copy()
dfx[GEOM_COLS] = ((dfx[GEOM_COLS] - dfx[GEOM_COLS].mean())
                  / (dfx[GEOM_COLS].std() + 1e-8))

gkf = GroupKFold(n_splits=FOLDS)
splits = list(gkf.split(dfx, groups=dfx["animal_id"]))
y_all = dfx["weight"].to_numpy(float)

results, oof_store = [], {}

for exp in EXPERIMENTS:
    print(f"\n{'='*72}\n{exp['name']}\n{'='*72}")
    feat_cols = GEOM_COLS if exp["geom"] else []
    oof = {t: np.full(len(dfx), np.nan) for t in REG_TARGETS}

    for k, (tr_i, va_i) in enumerate(splits, 1):
        print(f"   fold {k}/{FOLDS}  train {len(tr_i)} / val {len(va_i)} "
              f"({dfx.iloc[va_i]['animal_id'].nunique()} animals)")
        r = run_fold(dfx.iloc[tr_i], dfx.iloc[va_i],
                     crop_mode=exp["crop"], feat_cols=feat_cols,
                     multitask=exp["multitask"], seed=SEED + k)
        for t, v in r.items():
            oof[t][va_i] = v

    m = report(y_all, oof["weight"], f"{exp['name']:<22} [direct]")

    # Weight implied by the predicted height and length, via the Block-4 rule.
    row = dict(variant=exp["name"], crop=exp["crop"], geom=exp["geom"],
               multitask=exp["multitask"], **m)
    if exp["multitask"] and not np.isnan(oof["Height"]).all():
        w_phys = PHYS_A * (oof["Height"]**2 * oof["L"] / 10_000) + PHYS_B
        mp = report(y_all, w_phys, f"{exp['name']:<22} [physics H,L]")
        w_blend = (oof["weight"] + w_phys) / 2
        mb = report(y_all, w_blend, f"{exp['name']:<22} [blend]")
        row.update(MAE_phys=mp["MAE"], MAE_blend=mb["MAE"])
        oof_store[exp["name"] + "|phys"] = w_phys

    results.append(row)
    oof_store[exp["name"]] = oof["weight"]

res_df = pd.DataFrame(results).sort_values("MAE")
print("\n" + "="*72)
display(res_df[["variant", "MAE", "RMSE", "MAPE", "R2", "bias", "within10"]]
        .round(2).reset_index(drop=True))
""")

# =========================================================================
md("""
## Block 10 — Read the result

The decisive comparison is **A vs B**: identical in every respect except that B
throws away the frame.
""")

code(r"""
import matplotlib.pyplot as plt

base_rmse = float(np.sqrt(((y_all - y_all.mean())**2).mean()))
print(f"predict-the-mean RMSE ....... {base_rmse:6.2f} kg   <- beat this or the model is useless")
print(f"notebook CNN (image only) ... {92.66:6.2f} kg")
print(f"best variant here ........... {res_df.iloc[0]['RMSE']:6.2f} kg  ({res_df.iloc[0]['variant']})")

try:
    a = res_df.set_index("variant").loc["A full-frame", "MAE"]
    b = res_df.set_index("variant").loc["B tight-crop", "MAE"]
    print(f"\nA full-frame  MAE {a:.2f}   vs   B tight-crop  MAE {b:.2f}")
    print("=> cropping destroys scale; keep the frame or feed geometry back in."
          if a < b else
          "=> cropping helps here, so background clutter was the larger problem.")
except KeyError:
    pass

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
best = res_df.iloc[0]["variant"]
p = oof_store[best]

ax[0].barh(res_df["variant"][::-1], res_df["MAE"][::-1])
ax[0].axvline(79.91, color="crimson", ls="--", label="notebook CNN")
ax[0].axvline(np.abs(y_all - y_all.mean()).mean(), color="gray", ls=":", label="mean baseline")
ax[0].set_xlabel("MAE (kg)"); ax[0].legend(); ax[0].set_title("variants")

ax[1].scatter(y_all, p, alpha=.4)
lim = [y_all.min(), y_all.max()]
ax[1].plot(lim, lim, "r--"); ax[1].set_xlabel("actual (kg)")
ax[1].set_ylabel("predicted (kg)"); ax[1].set_title(f"{best}: actual vs predicted")

d = p - y_all; bias, sd = d.mean(), d.std(ddof=1)
ax[2].scatter((y_all + p) / 2, d, alpha=.4)
ax[2].axhline(bias, color="crimson", label=f"bias {bias:+.1f}")
ax[2].axhline(bias + 1.96*sd, color="gray", ls="--")
ax[2].axhline(bias - 1.96*sd, color="gray", ls="--")
ax[2].set_xlabel("mean of actual & predicted"); ax[2].set_ylabel("predicted - actual")
ax[2].set_title("Bland-Altman"); ax[2].legend()
plt.tight_layout(); plt.show()

# Per-animal averaging - what the app should do with several photos.
pa = pd.DataFrame({"a": dfx["animal_id"], "y": y_all, "p": p}).groupby("a").mean()
report(pa["y"], pa["p"], "best variant, per-animal")
""")

# =========================================================================
md("""
## Block 11 — Which auxiliary targets were actually learnable

The prediction made at the top of this notebook is testable: ratios and breed
should be learnable from a photo, centimetres should not. A low R² on `Height`
and `L` is not a bug — it is the scale problem showing up directly in the
auxiliary heads, and it is the reason the ceiling here is what it is.
""")

code(r"""
best = res_df[res_df.multitask].iloc[0]["variant"]
feat_cols = GEOM_COLS if res_df.set_index("variant").loc[best, "geom"] else []
crop = res_df.set_index("variant").loc[best, "crop"]

oof_aux = {t: np.full(len(dfx), np.nan) for t in REG_TARGETS}
for k, (tr_i, va_i) in enumerate(splits, 1):
    r = run_fold(dfx.iloc[tr_i], dfx.iloc[va_i], crop_mode=crop,
                 feat_cols=feat_cols, multitask=True, seed=SEED + k)
    for t, v in r.items():
        oof_aux[t][va_i] = v

print(f"auxiliary targets, variant {best}\n")
for t in REG_TARGETS:
    ok = dfx[t].notna().to_numpy() & ~np.isnan(oof_aux[t])
    if ok.sum() < 10:
        continue
    report(dfx[t].to_numpy(float)[ok], oof_aux[t][ok], f"  {t}")

print("\nExpected pattern: ratio_lh learns well (scale-free), "
      "Height and L do not (absolute units).")
""")

# =========================================================================
md("""
## What to do with the outcome

| outcome | reading | next step |
|---|---|---|
| A (full frame) clearly beats B (tight crop) | scale is the binding constraint | stop cropping, or always feed geometry back in |
| D/E/F beat their no-geometry twins | explicit geometry restores what the resize removed | keep geometry as a model input |
| multi-task beats `E- single-task` | auxiliary supervision is regularising a too-large model | keep all heads |
| `ratio_lh` fits well but `Height`/`L` do not | confirms centimetres are unrecoverable from a bare photo | put an ArUco marker in the capture protocol |
| nothing beats the tabular ceiling from Block 4 | the image path is not the bottleneck | predict `Height`/`L` from images, then apply the fitted rule |

Whatever wins here, the honest ceiling stands: **absolute size cannot be
recovered from a single photo with no object of known size in frame.** Ratios
survive any crop or resize; centimetres do not. An ArUco marker printed on A4
costs nothing and removes the limit entirely.
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
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).parent / "Multitask_Weight_Experiment.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
