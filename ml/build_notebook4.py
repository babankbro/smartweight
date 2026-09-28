"""Generates ml/CowBuff_Weight_Training.ipynb - one weight model for cattle AND buffalo.

Same workflow as `Predition_Weight_V3 (1).ipynb` cells 35-39 / ai-api/app/weight.py:

    photo (640x640) -> YOLO11n box geometry
                    -> 4 fine-tuned multi-task backbones, flip-averaged features
                    -> SVR | Ridge | multi-task MLP | the nets' own weight output
                    -> weighted mean

Changes from V3:
  - cow + buffalo pooled into one training set (StratifiedGroupKFold by animal, species balanced)
  - fine-tuning sized for a V100: AMP fp16, 150 epochs, warmup + cosine, EMA, per-group LR
  - leakage fixed: backbones are re-fine-tuned inside every outer fold, so the
    out-of-fold numbers are honest (V3's MAE 28 was not - see ml/README.md)
  - output drops into ai-api/models/predict_weight/ unchanged
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))


# =========================================================================
md(r"""
# ประเมินน้ำหนักโค + กระบือ — โมเดลเดียว (V100)

Workflow เดิมของ V3 / `ai-api/app/weight.py`:

```
ภาพ 640×640 ─┬─ YOLO11n → เรขาคณิต bbox 8 ค่า
             └─ 4 backbone (fine-tune multi-task, พลิกซ้ายขวาเฉลี่ย) → ฟีเจอร์ 3208 มิติ
                     → SVR | Ridge | MLP multi-task | หัวของ backbone เอง → ถ่วงน้ำหนักรวม
```

| เปลี่ยนจาก V3 | เหตุผล |
|---|---|
| รวมโค + กระบือ เทรน backbone ชุดเดียว | ข้อมูลเพิ่ม ~2 เท่า · แบ่ง fold ตามตัวสัตว์และสมดุลชนิด |
| fine-tune 150 epoch · AMP fp16 · warmup+cosine · EMA · LR แยก backbone/หัว | ใช้ V100 เต็มที่ และไม่ต้องใช้ early stop ที่อาศัยชุด val |
| **fine-tune backbone ใหม่ในทุก fold** | V3 เทรน backbone บน 7/8 ของสัตว์แล้ว CV หัวบนสัตว์ชุดเดิม → MAE 28 ที่ได้คือ leak |
| เพิ่มหัวที่ 4 `direct` + น้ำหนักรวมจาก NNLS | ใช้เมื่อ OOF ดีกว่าการเฉลี่ยเท่ากันเท่านั้น |
| augmentation ระดับกลาง | flip · affine เล็กน้อย · สี · blur เบา ๆ — ไม่ crop เพราะขนาดในเฟรมคือสัญญาณน้ำหนัก |

**ข้อมูลที่ต้องมีใน `ROOT`:** `cow yolo dataset.zip`, `buff yolo dataset.zip`, `cow.csv`, `buff.csv`
(`buff.csv` คอลัมน์แบบเดียวกับ `cow.csv`: `ID` เช่น `B001`, `Weight`, และถ้ามี `Height`, `L`, `Age`, `Category`)

**เวลาโดยประมาณบน V100:** 4 backbone × (5 fold + 1 ชุดเต็ม) ≈ 5–7 ชม. ·
ทุกขั้นเซฟลง `OUT` และข้ามเมื่อมีไฟล์แล้ว — runtime หลุดก็รันต่อได้
""")

# =========================================================================
code(r"""
!pip -q install timm==1.0.30 ultralytics==8.4.153 scikit-learn==1.9.1   # = ai-api image, so the saved heads load there

import os, re, gc, copy, json, math, time, zipfile, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm, joblib, sklearn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
assert torch.cuda.is_available(), "needs a GPU runtime"
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| timm", timm.__version__,
      "| sklearn", sklearn.__version__)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

# ---- paths ---------------------------------------------------------------
ROOT = Path("/content/drive/MyDrive/KSU/Research/SmartWeight")   # zips + label csvs
WORK = Path("/content/work")                                       # local disk: unzipped images
OUT  = ROOT / "weight_cowbuff"                                     # everything worth keeping
SPECIES = {
    "cow":     dict(prefix="C", zip="cow yolo dataset.zip",  csv="cow.csv"),
    "buffalo": dict(prefix="B", zip="buff yolo dataset.zip", csv="buff.csv"),
}

# ---- training budget (V100) ------------------------------------------------
# Keys and order are part of the serving contract (ai-api/app/weight.py BACKBONE_KEYS).
BACKBONES = {
    "convnext_t": "convnext_tiny.fb_in22k",
    "effv2_s":    "tf_efficientnetv2_s.in21k",
    "dinov2_s":   "vit_small_patch14_dinov2.lvd142m",
    "clip_b16":   "vit_base_patch16_clip_224.openai",
}
LR_BB   = {"convnext_t": 1e-4, "effv2_s": 2e-4, "dinov2_s": 5e-5, "clip_b16": 3e-5}
LR_HEAD = 1e-3
IMG, BATCH, EPOCHS, WARMUP = 336, 32, 150, 5    # IMG = FEAT_IMG in ai-api
WD, DROP_PATH, EMA_DECAY = 0.05, 0.1, 0.998
FOLDS, MLP_EPOCHS, SEED = 5, 400, 42
NW = min(8, os.cpu_count() or 2)

REG_TARGETS = ["weight", "Height_y", "L_y", "age_years", "ratio_lh"]   # order = serving
GEOM_COLS = ["bbox_w", "bbox_h", "bbox_area_ratio", "bbox_aspect",
             "bbox_w_rel", "bbox_h_rel", "img_w", "img_h"]
COW_FORMULA = (5.854, -595.98)     # most cow.csv weights are exactly this function of L

for p in (WORK, OUT, OUT / "cv"):
    p.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
""")

# =========================================================================
md("## 1. รวมข้อมูลโค + กระบือ")

code(r"""
def norm_id(raw, prefix):
    # 'c0020-1_jpg.rf.x.jpg' / '110_jpg...' / 'B005' -> 'C020' / 'C110' / 'B005'
    s = re.sub(r"^[A-Za-z]+", "", str(raw).strip().split("_")[0].split("-")[0])
    return f"{prefix}{int(s):03d}" if s.isdigit() else None

def parse_age(v):
    m = re.match(r"^([\d.]+)\s*([YM]?)$", str(v).strip().upper())
    return (float(m.group(1)) / (12 if m.group(2) == "M" else 1)) if m else np.nan

def load_species(name, cfg):
    lab = pd.read_csv(ROOT / cfg["csv"], encoding="utf-8-sig")
    lab = lab[[c for c in lab.columns if not re.fullmatch(r"_.*_", str(c))]]
    lab["animal_id"] = lab["ID"].map(lambda v: norm_id(v, cfg["prefix"]))
    for c in ("Weight", "Height", "L"):
        lab[c] = pd.to_numeric(lab[c], errors="coerce") if c in lab else np.nan
    lab["age_years"] = lab["Age"].map(parse_age) if "Age" in lab else np.nan
    lab["Category"] = lab["Category"].fillna(name) if "Category" in lab else name
    lab = lab[lab.Weight > 0].dropna(subset=["animal_id"]).drop_duplicates("animal_id")

    root = WORK / name
    if not root.exists():
        with zipfile.ZipFile(ROOT / cfg["zip"]) as z:
            z.extractall(root)
    imgs = pd.DataFrame([dict(path=str(p), animal_id=norm_id(p.name, cfg["prefix"]))
                         for p in sorted(root.glob("*/images/*"))
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    d = imgs.merge(lab[["animal_id", "Weight", "Height", "L", "age_years", "Category"]],
                   on="animal_id", how="inner")
    d["species"] = name
    print(f"{name:<8} labelled animals {len(lab):>4} | images {len(d):>4} "
          f"({d.animal_id.nunique()} animals) | images without a label {len(imgs) - len(d)}")
    return d

df = pd.concat([load_species(k, v) for k, v in SPECIES.items()], ignore_index=True)
df = df.rename(columns={"Weight": "weight", "Height": "Height_y", "L": "L_y"})
df["ratio_lh"] = df.L_y / df.Height_y
df["label_is_formula"] = (df.species == "cow") & \
    ((df.weight - (COW_FORMULA[0] * df.L_y + COW_FORMULA[1])).abs() < 0.01)
df["Category"] = df.Category.astype("category")
BREEDS = list(df.Category.cat.categories)
N_BREEDS = len(BREEDS)

y = df.weight.to_numpy(float)
Y = df[REG_TARGETS].to_numpy(float)                   # NaN = target missing for that animal
MASK = ~np.isnan(Y)
BREED = df.Category.cat.codes.to_numpy(np.int64)
groups = df.animal_id.to_numpy()

display(df.groupby("species").agg(images=("path", "size"), animals=("animal_id", "nunique"),
                                  kg_mean=("weight", "mean"), kg_sd=("weight", "std"),
                                  kg_min=("weight", "min"), kg_max=("weight", "max"),
                                  formula_labels=("label_is_formula", "mean")).round(2))
print("breed classes:", BREEDS)
""")

# =========================================================================
md("## 2. เรขาคณิตจาก YOLO11n และภาพที่ย่อไว้ใน RAM\nเหมือน `FeatureExtractor.geometry` ใน ai-api ทุกประการ (กล่องที่ใหญ่ที่สุดของทุก class)")

code(r"""
geo_cache = OUT / "geometry.csv"
if geo_cache.exists():
    geo = pd.read_csv(geo_cache)
else:
    from ultralytics import YOLO
    det, rows = YOLO("yolo11n.pt"), []
    for p in df.path:
        im = Image.open(p).convert("RGB").resize((640, 640), Image.BILINEAR)
        r = det.predict(np.asarray(im)[:, :, ::-1], conf=0.25, verbose=False)[0]
        x1 = y1 = 0.0; x2 = y2 = 640.0; found = False
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in xy[((xy[:, 2]-xy[:, 0]) * (xy[:, 3]-xy[:, 1])).argmax()])
            found = True
        bw, bh = x2 - x1, y2 - y1
        rows.append(dict(file=Path(p).name, found=found, bbox_w=bw, bbox_h=bh,
                         bbox_area_ratio=bw * bh / 640**2, bbox_aspect=bw / bh if bh else 0.0,
                         bbox_w_rel=bw / 640, bbox_h_rel=bh / 640, img_w=640.0, img_h=640.0))
    geo = pd.DataFrame(rows)
    geo.to_csv(geo_cache, index=False)

df = df.drop(columns=[c for c in geo.columns if c in df], errors="ignore") \
       .assign(file=df.path.map(lambda p: Path(p).name)).merge(geo, on="file", how="left")
assert len(df) == len(y) and df[GEOM_COLS].notna().all().all(), "stale geometry.csv - delete it"
print("detector found an animal on", int(df.found.sum()), "/", len(df))

# One global standardisation, stored in the manifest; serving divides by (sd + 1e-8) too.
GEOM_STATS = {c: (float(df[c].mean()), float(df[c].std())) for c in GEOM_COLS}
G = np.stack([(df[c] - m) / (s + 1e-8) for c, (m, s) in GEOM_STATS.items()], 1).astype(np.float32)


class LetterboxPad:
    # identical to ai-api/app/weight.py
    def __init__(self, size, fill=114): self.size, self.fill = size, fill
    def __call__(self, im):
        w, h = im.size; s = self.size / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        c = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        c.paste(im, ((self.size - im.size[0]) // 2, (self.size - im.size[1]) // 2))
        return c

# Serving: stretch to 640 -> letterbox to IMG. Done once here; every epoch reads RAM.
_lb = LetterboxPad(IMG)
IMGS = [_lb(Image.open(p).convert("RGB").resize((640, 640), Image.BILINEAR)) for p in df.path]
print(f"{len(IMGS)} images cached at {IMG}px")
""")

# =========================================================================
md(r"""
## 3. Augmentation ระดับกลาง

ไม่มี `RandomResizedCrop` / `RandomErasing` — ขนาดตัวสัตว์ในเฟรมคือสิ่งเดียวที่บอกความใหญ่ได้
scale jitter จึงจำกัดไว้ ±7% ภาพ Roboflow มี rotation/shear/blur (กระบือ) และ crop/brightness (โค) มาแล้วรอบหนึ่ง
""")

code(r"""
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomApply([T.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.93, 1.07),
                                  shear=4, fill=114)], p=0.8),
    T.RandomApply([T.ColorJitter(0.25, 0.25, 0.2, 0.03)], p=0.8),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.15),
    T.RandomGrayscale(0.05),
    T.ToTensor(), NORM,
])
EVAL_TF = T.Compose([T.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, idx, tf, Yn=None):
        self.idx, self.tf = np.asarray(idx), tf
        self.Yn = Yn if Yn is not None else np.zeros_like(Y, dtype=np.float32)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(IMGS[j]), G[j], self.Yn[j], MASK[j], BREED[j]
""")

# =========================================================================
md("## 4. Backbone multi-task + fine-tune")

code(r"""
class WeightNet(nn.Module):
    # Same layout as ai-api/app/weight.py so checkpoints load there with strict=True.
    def __init__(self, backbone, n_feats=len(GEOM_COLS), n_breeds=N_BREEDS,
                 n_reg=len(REG_TARGETS), dropout=0.4, pretrained=False, drop_path=0.0):
        super().__init__()
        dyn = "dinov2" in backbone or "clip" in backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          drop_path_rate=drop_path,
                                          **({"dynamic_img_size": True} if dyn else {}))
        d_img = self.backbone.num_features
        self.tab = nn.Sequential(nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 64), nn.SiLU(),
                                 nn.Dropout(dropout / 2), nn.Linear(64, 64), nn.SiLU())
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_img + 64, 256), nn.SiLU())
        self.reg = nn.Linear(256, n_reg)
        self.breed = nn.Linear(256, n_breeds)
        self.log_var = nn.Parameter(torch.zeros(n_reg + 1))
        nn.init.zeros_(self.reg.weight); nn.init.zeros_(self.reg.bias)   # start at the mean

    def head(self, z_img, feats):
        z = self.trunk(torch.cat([z_img, self.tab(feats)], 1))
        return self.reg(z), self.breed(z)

    def forward(self, img, feats):
        return self.head(self.backbone(img), feats)


def mt_loss(reg, logit, y_, m, b, log_var):
    # weight always at full strength; auxiliaries uncertainty-weighted and masked
    loss = F.smooth_l1_loss(reg[:, 0], y_[:, 0])
    for t in range(1, reg.shape[1]):
        if m[:, t].any():
            loss = loss + torch.exp(-log_var[t]) * F.smooth_l1_loss(reg[m[:, t], t], y_[m[:, t], t]) + log_var[t]
    return loss + torch.exp(-log_var[-1]) * F.cross_entropy(logit, b, label_smoothing=0.1) + log_var[-1]


def target_norm(idx):
    mu, sd = np.nanmean(Y[idx], 0), np.nanstd(Y[idx], 0)
    return np.nan_to_num(mu), np.where(sd > 1e-6, sd, 1.0)


@torch.no_grad()
def extract(net, idx, wmu, wsd, tta=True):
    # -> flip-averaged backbone features, the net's own weight prediction in kg.
    # fp32 on purpose: serving runs fp32 on CPU and the heads see these exact features.
    net.eval()
    feats, preds = [], []
    for x, g, *_ in DataLoader(DS(idx, EVAL_TF), BATCH * 2, num_workers=NW):
        x, g = x.to(DEVICE), g.to(DEVICE)
        fs = [net.backbone(v) for v in ((x, x.flip(3)) if tta else (x,))]
        feats.append((sum(fs) / len(fs)).cpu())
        preds.append((sum(net.head(f, g)[0][:, 0] for f in fs) / len(fs)).cpu())
    return torch.cat(feats).numpy(), torch.cat(preds).numpy() * wsd + wmu


def fit_backbone(key, tr, va=None, seed=SEED, log_every=15):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = target_norm(tr)
    Yn = np.nan_to_num((Y - mu) / sd).astype(np.float32)
    dl = DataLoader(DS(tr, TRAIN_TF, Yn), BATCH, shuffle=True, drop_last=True,
                    num_workers=NW, pin_memory=True, persistent_workers=True)

    net = WeightNet(BACKBONES[key], pretrained=True, drop_path=DROP_PATH).to(DEVICE)
    bb = set(map(id, net.backbone.parameters()))
    opt = torch.optim.AdamW(
        [{"params": list(net.backbone.parameters()), "lr": LR_BB[key]},
         {"params": [p for p in net.parameters() if id(p) not in bb], "lr": LR_HEAD}],
        weight_decay=WD)
    total, warm = EPOCHS * len(dl), WARMUP * len(dl)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    t0, step = time.time(), 0
    for ep in range(EPOCHS):
        net.train(); run = 0.0
        for x, g, yb, m, b in dl:
            x, g, yb, m, b = (t.to(DEVICE, non_blocking=True) for t in (x, g, yb, m, b))
            with torch.autocast("cuda", dtype=torch.float16):
                reg, logit = net(x, g)
            loss = mt_loss(reg.float(), logit.float(), yb, m, b, net.log_var)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            step += 1; run += loss.item()
            d = min(EMA_DECAY, (1 + step) / (10 + step))
            with torch.no_grad():
                for e, p in zip(ema.state_dict().values(), net.state_dict().values()):
                    e.mul_(d).add_(p, alpha=1 - d) if e.is_floating_point() else e.copy_(p)
        if (ep + 1) % log_every == 0 or ep == 0:
            msg = f"    ep {ep+1:>3}/{EPOCHS} loss {run/len(dl):7.4f}"
            if va is not None:
                _, p = extract(ema, va, mu[0], sd[0], tta=False)
                msg += f" | val MAE (EMA) {np.abs(p - y[va]).mean():6.2f}"
            print(msg + f" | {(time.time()-t0)/60:5.1f} min", flush=True)
    del net, opt, dl; gc.collect(); torch.cuda.empty_cache()
    return ema, mu, sd
""")

# =========================================================================
md(r"""
## 5. Nested CV — fine-tune backbone ใหม่ในทุก fold

แต่ละ fold: เทรน backbone บนสัตว์ใน train → ดึงฟีเจอร์ทุกภาพ → เซฟ `cv/fold{k}_{key}.npz`
ตัวเลขที่ได้คือ**ค่าจริงบนสัตว์ที่โมเดลไม่เคยเห็น** ไม่มีการเลือก epoch จากชุด val (EMA + จำนวน epoch คงที่)
""")

code(r"""
split_file = OUT / "splits.json"
if split_file.exists():                               # resume on exactly the same folds
    val_sets = json.load(open(split_file))
    SPLITS = [(np.where(~np.isin(groups, v))[0], np.where(np.isin(groups, v))[0]) for v in val_sets]
else:
    sgkf = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    SPLITS = list(sgkf.split(df, df.species, groups))
    json.dump([sorted(set(groups[va])) for _, va in SPLITS], open(split_file, "w"))

for k, (tr, va) in enumerate(SPLITS):
    assert not set(groups[tr]) & set(groups[va])
    print(f"fold {k}: val {len(va)} images / {len(set(groups[va]))} animals "
          f"({ {k: int(v) for k, v in df.species.iloc[va].value_counts().items()} })")

for k, (tr, va) in enumerate(SPLITS):
    for key in BACKBONES:
        f = OUT / "cv" / f"fold{k}_{key}.npz"
        if f.exists():
            continue
        print(f"\n=== fold {k+1}/{FOLDS} · {key} ===")
        net, mu, sd = fit_backbone(key, tr, va, seed=SEED + k)
        feats, direct = extract(net, np.arange(len(df)), mu[0], sd[0])
        np.savez(f, feats=feats, direct=direct)
        print(f"    -> val MAE (flip TTA) {np.abs(direct[va] - y[va]).mean():.2f} kg")
        del net; gc.collect(); torch.cuda.empty_cache()
""")

# =========================================================================
md(r"""
## 6. หัว regression + ensemble

หัวเดิม SVR · Ridge · MLP multi-task (บนฟีเจอร์ 4 backbone + เรขาคณิต) และเพิ่ม `direct` = หัวของ backbone เอง
น้ำหนักรวมหาด้วย NNLS แบบ leave-one-fold-out (ไม่ใช้ fold ที่ประเมินในการหาน้ำหนัก)
""")

code(r"""
class FrozenMultitaskMLP(nn.Module):
    # identical to ai-api/app/weight.py
    def __init__(self, in_features, n_reg=len(REG_TARGETS), dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, n_reg))
        self.log_var = nn.Parameter(torch.zeros(n_reg))
    def forward(self, x): return self.net(x)


def train_mlp(X, idx, seed=SEED):
    # Final-epoch weights: picking the best epoch on the val fold would leak it.
    torch.manual_seed(seed)
    mu, sd = target_norm(idx)
    Xt = torch.tensor(X[idx], dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(np.nan_to_num((Y[idx] - mu) / sd), dtype=torch.float32, device=DEVICE)
    Mt = torch.tensor(MASK[idx], device=DEVICE)
    net = FrozenMultitaskMLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    for _ in range(MLP_EPOCHS):
        net.train()
        perm = torch.randperm(len(idx), device=DEVICE)
        for i in range(0, len(idx) - 31, 32):               # drop_last: BatchNorm needs > 1
            b = perm[i:i + 32]; out, m = net(Xt[b]), Mt[b]
            loss = sum(torch.exp(-net.log_var[t]) * F.smooth_l1_loss(out[m[:, t], t], Yt[b][m[:, t], t])
                       + net.log_var[t] for t in range(out.shape[1]) if m[:, t].any())
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net.eval().cpu(), mu, sd

def mlp_predict(net, mu, sd, X):
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy() * sd + mu

make_svr   = lambda: make_pipeline(StandardScaler(), SVR(C=100, epsilon=5))
make_ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 40)))

def fit_heads(X, idx, seed=SEED):
    return dict(svr=make_svr().fit(X[idx], y[idx]), ridge=make_ridge().fit(X[idx], y[idx]),
                mlp=train_mlp(X, idx, seed))

def head_preds(H, X):
    aux = mlp_predict(*H["mlp"], X)
    return dict(svr=H["svr"].predict(X), ridge=H["ridge"].predict(X), mlp=aux[:, 0]), aux

def load_fold(k):
    Z = [np.load(OUT / "cv" / f"fold{k}_{key}.npz") for key in BACKBONES]
    return np.hstack([z["feats"] for z in Z] + [G]), np.mean([z["direct"] for z in Z], 0)

HEADS = ["svr", "ridge", "mlp", "direct"]
oof = {h: np.full(len(df), np.nan) for h in HEADS}
oof_aux = np.full(Y.shape, np.nan)
for k, (tr, va) in enumerate(SPLITS):
    X, direct = load_fold(k)
    H = fit_heads(X, tr, seed=SEED + k)
    p, aux = head_preds(H, X[va])
    for h in ("svr", "ridge", "mlp"):
        oof[h][va] = p[h]
    oof["direct"][va], oof_aux[va] = direct[va], aux
    print(f"fold {k+1}/{FOLDS} heads done")

def nnls_w(P, t):
    w, _ = nnls(P, t)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])

P = np.column_stack([oof[h] for h in HEADS])
oof["mean"] = P.mean(1)
oof["weighted"] = np.zeros(len(df))
for tr, va in SPLITS:
    oof["weighted"][va] = P[va] @ nnls_w(P[tr], y[tr])
""")

# =========================================================================
md("## 7. ผลลัพธ์ (out-of-fold, สัตว์ที่ backbone ไม่เคยเห็น)")

code(r"""
def report(t, p, label=""):
    t, p = np.asarray(t, float), np.asarray(p, float); e = p - t
    ss = ((t - t.mean()) ** 2).sum()
    m = dict(n=int(len(t)), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
             MAPE=float(np.abs(e / t).mean() * 100), R2=float(1 - (e ** 2).sum() / ss) if ss else None,
             bias=float(e.mean()), within10=float((np.abs(e / t) <= .10).mean() * 100))
    if label:
        print(f"  {label:<30} n={m['n']:>4} | MAE {m['MAE']:6.2f} | RMSE {m['RMSE']:6.2f} | "
              f"MAPE {m['MAPE']:5.2f}% | R2 {m['R2'] or float('nan'):6.3f} | "
              f"bias {m['bias']:+6.1f} | <=10% {m['within10']:5.1f}%")
    return m

metrics = {"all": {}, "by_species": {}}
print("per image, all animals")
for h, p in oof.items():
    metrics["all"][h] = report(y, p, h)
report(y, np.full_like(y, y.mean()), "reference: predict-the-mean")

COMBINE = "weighted" if metrics["all"]["weighted"]["MAE"] < metrics["all"]["mean"]["MAE"] else "mean"
W_FINAL = nnls_w(P, y) if COMBINE == "weighted" else np.full(len(HEADS), 1 / len(HEADS))
best = oof[COMBINE]
print(f"\ncombine = {COMBINE}  weights { {h: round(float(w), 3) for h, w in zip(HEADS, W_FINAL)} }")

print("\nby species")
for sp in SPECIES:
    s = (df.species == sp).to_numpy()
    metrics["by_species"][sp] = report(y[s], best[s], sp)
    report(y[s], np.full(s.sum(), y[s].mean()), f"  {sp} predict-the-mean")

pa = pd.DataFrame({"a": groups, "y": y, "p": best}).groupby("a").mean()
metrics["per_animal"] = report(pa.y, pa.p, "per animal (photos averaged)")
f = df.label_is_formula.to_numpy()
metrics["cow_formula_labels"] = report(y[f], best[f], "cow, formula labels")
metrics["weighed_labels"] = report(y[~f], best[~f], "all others (weighed-looking)")

print("\nauxiliary targets (MLP)")
metrics["aux"] = {t: report(Y[MASK[:, i], i], oof_aux[MASK[:, i], i], t)
                  for i, t in enumerate(REG_TARGETS) if i and MASK[:, i].sum() > 10}
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for sp, c in zip(SPECIES, ("tab:blue", "tab:orange")):
    s = (df.species == sp).to_numpy()
    ax[0].scatter(y[s], best[s], alpha=.35, c=c, label=sp)
lim = [y.min(), y.max()]; ax[0].plot(lim, lim, "k--")
ax[0].set_xlabel("actual (kg)"); ax[0].set_ylabel("predicted (kg)"); ax[0].legend()
ax[0].set_title(f"OOF {COMBINE} ensemble")

d = best - y; b_, s_ = d.mean(), d.std(ddof=1)
ax[1].scatter((y + best) / 2, d, alpha=.35)
for v, ls in ((b_, "-"), (b_ + 1.96 * s_, "--"), (b_ - 1.96 * s_, "--")):
    ax[1].axhline(v, color="crimson", ls=ls)
ax[1].set_title(f"Bland-Altman  LoA {b_-1.96*s_:+.0f} .. {b_+1.96*s_:+.0f} kg")

names = list(metrics["all"]); ax[2].barh(names, [metrics["all"][n]["MAE"] for n in names])
ax[2].axvline(np.abs(y - y.mean()).mean(), color="gray", ls=":", label="predict-the-mean")
ax[2].set_xlabel("MAE (kg)"); ax[2].legend(); ax[2].set_title("heads")
plt.tight_layout(); plt.show()
""")

# =========================================================================
md(r"""
## 8. เทรนชุดเต็มและเซฟสำหรับ ai-api

fine-tune backbone ด้วยสูตรเดียวกับใน CV บนสัตว์ทุกตัว → ฟีเจอร์ → หัวทั้งหมด → `OUT/deploy/`

นำไปใช้: คัดลอกทุกไฟล์ใน `deploy/` ไปที่ `ai-api/models/predict_weight/`
แล้วตั้ง `WEIGHT_ENSEMBLE=ensemble_cowbuff` ใน `docker-compose.yml` (ลบ env ออก = ย้อนกลับรุ่นเดิม)
""")

code(r"""
DEPLOY = OUT / "deploy"; ENS = DEPLOY / "ensemble_cowbuff"
ENS.mkdir(parents=True, exist_ok=True)
ALL = np.arange(len(df))

feats, directs = [], []
for key, name in BACKBONES.items():
    f = DEPLOY / f"{key}_cowbuff.pth"
    if f.exists():
        ck = torch.load(f, map_location=DEVICE, weights_only=True)
        net = WeightNet(name).to(DEVICE); net.load_state_dict(ck["state_dict"])
        mu, sd = ck["norm"]["weight"]
    else:
        print(f"\n=== full data · {key} ===")
        net, mu_all, sd_all = fit_backbone(key, ALL, None, seed=SEED)
        mu, sd = float(mu_all[0]), float(sd_all[0])
        torch.save(dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()},
                        backbone=name, n_breeds=N_BREEDS, breeds=BREEDS, img=IMG, epochs=EPOCHS,
                        norm={t: [float(mu_all[i]), float(sd_all[i])] for i, t in enumerate(REG_TARGETS)}),
                   f)
    F_, D_ = extract(net, ALL, mu, sd)
    feats.append(F_); directs.append(D_)
    del net; gc.collect(); torch.cuda.empty_cache()

X_all = np.hstack(feats + [G])
H = fit_heads(X_all, ALL)
joblib.dump(H["svr"], ENS / "svr.joblib")
joblib.dump(H["ridge"], ENS / "ridge.joblib")
mlp, mmu, msd = H["mlp"]
torch.save(dict(state_dict=mlp.state_dict(), in_features=int(X_all.shape[1]),
                y_mu=mmu.tolist(), y_sd=msd.tolist(), targets=REG_TARGETS), ENS / "mlp.pt")
print("in-sample MAE (sanity only):",
      round(float(np.abs(np.column_stack([*head_preds(H, X_all)[0].values(), np.mean(directs, 0)])
                         @ W_FINAL - y).mean()), 2))

now = datetime.now(timezone.utc)
manifest = dict(
    version=f"cowbuff-ensemble@{now:%Y-%m-%d}",
    created=now.isoformat(timespec="seconds"),
    species=list(SPECIES),
    backbones=list(BACKBONES),
    backbone_files=[f"{k}_cowbuff.pth" for k in BACKBONES],
    detector="yolo11n.pt (COCO, largest box of any class)",
    input_size=640, feat_img=IMG, n_features=int(X_all.shape[1]),
    geom_cols=GEOM_COLS, geom_stats=GEOM_STATS,
    heads=HEADS, weights=[float(w) for w in W_FINAL], combine=COMBINE,
    epochs=EPOCHS, mlp_epochs=MLP_EPOCHS, folds=FOLDS,
    n_images=int(len(df)), n_animals=int(df.animal_id.nunique()),
    n_animals_by_species={sp: int(df[df.species == sp].animal_id.nunique()) for sp in SPECIES},
    breeds=BREEDS,
    typical_error_kg=round(metrics["all"][COMBINE]["MAE"], 1),
    typical_error_source="nested StratifiedGroupKFold OOF, backbones re-fine-tuned per fold",
    typical_error_by_species={sp: round(m["MAE"], 1) for sp, m in metrics["by_species"].items()},
    aux_typical_error={t: {"MAE": round(m["MAE"], 2), "R2": round(m["R2"], 3)} for t, m in metrics["aux"].items()},
    aux_typical_error_source="same nested CV, MLP head",
    metrics=metrics,
    versions=dict(torch=torch.__version__, timm=timm.__version__, sklearn=sklearn.__version__),
    caveats=[f"{int(df[df.label_is_formula].animal_id.nunique())} cow animals have weight == "
             "5.854*L - 595.98 exactly: tape-formula labels, not scale weights."],
)
(ENS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame({"file": df.file, "animal_id": groups, "species": df.species, "actual": y,
              **{f"pred_{h}": v for h, v in oof.items()}}).to_csv(ENS / "oof_predictions.csv", index=False)
print("saved:", sorted(p.name for p in DEPLOY.rglob("*") if p.is_file()))
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
        "colab": {"provenance": [], "gpuType": "V100"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).parent / "CowBuff_Weight_Training.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(CELLS)} cells)")
