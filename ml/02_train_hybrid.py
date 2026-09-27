"""Hybrid image + geometry regressor, trained with animal-grouped CV.

Fixes, in order of how much they matter:

1. Size information is restored. The CNN sees a letterboxed crop (aspect ratio
   preserved) and the head additionally receives the geometry in original-image
   pixels. Without this the network cannot tell a near calf from a distant bull.
2. The tape measurements in cow.csv - height, body length, age, breed - are fed
   in. The notebook ignored them entirely.
3. Splits are grouped by animal, so two photos of one cow can never straddle
   train and test.
4. The target is standardised and the head's bias starts at the mean weight,
   so training does not spend fifty epochs learning to output "about 380".
5. SmoothL1 replaces MSE: with a few hundred samples a single mislabelled cow
   should not dominate the gradient.
6. A smaller backbone. EfficientNet-V2-M has 54M parameters for ~220 distinct
   images; that is the overfitting seen as train RMSE 28 against valid 91.
7. The LR schedule completes. OneCycleLR set to 400 epochs and stopped at 187
   leaves the model at near-peak LR, never annealed.

    python ml/02_train_hybrid.py --data /content/cow_dataset \
        --geometry ml/geometry.csv --cow-csv /content/cow.csv --folds 5
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (  # noqa: E402
    COL_BREED, COL_HEIGHT, COL_ID, COL_LENGTH,
    add_body_features, load_cow_csv, normalise_id, regression_report,
)

IMG_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
class LetterboxPad:
    """Resize the long edge, pad the rest.

    A plain Resize((n, n)) squashes the image to a square and destroys the
    length-to-height ratio - one of the two strongest shape cues for weight.
    """

    def __init__(self, size: int, fill: int = 114):
        self.size, self.fill = size, fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.size / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img, ((self.size - img.size[0]) // 2, (self.size - img.size[1]) // 2))
        return canvas


NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    LetterboxPad(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    # Rotation and shear are kept small: a cow does not lean 15 degrees, and
    # aggressive geometric jitter fights the shape signal we want to preserve.
    transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
    transforms.ToTensor(),
    NORM,
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
])

EVAL_TF = transforms.Compose([LetterboxPad(IMG_SIZE), transforms.ToTensor(), NORM])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class CowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, feat_cols: list[str], tf):
        self.df = df.reset_index(drop=True)
        self.root, self.feat_cols, self.tf = Path(root), feat_cols, tf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.loc[i]
        img = self.tf(Image.open(self.root / r["path"]).convert("RGB"))
        feats = torch.tensor(r[self.feat_cols].to_numpy(dtype=np.float32))
        return img, feats, torch.tensor(float(r["weight"]), dtype=torch.float32)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class HybridNet(nn.Module):
    """Image features concatenated with measured geometry.

    The geometry branch is what carries absolute size; the image branch
    contributes condition and shape that a few numbers cannot express.
    """

    def __init__(self, n_features: int, mean: float, std: float, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        n_img = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.tab = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 64), nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 64), nn.SiLU(),
        ) if n_features else None

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_img + (64 if n_features else 0), 128), nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        # Start at the training mean instead of at zero. The notebook's first
        # epoch showed RMSE 406 - the mean weight - purely because of this.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 0.0)
        self.register_buffer("y_mean", torch.tensor(mean))
        self.register_buffer("y_std", torch.tensor(std))

    def forward(self, img, feats):
        z = self.backbone(img)
        if self.tab is not None:
            z = torch.cat([z, self.tab(feats)], dim=1)
        return self.head(z).squeeze(1) * self.y_std + self.y_mean


# --------------------------------------------------------------------------
# Train / evaluate one fold
# --------------------------------------------------------------------------
def run_fold(tr_df, va_df, root, feat_cols, epochs, batch_size, lr, patience, seed):
    torch.manual_seed(seed)

    y_mean, y_std = float(tr_df["weight"].mean()), float(tr_df["weight"].std())
    model = HybridNet(len(feat_cols), y_mean, y_std).to(DEVICE)

    tr = DataLoader(CowDataset(tr_df, root, feat_cols, TRAIN_TF),
                    batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(CowDataset(va_df, root, feat_cols, EVAL_TF),
                    batch_size=batch_size, shuffle=False, num_workers=2)

    # Huber in normalised units: robust to a mislabelled animal, and scale-free.
    crit = nn.SmoothL1Loss(beta=1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # Cosine over the full budget. Unlike OneCycleLR + early stopping, stopping
    # early here still leaves the weights at a sensibly annealed LR.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_wts, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(epochs):
        model.train()
        for img, f, y in tr:
            img, f, y = img.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit((model(img, f) - y_mean) / y_std, (y - y_mean) / y_std)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for img, f, y in va:
                preds.append(model(img.to(DEVICE), f.to(DEVICE)).cpu().numpy())
                trues.append(y.numpy())
        mae = float(np.abs(np.concatenate(preds) - np.concatenate(trues)).mean())

        if mae < best - 0.1:
            best, best_wts, stale = mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if ep % 10 == 0 or stale >= patience:
            print(f"    ep {ep + 1:>3}/{epochs}  val MAE {mae:6.2f}  best {best:6.2f}")
        if stale >= patience:
            print(f"    early stop at epoch {ep + 1}")
            break

    model.load_state_dict(best_wts)

    # Test-time augmentation: average the prediction with its mirror image.
    # Free accuracy, and it damps the variance that a 28-sample split suffers.
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for img, f, y in va:
            img, f = img.to(DEVICE), f.to(DEVICE)
            p = (model(img, f) + model(torch.flip(img, dims=[3]), f)) / 2
            preds.append(p.cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues), model


# --------------------------------------------------------------------------
def build_table(data_root, geometry_csv, cow_csv):
    """One row per image, carrying geometry, tape measurements and a group id."""
    df = pd.read_csv(geometry_csv) if geometry_csv else pd.read_csv(
        Path(data_root) / "cow_weight_dataset.csv"
    )
    df["animal_id"] = df["ID"].map(normalise_id)

    if cow_csv:
        meas = add_body_features(load_cow_csv(cow_csv))
        keep = [COL_ID, COL_HEIGHT, COL_LENGTH, "age_years", "h2l", "hl",
                "ratio_lh", COL_BREED]
        meas = meas[[c for c in keep if c in meas.columns]]
        df = df.merge(meas, left_on="animal_id", right_on=COL_ID, how="left",
                      suffixes=("", "_m"))

    if COL_BREED in df.columns:
        df = pd.concat([df, pd.get_dummies(df[COL_BREED], prefix="breed")], axis=1)

    skip = {"path", "ID", "animal_id", "weight", "category", COL_BREED, f"{COL_ID}_m"}
    feat_cols = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    # Standardise once, globally. Mild leakage of feature scale across folds is
    # acceptable here and far less harmful than per-fold scale drift on ~200 rows.
    df[feat_cols] = (df[feat_cols] - df[feat_cols].mean()) / (df[feat_cols].std() + 1e-8)
    return df, feat_cols


def main(a):
    df, feat_cols = build_table(a.data, a.geometry, a.cow_csv)
    print(f"images {len(df)} | animals {df['animal_id'].nunique()} | "
          f"tabular features {len(feat_cols)}")
    print(f"device {DEVICE}\n")

    if df["animal_id"].nunique() < a.folds:
        raise SystemExit("fewer animals than folds")

    # THE fix that makes the numbers trustworthy: every photo of one animal
    # stays inside a single fold.
    gkf = GroupKFold(n_splits=a.folds)
    oof = np.zeros(len(df))
    for k, (tr_i, va_i) in enumerate(gkf.split(df, groups=df["animal_id"]), 1):
        print(f"  fold {k}/{a.folds}  train {len(tr_i)} / val {len(va_i)} "
              f"({df.iloc[va_i]['animal_id'].nunique()} animals)")
        p, _, model = run_fold(
            df.iloc[tr_i], df.iloc[va_i], a.data, feat_cols,
            a.epochs, a.batch_size, a.lr, a.patience, a.seed + k,
        )
        oof[va_i] = p
        if a.save_dir:
            Path(a.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "features": feat_cols,
                        "img_size": IMG_SIZE},
                       Path(a.save_dir) / f"hybrid_fold{k}.pth")

    y = df["weight"].to_numpy(float)
    print("\n-- out-of-fold, grouped by animal ------------------------------------")
    regression_report(y, oof, "hybrid (image + geometry)")
    regression_report(y, np.full_like(y, y.mean()), "predict-the-mean")
    print(f"{'notebook CNN (image only)':<28} MAE  79.91 kg | RMSE  92.66 | R2 < 0")

    # Per-animal: averaging an animal's photos is what the app should do too.
    by_animal = (pd.DataFrame({"a": df["animal_id"], "y": y, "p": oof})
                 .groupby("a").mean())
    print()
    regression_report(by_animal["y"], by_animal["p"], "per-animal (photos averaged)")

    if a.out:
        pd.DataFrame({"animal_id": df["animal_id"], "path": df["path"],
                      "actual": y, "predicted": oof}).to_csv(a.out, index=False)
        print(f"\npredictions -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset root with train/ valid/ test/")
    p.add_argument("--geometry", default=None, help="geometry.csv from step 00")
    p.add_argument("--cow-csv", default=None, help="cow.csv with Height / L / Age")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--out", default="ml/oof_predictions.csv")
    main(p.parse_args())
