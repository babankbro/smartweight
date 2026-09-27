"""Shared pieces for the weight-regression experiments.

The central idea this module enforces: **split by animal, evaluate in kilograms
a farmer would recognise, and never let two photos of the same cow land on
opposite sides of a split.**
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- cow.csv column names -------------------------------------------------
# The raw file carries stray label columns ('_Height_', '_Age_', '_Weight_')
# next to the real ones; those are dropped on load.
COL_ID = "ID"
COL_WEIGHT = "Weight"
COL_HEIGHT = "Height"
COL_LENGTH = "L"
COL_AGE = "Age"
COL_BREED = "Category"


def parse_age_years(value) -> float:
    """'1.6Y' -> 1.6, '4Y' -> 4.0, '8M' -> 0.667.

    Age is stored as free text; a cow's age matters because weight-per-size
    changes as the animal matures, so it is worth recovering rather than
    dropping.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()
    m = re.match(r"^([\d.]+)\s*([YM]?)$", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    return n / 12.0 if m.group(2) == "M" else n


def normalise_id(raw: str) -> str:
    """'C139-1', 'c139', '139' -> 'C139'.

    The trailing '-1' is Roboflow's augmentation suffix: several files share
    one animal, and treating them as separate animals is exactly the leak this
    module exists to prevent.
    """
    s = str(raw).strip().upper()
    s = s.split("_")[0].split("-")[0]
    if s.startswith("C"):
        s = s[1:]
    return f"C{int(s):03d}" if s.isdigit() else str(raw).strip().upper()


def load_cow_csv(path: str | Path) -> pd.DataFrame:
    """Measurements table, cleaned: one row per animal."""
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if not re.fullmatch(r"_.*_", str(c))]]
    df[COL_ID] = df[COL_ID].map(normalise_id)
    df["age_years"] = df[COL_AGE].map(parse_age_years)
    for c in (COL_WEIGHT, COL_HEIGHT, COL_LENGTH):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[COL_WEIGHT]).drop_duplicates(subset=[COL_ID])


def add_body_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geometry a stockman would recognise, from height and body length.

    Schaeffer's rule is W proportional to girth^2 x length. Girth is not in the
    data, but chest depth scales with wither height, so height^2 x length is the
    same quantity with a different constant folded in.
    """
    out = df.copy()
    h, ln = out[COL_HEIGHT], out[COL_LENGTH]
    out["h2l"] = h**2 * ln / 10_000        # the Schaeffer-style volume proxy
    out["hl"] = h * ln / 100
    out["h3"] = h**3 / 10_000
    out["l3"] = ln**3 / 10_000
    out["ratio_lh"] = ln / h               # body proportion, scale-free
    out["log_h"] = np.log(h)
    out["log_l"] = np.log(ln)
    return out


# --- metrics --------------------------------------------------------------

def regression_report(y_true, y_pred, label: str = "") -> dict:
    """Metrics chosen so a non-specialist can act on them.

    MAPE and the within-10% rate are what a farmer actually cares about; R2 is
    kept because a negative value is the clearest possible signal that a model
    is worse than guessing the mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    ss_res = float((err**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    m = {
        "n": len(y_true),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE_%": float((np.abs(err) / y_true).mean() * 100),
        "R2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
        "bias": float(err.mean()),
        "within_5%": float((np.abs(err) / y_true <= 0.05).mean() * 100),
        "within_10%": float((np.abs(err) / y_true <= 0.10).mean() * 100),
        # Predicting the mean of the test set - the bar any model must clear.
        "RMSE_mean_baseline": float(np.sqrt(((y_true - y_true.mean()) ** 2).mean())),
    }
    if label:
        print(
            f"{label:<28} MAE {m['MAE']:6.2f} kg | RMSE {m['RMSE']:6.2f} | "
            f"MAPE {m['MAPE_%']:5.2f}% | R2 {m['R2']:6.3f} | "
            f"bias {m['bias']:+6.2f} | <=10% {m['within_10%']:5.1f}%"
        )
    return m


def bland_altman(y_true, y_pred, ax=None, title: str = "Bland-Altman"):
    """Agreement plot. Reveals two failures a scatter plot hides: a constant
    offset, and error that grows with the size of the animal."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mean, diff = (y_true + y_pred) / 2, y_pred - y_true
    bias, sd = diff.mean(), diff.std(ddof=1)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, alpha=0.6)
    ax.axhline(bias, color="crimson", label=f"bias {bias:+.1f} kg")
    for k, style in ((1.96, "--"), (-1.96, "--")):
        ax.axhline(bias + k * sd, color="gray", ls=style)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("mean of actual and predicted (kg)")
    ax.set_ylabel("predicted - actual (kg)")
    ax.set_title(f"{title}  (LoA {bias - 1.96 * sd:+.0f} .. {bias + 1.96 * sd:+.0f} kg)")
    ax.legend()
    return ax
