"""Baseline: predict weight from the tape measurements already in cow.csv.

Run this BEFORE touching the CNN. It answers a question the notebook never
asked: how much of the weight is explainable from height and body length alone?

That number is the ceiling any image model should be measured against. If a
54M-parameter network cannot beat a three-variable regression, the problem is
the inputs, not the architecture.

    python ml/01_baseline_tabular.py --csv /path/to/cow.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (  # noqa: E402
    COL_BREED, COL_HEIGHT, COL_LENGTH, COL_WEIGHT,
    add_body_features, load_cow_csv, regression_report,
)


def main(csv_path: str, seed: int = 42) -> pd.DataFrame:
    df = add_body_features(load_cow_csv(csv_path))
    df = df.dropna(subset=[COL_HEIGHT, COL_LENGTH])

    print(f"animals: {len(df)}")
    print(
        f"weight : {df[COL_WEIGHT].min():.0f}-{df[COL_WEIGHT].max():.0f} kg "
        f"(mean {df[COL_WEIGHT].mean():.1f}, sd {df[COL_WEIGHT].std():.1f})"
    )
    if COL_BREED in df:
        print(f"breeds : {df[COL_BREED].nunique()}")
    print()

    y = df[COL_WEIGHT].to_numpy(float)
    # One row per animal here, so plain KFold already keeps animals whole.
    # The image models must use GroupKFold instead - see 02_train_hybrid.py.
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    feature_sets = {
        "height only":        [COL_HEIGHT],
        "length only":        [COL_LENGTH],
        "height + length":    [COL_HEIGHT, COL_LENGTH],
        "Schaeffer h^2*L":    ["h2l"],
        "full geometry":      [COL_HEIGHT, COL_LENGTH, "h2l", "hl", "h3", "l3", "ratio_lh"],
        "full geometry + age": [COL_HEIGHT, COL_LENGTH, "h2l", "hl", "h3", "l3",
                                "ratio_lh", "age_years"],
    }

    models = {
        "ridge": lambda: make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))
        ),
        "rf": lambda: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
    }

    print("-- cross-validated, 5-fold ------------------------------------------")
    rows = []
    for fname, feats in feature_sets.items():
        X = df[feats].to_numpy(float)
        if np.isnan(X).any():                      # age is sometimes missing
            X = np.nan_to_num(X, nan=np.nanmean(X))
        for mname, make in models.items():
            pred = cross_val_predict(make(), X, y, cv=cv)
            m = regression_report(y, pred, f"{fname:<20} {mname}")
            rows.append({"features": fname, "model": mname, **m})

    # The bar to clear: always answer with the mean.
    print()
    regression_report(y, np.full_like(y, y.mean()), "predict-the-mean")

    print("\n-- notebook's CNN, for comparison -----------------------------------")
    print(f"{'EfficientNet-V2-M image only':<28} MAE  79.91 kg | RMSE  92.66 | "
          f"MAPE ~21%   | R2 < 0")

    out = pd.DataFrame(rows).sort_values("MAE")
    best = out.iloc[0]
    print(
        f"\nBest tabular: {best['features']} / {best['model']} -> "
        f"MAE {best['MAE']:.2f} kg, MAPE {best['MAPE_%']:.2f}%, R2 {best['R2']:.3f}"
    )
    if best["MAE"] < 79.91:
        print(
            f"=> {79.91 / best['MAE']:.1f}x better than the image model, from "
            f"{len(feature_sets[best['features']])} numbers and no GPU."
        )

    # Fitted coefficients make the relationship publishable, not just accurate.
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    ridge.fit(df[["h2l"]].to_numpy(float), y)
    a = ridge[-1].coef_[0] / df["h2l"].std()
    b = ridge[-1].intercept_ - a * df["h2l"].mean()
    print(f"\nFitted rule:  W(kg) ~= {a:.3f} * (Height^2 x L / 10000) + {b:.1f}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="cow.csv", help="path to cow.csv")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.csv, a.seed)
