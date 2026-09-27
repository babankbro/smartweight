"""Recover the size information the cropping step threw away.

The notebook cropped each cow to its bounding box and then resized that crop to
384x384. That resize is the reason the CNN cannot work: a 250 kg calf
photographed close up and a 500 kg bull photographed from far away produce
almost the same picture, and squashing to a square also destroys the
length-to-height ratio.

This script runs over the ORIGINAL (uncropped) images and records the geometry
in original-image pixels, plus per-zone areas from the project's own 7-zone
segmentation model when it is available. Those numbers go back into the model
as tabular inputs in 02_train_hybrid.py.

    python ml/00_extract_geometry.py \
        --images  /content/cow_dataset \
        --csv     /content/cow_dataset/cow_weight_dataset.csv \
        --model   ai-api/models/segment.pt \
        --out     ml/geometry.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ai-api"))
from app import geometry as G  # noqa: E402  (reuses the serving code, no drift)


def zone_geometry(model, img: np.ndarray) -> dict:
    """Per-zone areas from the 7-zone segmentation model."""
    h, w = img.shape[:2]
    res = model.predict(img, imgsz=640, conf=0.25, retina_masks=True, verbose=False)[0]
    if res.masks is None:
        return {}

    data = res.masks.data.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)

    per_zone: dict[int, np.ndarray] = {}
    for m, c in zip(data, cls):
        b = (m > 0.5).astype(np.uint8)
        if b.shape != (h, w):
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST)
        per_zone[c] = per_zone.get(c, np.zeros((h, w), np.uint8)) | b

    merged = G.union_mask(list(per_zone.values()), (h, w))
    body, others = G.primary_component(merged)
    total = G.mask_area(body)
    if not total:
        return {}

    out: dict[str, float] = {"other_animals": others, "zones_found": 0}
    for cid, mask in per_zone.items():
        clipped = mask & body
        area = G.mask_area(clipped)
        # `share` is the one feature here that does not change with camera
        # distance, so it stays useful even for photos with no scale reference.
        out[f"zone{cid + 1}_area"] = area
        out[f"zone{cid + 1}_share"] = area / total
        out["zones_found"] += int(area > 0)

    shape = G.shape_metrics(body)
    out.update({
        "body_area_px": total,
        "body_area_ratio": total / (w * h),
        "body_length_px": shape.get("length_px", np.nan),
        "body_height_px": shape.get("height_px", np.nan),
        "body_aspect": shape.get("aspect", np.nan),
        "body_solidity": shape.get("solidity", np.nan),
        "body_perimeter_px": shape.get("perimeter_px", np.nan),
        "tilt_deg": shape.get("tilt_deg", np.nan),
    })
    for i, d in enumerate(G.profile_depths(body)):
        out[f"depth_q{i}"] = d
    return out


def box_geometry(model, img: np.ndarray) -> dict:
    """Fallback when no segmentation model is given: a COCO detector's box."""
    h, w = img.shape[:2]
    res = model.predict(img, verbose=False)[0]
    if res.boxes is None or not len(res.boxes):
        return {}
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = xyxy[int(areas.argmax())]
    bw, bh = x2 - x1, y2 - y1
    return {
        "bbox_w_px": float(bw),
        "bbox_h_px": float(bh),
        "bbox_area_px": float(bw * bh),
        "bbox_area_ratio": float(bw * bh / (w * h)),
        "bbox_aspect": float(bw / bh) if bh else np.nan,
        "img_w": w,
        "img_h": h,
    }


def main(images_root: str, csv_path: str, model_path: str, out_path: str) -> None:
    from ultralytics import YOLO

    df = pd.read_csv(csv_path)
    model = YOLO(model_path)
    segmentation = "-seg" in str(getattr(model, "task", "")) or model.task == "segment"
    print(f"model: {model_path} (task={model.task}) -> "
          f"{'zone geometry' if segmentation else 'bounding box only'}")

    root = Path(images_root)
    rows, missing = [], 0
    for i, row in df.iterrows():
        p = root / row["path"]
        img = cv2.imread(str(p))
        if img is None:
            missing += 1
            continue

        g = zone_geometry(model, img) if segmentation else box_geometry(model, img)
        h, w = img.shape[:2]
        # Always keep the raw frame size: it is what makes a pixel area
        # interpretable at all.
        rows.append({**row.to_dict(), "img_w": w, "img_h": h, **g})

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}")

    out = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}: {len(out)} rows, {out.shape[1]} columns "
          f"({missing} images unreadable)")
    if "body_area_px" in out:
        ok = out["body_area_px"].notna().sum()
        print(f"segmented {ok}/{len(out)} ({ok / len(out) * 100:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="root holding train/ valid/ test/")
    p.add_argument("--csv", required=True, help="cow_weight_dataset.csv")
    p.add_argument("--model", default="ai-api/models/segment.pt",
                   help="7-zone segment.pt, or yolo11n.pt for boxes only")
    p.add_argument("--out", default="ml/geometry.csv")
    a = p.parse_args()
    main(a.images, a.csv, a.model, a.out)
