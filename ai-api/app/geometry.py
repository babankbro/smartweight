"""Geometric measurements derived from segmentation masks.

Everything here is in PIXELS. Converting to centimetres needs a scale
reference (e.g. an ArUco marker) which the capture protocol does not have yet,
so we deliberately do not pretend to know real-world sizes.
"""

from typing import Any

import cv2
import numpy as np


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def mask_area(mask: np.ndarray) -> int:
    """Number of foreground pixels. Counted directly, not via contour area,
    so holes and disjoint blobs are handled correctly."""
    return int(np.count_nonzero(mask))


def centroid(mask: np.ndarray) -> list[float] | None:
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None
    return [round(m["m10"] / m["m00"], 1), round(m["m01"] / m["m00"], 1)]


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def shape_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Body-shape descriptors of a binary mask.

    `length_px` / `height_px` come from a *rotated* rectangle so an animal
    standing at a slight angle is not measured as if it were wider than it is.
    """
    out: dict[str, Any] = {
        "area_px": mask_area(mask),
        "bbox": bbox(mask),
        "centroid": centroid(mask),
    }
    c = _largest_contour(mask)
    if c is None or len(c) < 5:
        return out

    (_, _), (rw, rh), angle = cv2.minAreaRect(c)
    length, height = max(rw, rh), min(rw, rh)
    if rw < rh:                       # angle refers to the shorter side
        angle -= 90
    angle = (angle + 90) % 180 - 90   # normalise to (-90, 90]
    perimeter = cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))

    out["length_px"] = round(length, 1)
    out["height_px"] = round(height, 1)
    out["perimeter_px"] = round(perimeter, 1)
    # minAreaRect's angle is relative to the longer side only after this fix-up
    out["tilt_deg"] = round(angle, 1)
    out["aspect"] = round(length / height, 3) if height > 0 else None
    # solidity < 1 means concavities - legs apart, dipped back, occlusion
    out["solidity"] = round(out["area_px"] / hull_area, 3) if hull_area > 0 else None
    out["compactness"] = (
        round(4 * np.pi * out["area_px"] / perimeter**2, 3) if perimeter > 0 else None
    )
    return out


def profile_depths(mask: np.ndarray, n: int = 10) -> list[float]:
    """Body depth sampled at `n` slices along the length.

    Captures shape that a single area number cannot: a deep-chested animal and
    a pot-bellied one can have identical total area.
    """
    cols = np.nonzero(mask.any(axis=0))[0]
    if cols.size == 0:
        return []
    heights = np.count_nonzero(mask[:, cols[0] : cols[-1] + 1], axis=0)
    return [round(float(s.mean()), 1) for s in np.array_split(heights, n)]


def union_mask(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Whole-body mask.

    Uses OR, not a sum of areas: the model was trained with `overlap_mask=True`
    so zones can overlap, and summing would double-count those pixels.
    """
    total = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        total |= m
    return total


def primary_component(mask: np.ndarray, min_ratio: float = 0.15) -> tuple[np.ndarray, int]:
    """Isolate the single animal the photo is *about*.

    A field photo often catches other animals at the edges of the frame; merging
    them into one body would corrupt every area measurement. We keep the largest
    connected blob and report how many other plausible animals were seen, so the
    UI can warn the user.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    largest = int(areas.max())
    others = int(np.count_nonzero(areas >= largest * min_ratio) - 1)
    return (labels == main).astype(np.uint8), others
