"""Render a colour-coded overlay so a user can see what the model actually cut.

Showing this matters more than it sounds: it is the only way a farmer in the
field can tell a good scan from a bad one, and the only way we can debug a
wrong number after the fact.
"""

import base64

import cv2
import numpy as np

ALPHA = 0.45
MAX_WIDTH = 900


def _put_text(img, text, org, color, scale=0.5, thick=1):
    # Draw a dark outline first so labels stay readable on any background.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(image_bgr: np.ndarray, zones: list[dict]) -> str:
    """zones: [{mask, color (r,g,b), name_en, share}] -> base64 PNG data."""
    canvas = image_bgr.copy()
    tint = canvas.copy()

    for z in zones:
        b, g, r = z["color"][2], z["color"][1], z["color"][0]
        tint[z["mask"] > 0] = (b, g, r)

    canvas = cv2.addWeighted(tint, ALPHA, canvas, 1 - ALPHA, 0)

    # Outline each zone so adjacent zones stay distinguishable after blending.
    for z in zones:
        cnts, _ = cv2.findContours(z["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (z["color"][2], z["color"][1], z["color"][0]), 2)

    # Legend. English only - OpenCV's built-in fonts cannot render Thai script,
    # so Thai names are shown by the web UI instead.
    pad, row_h = 10, 22
    panel_h = row_h * len(zones) + pad
    panel_w = 250
    # Dark panel behind the text, otherwise the legend vanishes on pale photos.
    box = canvas[0:panel_h, 0:panel_w].copy()
    canvas[0:panel_h, 0:panel_w] = cv2.addWeighted(
        box, 0.25, np.zeros_like(box), 0.75, 0
    )
    for i, z in enumerate(zones):
        y = pad + row_h * (i + 1)
        cv2.rectangle(
            canvas, (pad, y - 12), (pad + 16, y + 2),
            (z["color"][2], z["color"][1], z["color"][0]), -1
        )
        _put_text(canvas, f'{z["name_en"]}  {z["share"] * 100:.1f}%', (pad + 24, y), (255, 255, 255))

    if canvas.shape[1] > MAX_WIDTH:
        h = int(canvas.shape[0] * MAX_WIDTH / canvas.shape[1])
        canvas = cv2.resize(canvas, (MAX_WIDTH, h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
