"""End-to-end demo: photo -> web /api/analyze -> ai-api (segment + weight) -> result card.

Picks animals whose cow.csv weight looks like a real scale reading (not the
5.854*L-595.98 formula), one light / one medium / one heavy, sends each photo
through the web app's proxy exactly as the scan page does, and renders a PNG
card per animal into ml/demo/. Every dataset photo was used to fit the final
heads, so each card also shows the cross-validated (out-of-fold) prediction -
the fair number - next to the live one.

    python ml/04_demo_predictions.py --images D:/.../dataset/cow_dataset
"""

import argparse
import base64
import io
import json
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
ENSEMBLE = REPO / "ai-api" / "models" / "predict_weight" / "ensemble"
FONT = "C:/Windows/Fonts/tahoma.ttf"
FONT_B = "C:/Windows/Fonts/tahomabd.ttf"


def pick(oof: pd.DataFrame, n: int) -> pd.DataFrame:
    """One photo per animal, weighed-looking labels only, spread by weight."""
    un = oof[~oof.label_is_formula].drop_duplicates("animal_id").sort_values("actual")
    idx = [round(i * (len(un) - 1) / (n - 1)) for i in range(n)]
    return un.iloc[idx]


def card(photo: Image.Image, res: dict, row, out: Path) -> None:
    f = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s)
    overlay = Image.open(io.BytesIO(base64.b64decode(res["overlay_png_base64"].split(",", 1)[1])))
    side = 460
    ph, ov = (im.convert("RGB").resize((side, side)) for im in (photo, overlay))

    W, H = side * 2 + 60, side + 330
    c = Image.new("RGB", (W, H), (245, 247, 250))
    d = ImageDraw.Draw(c)
    d.text((20, 14), f"{row.animal_id}  ·  {row.path[:38]}", font=f(18, True), fill=(30, 58, 138))
    c.paste(ph, (20, 50)); c.paste(ov, (side + 40, 50))
    d.text((20, 50 + side + 6), "input photo", font=f(14), fill=(100, 100, 100))
    d.text((side + 40, 50 + side + 6), "1) segmentation - 7 zones", font=f(14), fill=(100, 100, 100))

    w, q = res["weight"], res["quality"]
    y0 = 50 + side + 36
    err = w["kg"] - row.actual
    d.rounded_rectangle((20, y0, W - 20, H - 20), 16, fill=(236, 253, 245), outline=(16, 185, 129), width=2)
    d.text((40, y0 + 14), "2) predicted weight", font=f(16), fill=(4, 120, 87))
    d.text((40, y0 + 38), f"{w['kg']:.0f} kg", font=f(56, True), fill=(4, 120, 87))
    d.text((40, y0 + 108), f"likely range {w['kg'] - w['typical_error_kg']:.0f}–{w['kg'] + w['typical_error_kg']:.0f} kg"
           f"   (±{w['typical_error_kg']:.0f} kg typical error)", font=f(15), fill=(6, 95, 70))

    x2 = W // 2 + 20
    lab = "tape formula" if row.label_is_formula else "weighed"
    d.text((x2, y0 + 14), "actual (cow.csv)", font=f(16), fill=(80, 80, 80))
    d.text((x2, y0 + 38), f"{row.actual:.0f} kg", font=f(56, True), fill=(55, 65, 81))
    d.text((x2, y0 + 108), f"error {err:+.0f} kg ({err / row.actual * 100:+.1f}%)   label: {lab}",
           font=f(15), fill=(185, 28, 28) if abs(err) / row.actual > .10 else (6, 95, 70))

    pm = w["per_model"]
    lines = [
        f"heads   SVR {pm['svr']:.0f}  ·  Ridge {pm['ridge']:.0f}  ·  MLP {pm['mlp']:.0f}   (spread {w['spread_kg']:.0f} kg)",
        f"quality {q['flag']}  ·  zones {q['zones_found']}/7  ·  segment {res['inference_ms']} ms  ·  "
        f"weight {w['inference_ms']} ms  ·  {w['model_version']}",
        f"cross-validated (animal held out of the heads): {row.pred_ensemble:.0f} kg, "
        f"error {row.pred_ensemble - row.actual:+.0f} kg  ·  live number above is in-sample",
    ]
    for i, t in enumerate(lines):
        d.text((40, y0 + 150 + i * 24), t, font=f(14), fill=(60, 60, 60))
    c.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--url", default="http://localhost:3010/api/analyze")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "ml" / "demo")
    args = ap.parse_args()

    oof = pd.read_csv(ENSEMBLE / "oof_predictions.csv")
    cow = pd.read_csv(Path(args.images).parent / "cow.csv", encoding="utf-8-sig")
    cow["animal_id"] = cow["ID"].str.upper().str.replace(r"^C0*", "", regex=True).astype(int) \
        .map(lambda n: f"C{n:03d}")
    cow["label_is_formula"] = (cow["Weight"] - (5.854 * cow["L"] - 595.98)).abs() < .01
    oof = oof.merge(cow[["animal_id", "label_is_formula"]], on="animal_id")
    rows = pick(oof, args.n)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows.itertuples():
        path = next(args.images.glob(f"*/images/{row.path}"))
        photo = Image.open(path)
        # the scan page downsizes to 600 px wide JPEG q0.6 before upload - do the same
        buf = io.BytesIO()
        photo.convert("RGB").resize((600, round(600 * photo.height / photo.width))).save(buf, "JPEG", quality=60)
        r = requests.post(args.url, files={"file": ("scan.jpg", buf.getvalue(), "image/jpeg")}, timeout=120)
        res = r.json()
        if not r.ok or not res.get("weight"):
            print(f"{row.animal_id}: HTTP {r.status_code} {res.get('error') or 'no weight in response'}")
            continue
        out = args.out / f"demo_{row.animal_id}.png"
        card(photo, res, row, out)
        w = res["weight"]
        summary.append(dict(animal=row.animal_id, actual=row.actual, predicted=w["kg"],
                            error=round(w["kg"] - row.actual, 1), label_formula=bool(row.label_is_formula),
                            flag=res["quality"]["flag"], per_model=w["per_model"], card=out.name))
        print(f"{row.animal_id}: actual {row.actual:.0f} | predicted {w['kg']:.0f} "
              f"(err {w['kg'] - row.actual:+.0f}) | {res['quality']['flag']} | -> {out}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
