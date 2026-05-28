#!/usr/bin/env python3
"""Download COCO val2017 images and resize to 2560x1920 (5MP security camera resolution)."""

import os
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
NUM_IMAGES = 50
TARGET_W, TARGET_H = 2560, 1920

COCO_VAL_URL = "http://images.cocodataset.org/val2017"

IMAGE_IDS = [
    "000000000139", "000000000285", "000000000632", "000000000724", "000000000776",
    "000000000785", "000000000802", "000000000872", "000000000885", "000000001000",
    "000000001268", "000000001296", "000000001353", "000000001425", "000000001490",
    "000000001503", "000000001532", "000000001584", "000000001675", "000000001761",
    "000000001818", "000000001993", "000000002006", "000000002149", "000000002153",
    "000000002157", "000000002261", "000000002299", "000000002431", "000000002473",
    "000000002532", "000000002587", "000000002592", "000000002685", "000000002923",
    "000000003156", "000000003255", "000000003501", "000000003553", "000000003845",
    "000000004134", "000000004395", "000000004495", "000000005001", "000000005037",
    "000000005060", "000000005193", "000000005529", "000000005586", "000000005654",
]


def main():
    from PIL import Image

    os.makedirs(DATA_DIR, exist_ok=True)
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    if len(existing) >= NUM_IMAGES:
        # Verify resolution of first image
        sample = Image.open(os.path.join(DATA_DIR, existing[0]))
        if sample.size == (TARGET_W, TARGET_H):
            print(f"Already have {len(existing)} images at {TARGET_W}x{TARGET_H} in {DATA_DIR}, skipping.")
            return
        else:
            print(f"Images exist but at wrong resolution ({sample.size}), re-downloading...")

    tmp_dir = os.path.join(DATA_DIR, "_raw")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Downloading {NUM_IMAGES} COCO val2017 images...")
    for i, img_id in enumerate(IMAGE_IDS[:NUM_IMAGES]):
        fname = f"{img_id}.jpg"
        tmp_path = os.path.join(tmp_dir, fname)
        if not os.path.exists(tmp_path):
            url = f"{COCO_VAL_URL}/{fname}"
            sys.stdout.write(f"\r  Downloading [{i+1}/{NUM_IMAGES}] {fname}")
            sys.stdout.flush()
            try:
                urllib.request.urlretrieve(url, tmp_path)
            except Exception as e:
                print(f"\n  WARNING: Failed to download {fname}: {e}")
                continue

    print(f"\nResizing to {TARGET_W}x{TARGET_H} (security camera resolution)...")
    count = 0
    for fname in sorted(os.listdir(tmp_dir)):
        if not fname.endswith(".jpg"):
            continue
        src = os.path.join(tmp_dir, fname)
        dst = os.path.join(DATA_DIR, fname)
        if os.path.exists(dst):
            img = Image.open(dst)
            if img.size == (TARGET_W, TARGET_H):
                count += 1
                continue
        img = Image.open(src).convert("RGB")
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        img.save(dst, quality=95)
        count += 1
        sys.stdout.write(f"\r  Resized [{count}/{NUM_IMAGES}]")
        sys.stdout.flush()

    print(f"\nDone. {count} images at {TARGET_W}x{TARGET_H} in {DATA_DIR}")


if __name__ == "__main__":
    main()
