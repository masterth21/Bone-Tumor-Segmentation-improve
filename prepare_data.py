"""
Script chuẩn bị dữ liệu cho NaLaFormer:
  1. Đọc JSON annotations (LabelMe format) → Tạo mask ảnh (PNG)
  2. Chia train/val (80/20) random
  3. Copy ảnh + mask vào cấu trúc thư mục chuẩn

Cấu trúc output:
  D:/TumorBone/data/BTXRD/split/
    ├── train/
    │   ├── images/
    │   └── mask/
    └── val/
        ├── images/
        └── mask/

Labels:
  0 = background
  1 = osteosarcoma + other mt  (gộp thành 1 class "tumor")
"""

import json
import os
import glob
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================
DATA_ROOT = r"D:\TumorBone\data\BTXRD\BTXRD"
OUTPUT_ROOT = r"D:\TumorBone\data\BTXRD\split"
TRAIN_RATIO = 0.8   # 80% train, 20% val
SEED = 42
IMG_SIZE = (320, 320)  # Resize cho phù hợp với config model

# Label mapping:  3 classes (pixel values cho dễ nhìn)
#   0   = đen   → background (không u)
#   128 = xám   → lành tính (Benign Tumor - BT)
#   255 = trắng → ác tính  (Malignant Tumor - MT)
LABEL_MAP = {
    # Lành tính (BT) → 128 (xám)
    "osteochondroma": 128,
    "multiple osteochondromas": 128,
    "simple bone cyst": 128,
    "other bt": 128,
    "giant cell tumor": 128,
    "synovial osteochondroma": 128,
    "osteofibroma": 128,
    # Ác tính (MT) → 255 (trắng)
    "osteosarcoma": 255,
    "other mt": 255,
}

# Mapping pixel value → class index (dùng trong data generator)
# 0 → class 0,  128 → class 1,  255 → class 2
PIXEL_TO_CLASS = {0: 0, 128: 1, 255: 2}


# ============================================================================
# HÀM TẠO MASK TỪ JSON
# ============================================================================
def json_to_mask(json_path, img_width, img_height):
    """
    Đọc file JSON (LabelMe) → tạo mask array (H, W) với giá trị class.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy kích thước ảnh từ JSON nếu có
    h = data.get("imageHeight", img_height)
    w = data.get("imageWidth", img_width)

    mask = Image.new("L", (w, h), 0)  # background = 0
    draw = ImageDraw.Draw(mask)

    for shape in data["shapes"]:
        label = shape["label"]
        class_id = LABEL_MAP.get(label, 0)
        if class_id == 0:
            continue

        shape_type = shape["shape_type"]
        points = shape["points"]

        if shape_type == "polygon":
            poly = [(p[0], p[1]) for p in points]
            if len(poly) >= 3:
                draw.polygon(poly, fill=class_id)
        # Bỏ qua rectangle (bounding box) — chỉ dùng polygon cho mask chính xác

    return mask


# ============================================================================
# MAIN
# ============================================================================
def main():
    images_dir = os.path.join(DATA_ROOT, "images")
    annot_dir = os.path.join(DATA_ROOT, "Annotations")

    # Tìm tất cả ảnh CÓ annotation tương ứng
    json_files = glob.glob(os.path.join(annot_dir, "*.json"))
    print(f"Tìm thấy {len(json_files)} file annotation")

    pairs = []
    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]
        # Tìm ảnh tương ứng (jpeg hoặc jpg hoặc png)
        img_path = None
        for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
            candidate = os.path.join(images_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path:
            pairs.append((img_path, jf))

    print(f"Tìm thấy {len(pairs)} cặp ảnh-annotation hợp lệ")

    if len(pairs) == 0:
        print("KHÔNG TÌM THẤY DỮ LIỆU! Kiểm tra lại đường dẫn.")
        return

    # Shuffle và chia 3 tập Train / Val / Test (tỷ lệ 8:1:1)
    random.seed(SEED)
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"\nChia dữ liệu 3 tập theo tỷ lệ 8:1:1 (Ảnh CÓ Annotation):")
    print(f"  1. Train (80%): {len(train_pairs)} ảnh (Huấn luyện mô hình)")
    print(f"  2. Val   (10%): {len(val_pairs)} ảnh (Theo dõi loss & early stopping khi train)")
    print(f"  3. Test  (10%): {len(test_pairs)} ảnh (Tập kiểm thử độc lập hoàn toàn)")

    # Tạo thư mục output cho 3 tập
    for split in ["train", "val", "test"]:
        for sub in ["images", "mask"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, sub), exist_ok=True)

    # Xử lý từng split
    splits = [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]
    for split_name, split_pairs in splits:
        print(f"\n{'='*50}")
        print(f"  Đang xử lý: {split_name} ({len(split_pairs)} ảnh)")
        print(f"{'='*50}")

        for img_path, json_path in tqdm(split_pairs, desc=split_name):
            base = os.path.splitext(os.path.basename(img_path))[0]

            # Đọc ảnh gốc để lấy kích thước
            img = Image.open(img_path)
            orig_w, orig_h = img.size

            # Tạo mask từ JSON
            mask = json_to_mask(json_path, orig_w, orig_h)

            # Resize ảnh và mask
            img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
            mask_resized = mask.resize(IMG_SIZE, Image.NEAREST)  # NEAREST để giữ class ID

            # Lưu
            out_img = os.path.join(OUTPUT_ROOT, split_name, "images", base + ".png")
            out_mask = os.path.join(OUTPUT_ROOT, split_name, "mask", base + ".png")

            img_resized.save(out_img)
            mask_resized.save(out_mask)

    # Thống kê
    print(f"\n{'='*50}")
    print(f"  ✅ HOÀN TẤT CHIA DỮ LIỆU 3 TẬP (8:1:1)!")
    print(f"{'='*50}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Train (80%): {len(train_pairs)} ảnh")
    print(f"  Val   (10%): {len(val_pairs)} ảnh")
    print(f"  Test  (10%): {len(test_pairs)} ảnh")
    print(f"  Resize: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  Classes: 0=background, 1=benign, 2=malignant")


if __name__ == "__main__":
    main()
