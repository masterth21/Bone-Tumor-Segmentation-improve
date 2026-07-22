"""
Script chuẩn bị TOÀN BỘ dữ liệu (cả ảnh có U và ảnh KHÔNG U):
  1. Duyệt tất cả ảnh trong thư mục images.
  2. Ảnh có file .json annotation -> Tạo mask polygon (0=đen, 128=xám u lành, 255=trắng u ác).
  3. Ảnh KHÔNG có file .json annotation -> Tạo mask đen toàn bộ (0 = không u).
  4. Chia ngẫu nhiên 80% train_full / 20% val_full.

Cấu trúc output:
  D:/TumorBone/data/BTXRD/split_full/
    ├── train_full/
    │   ├── images/
    │   └── mask/
    └── val_full/
        ├── images/
        └── mask/
"""

import json
import os
import glob
import random
from PIL import Image, ImageDraw
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================
DATA_ROOT = r"D:\TumorBone\data\BTXRD\BTXRD"
OUTPUT_ROOT = r"D:\TumorBone\data\BTXRD\split_full"
TRAIN_RATIO = 0.8   # 80% train, 20% val
SEED = 42
IMG_SIZE = (320, 320)  # Resize cho phù hợp với config model

# Label mapping:  3 classes (pixel values cho hiển thị)
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


# ============================================================================
# HÀM TẠO MASK TỪ JSON (POLYGON ONLY)
# ============================================================================
def json_to_mask(json_path, img_width, img_height):
    """
    Đọc file JSON (LabelMe) → tạo mask array (H, W) với giá trị class.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight", img_height)
    w = data.get("imageWidth", img_width)

    mask = Image.new("L", (w, h), 0)  # background = 0 (đen)
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

    return mask


# ============================================================================
# MAIN
# ============================================================================
def main():
    images_dir = os.path.join(DATA_ROOT, "images")
    annot_dir = os.path.join(DATA_ROOT, "Annotations")

    # 1. Tìm tất cả các file ảnh trong thư mục images
    all_image_files = []
    for ext in ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]:
        all_image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    print(f"Tổng số ảnh tìm thấy trong thư mục: {len(all_image_files)}")

    if len(all_image_files) == 0:
        print("KHÔNG TÌM THẤY ẢNH! Kiểm tra lại đường dẫn.")
        return

    # 2. Ghép cặp với file JSON (nếu có)
    pairs = []
    annotated_count = 0
    unannotated_count = 0

    for img_path in all_image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(annot_dir, base + ".json")

        if os.path.exists(json_path):
            pairs.append((img_path, json_path))
            annotated_count += 1
        else:
            pairs.append((img_path, None))  # Không có annotation -> mask đen
            unannotated_count += 1

    print(f"  - Số ảnh có u/annotation: {annotated_count}")
    print(f"  - Số ảnh KHÔNG u (mask đen): {unannotated_count}")

    # 3. Shuffle ngẫu nhiên và chia train_full (80%) / val_full (20%)
    random.seed(SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"\nPhân chia dữ liệu toàn bộ (split_full):")
    print(f"  - Tập train_full: {len(train_pairs)} ảnh")
    print(f"  - Tập val_full:   {len(val_pairs)} ảnh")

    # 4. Tạo thư mục output
    for split_folder in ["train_full", "val_full"]:
        for sub in ["images", "mask"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split_folder, sub), exist_ok=True)

    # 5. Xử lý và lưu ảnh/mask
    splits = [("train_full", train_pairs), ("val_full", val_pairs)]

    for split_name, split_pairs in splits:
        print(f"\n{'='*50}")
        print(f"  Đang xử lý: {split_name} ({len(split_pairs)} ảnh)")
        print(f"{'='*50}")

        for img_path, json_path in tqdm(split_pairs, desc=split_name):
            base = os.path.splitext(os.path.basename(img_path))[0]

            # Đọc ảnh gốc
            img = Image.open(img_path)
            orig_w, orig_h = img.size

            if json_path is not None:
                # Ảnh có annotation -> tạo mask polygon
                mask = json_to_mask(json_path, orig_w, orig_h)
            else:
                # Ảnh không có annotation -> tạo mask đen hoàn toàn
                mask = Image.new("L", (orig_w, orig_h), 0)

            # Resize ảnh và mask
            img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
            mask_resized = mask.resize(IMG_SIZE, Image.NEAREST)

            # Lưu ảnh và mask dạng PNG
            out_img = os.path.join(OUTPUT_ROOT, split_name, "images", base + ".png")
            out_mask = os.path.join(OUTPUT_ROOT, split_name, "mask", base + ".png")

            img_resized.save(out_img)
            mask_resized.save(out_mask)

    # Thống kê kết quả
    print(f"\n{'='*50}")
    print(f"  ✅ HOÀN TẤT TẠO DATASET FULL!")
    print(f"{'='*50}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"  - train_full/images & train_full/mask: {len(train_pairs)} mẫu")
    print(f"  - val_full/images   & val_full/mask:   {len(val_pairs)} mẫu")
    print(f"  - Kích thước ảnh: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  - Chú thích Mask: 0=Đen (Không u / BG), 128=Xám (U lành), 255=Trắng (U ác)")
    print(f"\n  Khi muốn train với tập full này, cập nhật config.yaml:")
    print(f'    DATASET:')
    print(f'      TRAIN:')
    print(f'        IMAGES_PATH: "D:/TumorBone/data/BTXRD/split_full/train_full/images"')
    print(f'        MASK_PATH:   "D:/TumorBone/data/BTXRD/split_full/train_full/mask"')
    print(f'      VAL:')
    print(f'        IMAGES_PATH: "D:/TumorBone/data/BTXRD/split_full/val_full/images"')
    print(f'        MASK_PATH:   "D:/TumorBone/data/BTXRD/split_full/val_full/mask"')


if __name__ == "__main__":
    main()
