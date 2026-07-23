"""
Script tự động duyệt qua toàn bộ folder ảnh test/val:
1. Nạp model đã train (load checkpoint .hdf5).
2. Tạo thư mục 'mask_test' ngay bên cạnh thư mục chứa ảnh để lưu các file mask dự đoán dạng PNG (0=Đen, 128=Xám, 255=Trắng).
3. Tạo thư mục 'show_predict' ngay bên cạnh thư mục chứa ảnh để lưu toàn bộ dãy 3 ảnh song song:
   [Ảnh gốc X-ray | Ground Truth | Prediction Overlay (Xanh=Lành, Đỏ=Ác)]
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from data_generators import tf_data_generator
from utils.general_utils import join_paths, suppress_warnings
from models.model import prepare_model
from show_predictions import compute_class_dice, get_colored_overlay


def resolve_path(cfg, cfg_path):
    if not cfg_path:
        return None
    if os.path.exists(cfg_path):
        return cfg_path
    fixed_typo = cfg_path.replace("Datatset_", "Dataset_")
    if os.path.exists(fixed_typo):
        return fixed_typo
    current_drive = os.path.splitdrive(os.path.abspath(cfg.WORK_DIR))[0]
    if current_drive:
        path_drive = os.path.splitdrive(cfg_path)[0]
        if path_drive:
            alt_path = cfg_path.replace(path_drive, current_drive)
            if os.path.exists(alt_path):
                return alt_path
            alt_fixed = fixed_typo.replace(path_drive, current_drive)
            if os.path.exists(alt_fixed):
                return alt_fixed
    return join_paths(cfg.WORK_DIR, fixed_typo)


def process_test_dataset(cfg: DictConfig):
    suppress_warnings()
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

    # 1. Nạp dữ liệu tập TEST (hoặc VAL nếu không có TEST)
    mode = "TEST" if hasattr(cfg.DATASET, "TEST") and cfg.DATASET.TEST.IMAGES_PATH else "VAL"
    test_generator = tf_data_generator.DataGenerator(cfg, mode=mode)
    
    # Dò tìm thư mục chứa ảnh tập TEST/VAL
    images_dir_raw = cfg.DATASET[mode].IMAGES_PATH
    images_dir = resolve_path(cfg, images_dir_raw)
    
    # Lấy gốc thư mục Dataset_Bone_Tumor bên ngoài
    root_dataset_dir = os.path.dirname(os.path.dirname(images_dir))
    if not os.path.exists(root_dataset_dir):
        root_dataset_dir = os.path.dirname(images_dir)

    # 2. Tạo 2 thư mục xuất kết quả nằm trực tiếp ở gốc thư mục Dataset_Bone_Tumor bên ngoài
    mask_test_dir = os.path.join(root_dataset_dir, "mask_test")
    show_predict_dir = os.path.join(root_dataset_dir, "show_predict")

    os.makedirs(mask_test_dir, exist_ok=True)
    os.makedirs(show_predict_dir, exist_ok=True)

    print(f"\n Thư mục chứa ảnh test: {images_dir}")
    print(f" Thư mục xuất mask_test: {mask_test_dir}")
    print(f" Thư mục xuất show_predict: {show_predict_dir}\n")

    # 3. Khởi tạo Model & Nạp trọng số
    model = prepare_model(cfg, training=False)
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )

    print(f"Loading weights from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Không tìm thấy file trọng số tại: {checkpoint_path}"
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)

    # Tích lũy chỉ số đánh giá trên toàn bộ tập dữ liệu
    all_dice_benign = []
    all_dice_malignant = []
    all_dice_overall = []

    global_inter_benign = 0
    global_union_benign = 0
    global_inter_malignant = 0
    global_union_malignant = 0

    # Danh sách dữ liệu ghi file CSV chi tiết từng ảnh
    csv_rows = [["Image_Name", "Dice_Benign", "Dice_Malignant", "Dice_Overall"]]

    # Lấy danh sách tên file ảnh gốc
    image_files = [os.path.basename(p) for p in test_generator.images_paths]

    print(f"\n>>> Đang tự động tạo Mask và Dãy 3 ảnh hiển thị cho toàn bộ tập {mode}...")
    
    for idx in tqdm(range(len(test_generator))):
        images, masks = test_generator[idx]
        img_filename = image_files[idx]
        base_name = os.path.splitext(img_filename)[0]

        # Predict
        preds = model.predict_on_batch(images)
        if len(model.outputs) > 1:
            preds = preds[0]

        img = images[0]
        img_show = img[:, :, 1] if cfg.SHOW_CENTER_CHANNEL_IMAGE else img

        pred_mask = np.argmax(preds[0], axis=-1).astype(np.uint8)
        gt_mask = np.argmax(masks[0], axis=-1).astype(np.uint8)

        # A. XUẤT FILE MASK TEST (0=Đen, 128=Xám u lành, 255=Trắng u ác)
        vis_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        vis_mask[pred_mask == 1] = 128
        vis_mask[pred_mask == 2] = 255
        
        mask_save_path = os.path.join(mask_test_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_save_path, vis_mask)

        # B. XUẤT DÃY 3 ẢNH SONG SONG SHOW PREDICT
        dice_benign = compute_class_dice(gt_mask, pred_mask, class_id=1)
        dice_malignant = compute_class_dice(gt_mask, pred_mask, class_id=2)

        if dice_benign is not None:
            all_dice_benign.append(dice_benign)
        if dice_malignant is not None:
            all_dice_malignant.append(dice_malignant)

        # Tích lũy Dice tổng thể vùng u (lành + ác) cho ảnh này
        true_tumor = (gt_mask > 0)
        pred_tumor = (pred_mask > 0)
        inter_tumor = np.sum(true_tumor & pred_tumor)
        union_tumor = np.sum(true_tumor) + np.sum(pred_tumor)
        
        dice_ov_val = (2.0 * inter_tumor) / union_tumor if union_tumor > 0 else None
        if dice_ov_val is not None:
            all_dice_overall.append(dice_ov_val)

        # Ghi dòng chi tiết từng ảnh vào CSV
        str_b_csv = f"{dice_benign:.4f}" if dice_benign is not None else "N/A"
        str_m_csv = f"{dice_malignant:.4f}" if dice_malignant is not None else "N/A"
        str_ov_csv = f"{dice_ov_val:.4f}" if dice_ov_val is not None else "N/A"
        csv_rows.append([img_filename, str_b_csv, str_m_csv, str_ov_csv])

        # Tích lũy cho Global Dataset Dice
        b_gt, b_pr = (gt_mask == 1), (pred_mask == 1)
        global_inter_benign += np.sum(b_gt & b_pr)
        global_union_benign += np.sum(b_gt) + np.sum(b_pr)

        m_gt, m_pr = (gt_mask == 2), (pred_mask == 2)
        global_inter_malignant += np.sum(m_gt & m_pr)
        global_union_malignant += np.sum(m_gt) + np.sum(m_pr)

        vis_gt = np.zeros_like(gt_mask, dtype=np.uint8)
        vis_gt[gt_mask == 1] = 128
        vis_gt[gt_mask == 2] = 255

        blended_pred = get_colored_overlay(img_show, pred_mask, alpha=0.45)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.squeeze(img_show), cmap='gray')
        axes[0].set_title(f"Original X-ray ({base_name})")
        axes[0].axis('off')

        axes[1].imshow(vis_gt, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        # Hiển thị tiêu đề Dice Score
        str_b = f"{dice_benign:.4f}" if dice_benign is not None else "N/A"
        str_m = f"{dice_malignant:.4f}" if dice_malignant is not None else "N/A"

        axes[2].imshow(blended_pred)
        axes[2].set_title(f"Prediction (Dice B: {str_b}, M: {str_m})")
        axes[2].axis('off')

        plt.tight_layout()

        show_predict_path = os.path.join(show_predict_dir, f"{base_name}_predict.png")
        plt.savefig(show_predict_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    # 5. TÍNH TOÁN VÀ IN BÁO CÁO TỔNG THỂ CỦA TẬP PREDICT
    mean_dice_b = np.mean(all_dice_benign) if len(all_dice_benign) > 0 else 0.0
    mean_dice_m = np.mean(all_dice_malignant) if len(all_dice_malignant) > 0 else 0.0
    mean_dice_ov = np.mean(all_dice_overall) if len(all_dice_overall) > 0 else 0.0

    global_dice_b = (2.0 * global_inter_benign) / global_union_benign if global_union_benign > 0 else 0.0
    global_dice_m = (2.0 * global_inter_malignant) / global_union_malignant if global_union_malignant > 0 else 0.0

    summary_str = f"""
======================================================================
📊 KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ TRÊN TẬP {mode} ({len(test_generator)} ẢNH)
======================================================================
1. ĐÁNH GIÁ DỰA TRÊN TRUNG BÌNH TỪNG ẢNH (Per-Image Mean Dice):
   - Mean Dice U Lành  (Benign - Class 1)   : {mean_dice_b:.4f} ({mean_dice_b*100:.2f}%)
   - Mean Dice U Ác    (Malignant - Class 2): {mean_dice_m:.4f} ({mean_dice_m*100:.2f}%)
   - Mean Dice Vùng U  (Overall Tumor)      : {mean_dice_ov:.4f} ({mean_dice_ov*100:.2f}%)

2. ĐÁNH GIÁ TỔNG TÍCH LŨY TOÀN BỘ TẬP DỮ LIỆU (Global Dataset Dice):
   - Global Dice U Lành (Benign)            : {global_dice_b:.4f} ({global_dice_b*100:.2f}%)
   - Global Dice U Ác   (Malignant)         : {global_dice_m:.4f} ({global_dice_m*100:.2f}%)
======================================================================
"""
    print(summary_str)

    # Lưu file CSV chi tiết từng ảnh
    csv_log_path = os.path.join(root_dataset_dir, f"prediction_details_{mode.lower()}.csv")
    import csv
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    # Lưu file txt tổng kết vào thư mục xuất kết quả
    summary_txt_path = os.path.join(root_dataset_dir, f"summary_evaluation_{mode.lower()}.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_str)

    print(f"🎉 HOÀN THÀNH!")
    print(f" MASK TEST lưu tại: {mask_test_dir}")
    print(f" SHOW PREDICT lưu tại: {show_predict_dir}")
    print(f" FILE LOG CSV CHI TIẾT TỪNG ẢNH: {csv_log_path}")
    print(f" FILE BÁO CÁO TỔNG THỂ TXT: {summary_txt_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    process_test_dataset(cfg)


if __name__ == "__main__":
    main()
