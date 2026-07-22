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

    # 1. Nạp dữ liệu tập VAL / TEST
    val_generator = tf_data_generator.DataGenerator(cfg, mode="VAL")
    
    # Dò tìm thư mục chứa ảnh tập VAL/TEST
    images_dir_raw = cfg.DATASET.VAL.IMAGES_PATH
    images_dir = resolve_path(cfg, images_dir_raw)
    parent_dir = os.path.dirname(images_dir)

    # 2. Tạo 2 thư mục xuất kết quả ngay bên cạnh thư mục ảnh
    mask_test_dir = os.path.join(parent_dir, "mask_test")
    show_predict_dir = os.path.join(parent_dir, "show_predict")

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

    # 4. Lấy danh sách tên file ảnh gốc
    image_files = [os.path.basename(p) for p in val_generator.images_paths]

    print("\n>>> Đang tự động tạo Mask và Dãy 3 ảnh hiển thị cho toàn bộ tập Test/Val...")
    
    for idx in tqdm(range(len(val_generator))):
        images, masks = val_generator[idx]
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

        axes[2].imshow(blended_pred)
        axes[2].set_title(f"Prediction (Dice B: {dice_benign:.4f}, M: {dice_malignant:.4f})")
        axes[2].axis('off')

        plt.tight_layout()

        show_predict_path = os.path.join(show_predict_dir, f"{base_name}_predict.png")
        plt.savefig(show_predict_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"\n🎉 HOÀN THÀNH!")
    print(f" MASK TEST lưu tại: {mask_test_dir}")
    print(f" SHOW PREDICT lưu tại: {show_predict_dir}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    process_test_dataset(cfg)


if __name__ == "__main__":
    main()
