"""
Script hiển thị kết quả dự đoán (Visualization) của model NaLaFormer-TransUNet.
Hiển thị 3 ảnh side-by-side:
  1. Original X-Ray (ID: ...)
  2. Ground Truth (0=Đen, 128=Xám, 255=Trắng)
  3. Prediction Overlay (U lành màu Xanh lá, U ác màu Đỏ) chồng lên ảnh gốc.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import hydra
from omegaconf import DictConfig

from data_generators import tf_data_generator
from utils.general_utils import join_paths, suppress_warnings
from models.model import prepare_model


def compute_class_dice(y_true, y_pred, class_id):
    """
    Tính Dice score cho một class cụ thể.
    """
    true_mask = (y_true == class_id)
    pred_mask = (y_pred == class_id)
    
    intersection = np.sum(true_mask & pred_mask)
    volume_sum = np.sum(true_mask) + np.sum(pred_mask)
    
    if volume_sum == 0:
        return 1.0  # Không có u trong cả ground truth và dự đoán -> Dice = 1.0
    return (2.0 * intersection) / volume_sum


def get_colored_overlay(image, pred_mask, alpha=0.4):
    """
    Tạo ảnh blended giữa ảnh gốc (grayscale -> RGB) và màu overlay:
      - U lành (class 1) -> Xanh lá [0, 255, 0]
      - U ác (class 2) -> Đỏ [255, 0, 0]
    """
    # Đảm bảo ảnh gốc là RGB để vẽ màu
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image_rgb = cv2.cvtColor(np.squeeze(image), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
        
    # Scale ảnh về [0, 255] nếu nó đang là [0, 1]
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)

    h, w = pred_mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Gán màu:
    overlay[pred_mask == 1] = [0, 255, 0]  # U lành -> Xanh lá
    overlay[pred_mask == 2] = [255, 0, 0]  # U ác -> Đỏ
    
    # Tạo mặt nạ vùng có u (class 1 hoặc 2)
    tumor_mask = (pred_mask > 0)[:, :, np.newaxis]
    
    # Blend màu
    blended = image_rgb.copy()
    blended_overlay = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
    
    # Chỉ áp dụng overlay lên những pixel được dự đoán là u
    np.copyto(blended, blended_overlay, where=tumor_mask)
    return blended


def predict_and_visualize(cfg: DictConfig):
    suppress_warnings()
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1  # Predict từng ảnh một

    # Khởi tạo data generator cho tập Validation
    val_generator = tf_data_generator.DataGenerator(cfg, mode="VAL")
    
    # Khởi tạo model
    model = prepare_model(cfg, training=False)
    
    # Đường dẫn trọng số
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )
    
    print(f"Loading weights from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Không tìm thấy file trọng số tại: {checkpoint_path}"
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    
    # Check dataset
    mask_available = (cfg.DATASET.VAL.MASK_PATH is not None and 
                      str(cfg.DATASET.VAL.MASK_PATH).lower() != "none")
    
    print("\n>>> Bắt đầu hiển thị dự đoán. Bấm [Q] hoặc tắt cửa sổ ảnh để xem ảnh tiếp theo.")
    
    for idx, (images, masks) in enumerate(val_generator):
        # Predict
        preds = model.predict_on_batch(images)
        if len(model.outputs) > 1:
            preds = preds[0]
            
        img = images[0]
        # Lấy kênh center nếu được thiết lập
        if cfg.SHOW_CENTER_CHANNEL_IMAGE:
            img_show = img[:, :, 1]
        else:
            img_show = img

        # Lấy nhãn class dự đoán (0, 1, 2)
        pred_mask = np.argmax(preds[0], axis=-1).astype(np.uint8)
        
        # Lấy nhãn class ground truth (0, 1, 2)
        gt_mask = np.argmax(masks[0], axis=-1).astype(np.uint8)
        
        # Tính Dice scores
        dice_benign = compute_class_dice(gt_mask, pred_mask, class_id=1)
        dice_malignant = compute_class_dice(gt_mask, pred_mask, class_id=2)
        
        # 1. Vẽ Ground Truth theo yêu cầu: 0=Đen, 128=Xám, 255=Trắng
        vis_gt = np.zeros_like(gt_mask, dtype=np.uint8)
        vis_gt[gt_mask == 1] = 128  # U lành -> xám
        vis_gt[gt_mask == 2] = 255  # U ác -> trắng
        
        # 2. Vẽ Prediction với overlay màu (U lành -> xanh, U ác -> đỏ)
        blended_pred = get_colored_overlay(img_show, pred_mask, alpha=0.45)
        
        # Hiển thị Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Subplot 1: Ảnh gốc
        axes[0].imshow(np.squeeze(img_show), cmap='gray')
        axes[0].set_title(f"Original X-ray (ID: {idx})")
        axes[0].axis('off')
        
        # Subplot 2: Ground Truth
        axes[1].imshow(vis_gt, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        # Subplot 3: Dự đoán overlay
        axes[2].imshow(blended_pred)
        axes[2].set_title(f"Prediction (Dice B: {dice_benign:.4f}, M: {dice_malignant:.4f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Đợi người dùng tắt ảnh để sang ảnh tiếp theo
        plt.show()
        
        # Giới hạn xem tối đa 10 ảnh mẫu
        if idx >= 9:
            print("\nĐã xem hết 10 ảnh mẫu.")
            break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    predict_and_visualize(cfg)


if __name__ == "__main__":
    main()
