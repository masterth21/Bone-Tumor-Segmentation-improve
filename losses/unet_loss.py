"""
UNet 3+ Loss
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from .loss import focal_loss, ssim_loss, iou_loss

def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    """
    f_loss = focal_loss(y_true, y_pred)
    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jacard_loss = iou_loss(y_true, y_pred)

    return f_loss + ms_ssim_loss + jacard_loss

def weighted_dice_loss(y_true, y_pred):
    """
    Hybrid Categorical CrossEntropy + Weighted Dice Loss (Chuẩn nnU-Net).
    - Tính Dice tổng trên toàn Batch (axis=[0, 1, 2]) giúp triệt tiêu hiện tượng nổ Loss
      khi gặp ảnh trống hoặc nhiễu pixel nhỏ.
    - Kết hợp Categorical CrossEntropy giúp gradient mịn màng và hội tụ cực kỳ ổn định.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

    # 1. Categorical CrossEntropy Loss (Trọng số phạt: BG=0.1, Lành=0.4, Ác=1.0)
    class_weights = tf.constant([0.1, 0.4, 1.0], dtype=tf.float32)
    ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # (B, H, W)
    
    # Nhân trọng số class vào CE
    weight_map = tf.reduce_sum(y_true * class_weights, axis=-1)      # (B, H, W)
    weighted_ce = tf.reduce_mean(ce_loss * weight_map)

    # 2. Weighted Dice Loss trên toàn Batch (axis=[0, 1, 2])
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2])
    
    dice_per_class = (2.0 * intersection + 1e-5) / (union + 1e-5)
    dice_loss = tf.reduce_sum((1.0 - dice_per_class) * class_weights) / tf.reduce_sum(class_weights)

    # Tổng hợp 2 Loss
    return weighted_ce + dice_loss