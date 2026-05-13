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
    Weighted Dice Loss để trị mất cân bằng class.
    Trọng số: [Nền: 0.1, U Lành: 0.3, U Ác: 0.8]
    U ác thiểu số nên được đặt trọng số cao nhất.
    """
    weights = tf.constant([0.1, 0.3, 0.8], dtype=tf.float32)
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Tính Dice cho từng class (axis = [1, 2] tức là chiều Height, Width)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    
    # Loss = 1 - Dice, sau đó nhân với trọng số phạt
    loss = (1.0 - dice) * weights
    
    # Lấy trung bình loss của cả 3 class để backpropagate
    return tf.reduce_mean(loss)