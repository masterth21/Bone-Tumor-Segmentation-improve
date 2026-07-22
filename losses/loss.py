"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def iou(y_true, y_pred, smooth=1.e-7):
    """
    Calculate intersection over union (IoU) between images.
    Input shape: Batch x Height x Width x #Classes.
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=[1, 2, 3])
    union = K.sum(y_true_f, [1, 2, 3]) + K.sum(y_pred_f, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1.0 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss bảo vệ chống NaN
    """
    gamma = 2.0
    alpha = 4.0
    epsilon = 1.e-7

    y_true_c = tf.cast(y_true, tf.float32)
    y_pred_c = tf.clip_by_value(tf.cast(y_pred, tf.float32), epsilon, 1.0 - epsilon)

    ce = y_true_c * -tf.math.log(y_pred_c)
    weight = y_true_c * tf.math.pow(1.0 - y_pred_c, gamma)
    fl = alpha * weight * ce
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred, smooth=1.e-7):
    """
    Structural Similarity Index loss bảo vệ chống NaN
    """
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    
    ssim_value = tf.image.ssim(y_true_f, y_pred_f, max_val=1.0)
    return K.mean(1.0 - ssim_value + smooth, axis=0)


class DiceCoefficient(tf.keras.metrics.Metric):
    """
    Dice coefficient metric. Can be used to calculate dice on probabilities
    or on their respective classes
    """

    def __init__(self, post_processed: bool,
                 classes: int,
                 name='dice_coef',
                 **kwargs):
        """
        Set post_processed=False if dice coefficient needs to be calculated
        on probabilities. Set post_processed=True if probabilities needs to
        be first converted/mapped into their respective class.
        """
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_value = self.add_weight(name='dice_value', initializer='zeros')
        self.post_processed = post_processed
        self.classes = classes
        if self.classes == 1:
            self.axis = [1, 2, 3]
        else:
            self.axis = [1, 2, ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.post_processed:
            if self.classes == 1:
                y_true_ = y_true
                y_pred_ = tf.where(y_pred > .5, 1.0, 0.0)
            else:
                y_true_ = tf.math.argmax(y_true, axis=-1, output_type=tf.int32)
                y_pred_ = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)
                y_true_ = tf.cast(y_true_, dtype=tf.float32)
                y_pred_ = tf.cast(y_pred_, dtype=tf.float32)
        else:
            y_true_, y_pred_ = y_true, y_pred

        self.dice_value.assign(self.dice_coef(y_true_, y_pred_))

    def result(self):
        return self.dice_value

    def reset_state(self):
        self.dice_value.assign(0.0)  # reset metric state

    def dice_coef(self, y_true, y_pred, smooth=1.e-9):
        """
        Calculate dice coefficient.
        Input shape could be either Batch x Height x Width x #Classes (BxHxWxN)
        or Batch x Height x Width (BxHxW).
        Using Mean as reduction type for batch values.
        """
        intersection = K.sum(y_true * y_pred, axis=self.axis)
        union = K.sum(y_true, axis=self.axis) + K.sum(y_pred, axis=self.axis)
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


class ClassDice(tf.keras.metrics.Metric):
    """
    Dice coefficient metric cho một class cụ thể (ví dụ: u lành = 1, u ác = 2).
    """
    def __init__(self, class_id: int, name: str, **kwargs):
        super(ClassDice, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (B, H, W, C)
        # y_pred: (B, H, W, C) - các phân phối xác suất
        
        # Argmax để lấy dự đoán của class có xác suất cao nhất
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        y_pred_one_hot = tf.one_hot(y_pred_idx, tf.shape(y_pred)[-1])
        
        y_true_one_hot = tf.cast(y_true, tf.float32)
        y_pred_one_hot = tf.cast(y_pred_one_hot, tf.float32)

        # Lấy riêng channel của class cần tính Dice
        true_class = y_true_one_hot[..., self.class_id]
        pred_class = y_pred_one_hot[..., self.class_id]

        # Tính tổng giao và tổng hợp
        inter = tf.reduce_sum(true_class * pred_class)
        uni = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class)

        self.intersection.assign_add(inter)
        self.union.assign_add(uni)

    def result(self):
        return (2.0 * self.intersection + 1e-7) / (self.union + 1e-7)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

