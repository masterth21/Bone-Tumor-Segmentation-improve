"""
Returns Unet3+, U-Net++, or Hybrid CNN-Transformer models
"""
import tensorflow as tf
from omegaconf import DictConfig

# Import các module gốc
from .backbones import vgg16_backbone, vgg19_backbone, unet3plus_backbone
from .unet3plus import unet3plus
from .unet3plus_deep_supervision import unet3plus_deepsup
from .unet3plus_deep_supervision_cgm import unet3plus_deepsup_cgm
from .nalaformer_attention import NaLaFormerBottleneck

# segmentation_models chỉ dùng cho unet_plus_plus, lazy import để tránh lỗi Keras 3
sm = None

def build_hybrid_transunet(cfg):
    """TRUE TRANSUNET: ResNet50V2 + Transformer + Skip Connections"""
    img_size = cfg.INPUT.HEIGHT
    channels = cfg.INPUT.CHANNELS
    num_classes = cfg.OUTPUT.CLASSES

    # --- PHẦN 1: ENCODER (ResNet50V2) ---
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=(img_size, img_size, channels)
    )
    
    # 🎯 ĐÃ SỬA LỖI Ở ĐÂY: Rút đúng các trạm trước khi bị thu nhỏ ảnh!
    s1 = base_model.get_layer("conv1_conv").output       # Kích thước: 192x192
    s2 = base_model.get_layer("conv2_block2_out").output # Kích thước: 96x96
    s3 = base_model.get_layer("conv3_block3_out").output # Kích thước: 48x48
    s4 = base_model.get_layer("conv4_block5_out").output # Kích thước: 24x24
    cnn_out = base_model.output                          # Kích thước: 12x12

    # --- PHẦN 2: TRANSFORMER BOTTLENECK ---
    shape = tf.keras.backend.int_shape(cnn_out)
    h, w, c = shape[1], shape[2], shape[3]
    
    x = tf.keras.layers.Reshape((h * w, c))(cnn_out)
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=256)(x, x)
    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization()(x)
    ffn = tf.keras.layers.Dense(c, activation='relu')(x)
    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Reshape((h, w, c))(x) # Về lại 12x12
    
    # --- PHẦN 3: DECODER (Kết nối với Skip Connections) ---
    # Tầng 1: 12x12 -> 24x24
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s4]) # Ghép cầu nối s4 (24x24) -> KHỚP 100%
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 2: 24x24 -> 48x48
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s3]) # Ghép cầu nối s3 (48x48) -> KHỚP 100%
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 3: 48x48 -> 96x96
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s2]) # Ghép cầu nối s2 (96x96) -> KHỚP 100%
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 4: 96x96 -> 192x192
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s1]) # Ghép cầu nối s1 (192x192) -> KHỚP 100%
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 5: 192x192 -> 384x384
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Đầu ra duy nhất
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output_layer')(x)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name="True_TransUNet_With_Skips")


def build_nalaformer_transunet(cfg):
    """NALAFORMER TRANSUNET: ResNet50V2 + NaLaFormer Attention + Skip Connections
    
    Thay thế Softmax Multi-Head Attention bằng Q Norm-Aware Linear Attention.
    Complexity: O(N·d²) thay vì O(N²·d) → nhanh hơn với spatial resolution lớn.
    """
    img_size = cfg.INPUT.HEIGHT
    channels = cfg.INPUT.CHANNELS
    num_classes = cfg.OUTPUT.CLASSES

    # --- Đọc hyperparams từ config (nếu có), hoặc dùng giá trị mặc định ---
    nala_cfg = getattr(cfg.MODEL, 'NALAFORMER', None)
    d_model = nala_cfg.D_MODEL if nala_cfg and hasattr(nala_cfg, 'D_MODEL') else 256
    depth = nala_cfg.DEPTH if nala_cfg and hasattr(nala_cfg, 'DEPTH') else 4
    num_heads = nala_cfg.NUM_HEADS if nala_cfg and hasattr(nala_cfg, 'NUM_HEADS') else 8
    ff_expansion = nala_cfg.FF_EXPANSION if nala_cfg and hasattr(nala_cfg, 'FF_EXPANSION') else 4
    dropout_rate = nala_cfg.DROPOUT_RATE if nala_cfg and hasattr(nala_cfg, 'DROPOUT_RATE') else 0.1

    # --- PHẦN 1: ENCODER (ResNet50V2) ---
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_shape=(img_size, img_size, channels)
    )
    
    s1 = base_model.get_layer("conv1_conv").output       # 192x192
    s2 = base_model.get_layer("conv2_block2_out").output # 96x96
    s3 = base_model.get_layer("conv3_block3_out").output # 48x48
    s4 = base_model.get_layer("conv4_block5_out").output # 24x24
    cnn_out = base_model.output                          # 12x12

    # --- PHẦN 2: NALAFORMER BOTTLENECK (thay thế Softmax MHA) ---
    nala_bottleneck = NaLaFormerBottleneck(
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        ff_expansion=ff_expansion,
        dropout_rate=dropout_rate,
        name="nalaformer_bottleneck",
    )
    x = nala_bottleneck(cnn_out)  # (B, 12, 12, 2048) → (B, 12, 12, 2048)

    # --- PHẦN 3: DECODER (Kết nối với Skip Connections) ---
    # Tầng 1: 12x12 -> 24x24
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s4])
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 2: 24x24 -> 48x48
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s3])
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 3: 48x48 -> 96x96
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s2])
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 4: 96x96 -> 192x192
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, s1])
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Tầng 5: 192x192 -> 384x384
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Đầu ra
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output_layer')(x)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name="NaLaFormer_TransUNet")


def prepare_model(cfg: DictConfig, training=False):
    """TRẠM ĐIỀU PHỐI"""
    input_shape = [cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS]

    if cfg.MODEL.TYPE == "nalaformer_transunet":
        print(">>> Đang sử dụng kiến trúc: NaLaFormer TransUNet (ResNet50V2 + Q Norm-Aware Linear Attention)")
        return build_nalaformer_transunet(cfg)

    elif cfg.MODEL.TYPE == "hybrid_transunet":
        print(">>> Đang sử dụng kiến trúc: TRUE TransUNet (ResNet50V2 + Transformer + Skip Connections)")
        return build_hybrid_transunet(cfg)

    elif cfg.MODEL.TYPE == "unet_plus_plus":
        print(f">>> Đang sử dụng kiến trúc: Unet (Standard) với Backbone {cfg.MODEL.BACKBONE.TYPE}")
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        # Đã gỡ bỏ decoder_attention_type để tránh lỗi thư viện
        return sm.Unet( 
            backbone_name=cfg.MODEL.BACKBONE.TYPE, 
            encoder_weights='imagenet' if training else None,
            classes=cfg.OUTPUT.CLASSES,
            activation='softmax',
            input_shape=tuple(input_shape)
        )

    print(f">>> Đang sử dụng kiến trúc: UNet 3+ (Basic) với Backbone {cfg.MODEL.BACKBONE.TYPE}")
    input_layer = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    filters = [64, 128, 256, 512, 1024]

    if cfg.MODEL.BACKBONE.TYPE == "unet3plus":
        backbone_layers = unet3plus_backbone(input_layer, filters)
    elif cfg.MODEL.BACKBONE.TYPE == "vgg16":
        backbone_layers = vgg16_backbone(input_layer)
    elif cfg.MODEL.BACKBONE.TYPE == "vgg19":
        backbone_layers = vgg19_backbone(input_layer)
    else:
        raise ValueError("Wrong backbone type passed for UNet 3+.")

    if cfg.MODEL.TYPE == "unet3plus":
        outputs, model_name = unet3plus(backbone_layers, cfg.OUTPUT.CLASSES, filters)
    elif cfg.MODEL.TYPE == "unet3plus_deepsup":
        outputs, model_name = unet3plus_deepsup(backbone_layers, cfg.OUTPUT.CLASSES, filters, training)
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        outputs, model_name = unet3plus_deepsup_cgm(backbone_layers, cfg.OUTPUT.CLASSES, filters, training)
    else:
        raise ValueError("Wrong model type passed.")

    return tf.keras.Model(inputs=input_layer, outputs=outputs, name=model_name)