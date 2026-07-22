"""
General Utility functions
"""
import os
import tensorflow as tf
from omegaconf import DictConfig
from .images_utils import image_to_mask_name


def create_directory(path):
    """
    Create Directory if it already does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    """
    Concatenate multiple paths.
    """
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))


def set_gpus(gpu_ids):
    """
    Change number of visible gpus for tensorflow.
    gpu_ids: Could be integer or list of integers.
    In case Integer: if integer value is -1 then use all available gpus.
    otherwise if positive number, then use given number of gpus.
    In case list of Integer: each integer will be considered as gpu id
    """
    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    all_gpus_length = len(all_gpus)
    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            gpu_ids = range(all_gpus_length)
        else:
            gpu_ids = min(gpu_ids, all_gpus_length)
            gpu_ids = range(gpu_ids)

    selected_gpus = [all_gpus[gpu_id] for gpu_id in gpu_ids if gpu_id < all_gpus_length]

    try:
        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


def get_gpus_count():
    """
    Return length of available gpus.
    """
    gpus = len(tf.config.experimental.list_logical_devices('GPU'))
    if gpus == 0:
        gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    return max(1, gpus)


def get_data_paths(cfg: DictConfig, mode: str, mask_available: bool):
    """
    Return list of absolute images/mask paths.
    Supports both absolute paths (e.g. D:/data/...) and
    relative paths (joined with WORK_DIR).
    """

    def resolve_path(cfg_path):
        """Nếu path tồn tại thì dùng trực tiếp, tự động sửa lỗi chính tả và đổi ổ đĩa."""
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

    # read images from directory
    if isinstance(cfg.DATASET[mode].IMAGES_PATH, str):
        images_dir = resolve_path(cfg.DATASET[mode].IMAGES_PATH)
        images_paths = os.listdir(images_dir)

        if mask_available:
            mask_dir = resolve_path(cfg.DATASET[mode].MASK_PATH)
            mask_paths = [
                image_to_mask_name(image_name) for image_name in images_paths
            ]
            # create full mask paths from folder
            mask_paths = [
                os.path.join(mask_dir, mask_name)
                for mask_name in mask_paths
            ]

        # create full images paths from folder
        images_paths = [
            os.path.join(images_dir, image_name)
            for image_name in images_paths
        ]
    else:
        # read images and mask from absolute paths given in list
        images_paths = list(cfg.DATASET[mode].IMAGES_PATH)
        if mask_available:
            mask_paths = list(cfg.DATASET[mode].MASK_PATH)

    if mask_available:
        return images_paths, mask_paths
    else:
        return images_paths,


def suppress_warnings():
    """
    Suppress TensorFlow warnings.
    """
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('dali').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.autograph.set_verbosity(3)
