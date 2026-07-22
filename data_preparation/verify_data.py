"""
Verify for each image corresponding mask exist or not.
Check against both train and val data
"""
import os
import sys
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.append(os.path.abspath("./"))
from utils.general_utils import join_paths
from utils.images_utils import image_to_mask_name


def check_image_and_mask(cfg, mode):
    """
    Check and print names of those images whose mask are not found.
    """
    img_cfg = cfg.DATASET[mode].IMAGES_PATH
    mask_cfg = cfg.DATASET[mode].MASK_PATH

    def resolve_path(cfg_path):
        if os.path.exists(cfg_path):
            return cfg_path
        # Tự động sửa lỗi chính tả Datatset_ -> Dataset_
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
        rel_path = join_paths(cfg.WORK_DIR, cfg_path)
        if os.path.exists(rel_path):
            return rel_path
        return fixed_typo

    images_path = resolve_path(img_cfg)
    mask_path = resolve_path(mask_cfg)

    all_images = os.listdir(images_path)

    both_found = True
    for image in tqdm(all_images):
        mask_name = image_to_mask_name(image)
        if not (
                os.path.exists(
                    join_paths(images_path, image)
                ) and
                os.path.exists(
                    join_paths(mask_path, mask_name)
                )
        ):
            print(f"{mask_name} did not found against {image}")
            both_found = False

    return both_found


def verify_data(cfg: DictConfig):
    """
    For both train and val data, check for each image its
    corresponding mask exist or not. If not then stop the program.
    """
    assert check_image_and_mask(cfg, "TRAIN"), \
        "Train images and mask should be same in length"

    assert check_image_and_mask(cfg, "VAL"), \
        "Validation images and mask should be same in length"
