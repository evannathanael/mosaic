"""The robustness transform pipeline.

Implements the 6 transforms required by the problem statement, each with the
exact parameter grid specified:

    JPEG Compression   quality = 90, 70, 50, 30
    Gaussian Blur      kernel sigma = 0.5, 1.0, 2.0
    Resize             scale 0.5x / 0.25x then upscale back
    Gaussian Noise     sigma = 0.02, 0.05, 0.10
    Color Jitter       brightness/contrast/saturation +/- 20%
    Center Crop        crop 80%

Two entry points:
  - `random_train_transform(config)`  -> torchvision/albumentations pipeline
     that applies a RANDOM transform + severity at train time (simulates the
     real-world redistribution pipeline), plus optional train-only extras
     (rotation / extra jitter) to reduce shortcut learning.
  - `named_eval_transform(name, config)` -> a FIXED, named transform for
     building the robustness evaluation table (e.g. "jpeg_q30", "blur_sigma2").
"""
import io
import random

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Individual transform functions — operate on a numpy uint8 HWC RGB image.
# ---------------------------------------------------------------------------

def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = max(3, int(2 * round(3 * sigma) + 1))  # odd kernel size
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def apply_resize_roundtrip(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma * 255, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_color_jitter(img: np.ndarray, brightness=0.2, contrast=0.2, saturation=0.2) -> np.ndarray:
    pil_img = Image.fromarray(img)
    from torchvision.transforms import ColorJitter

    jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    return np.array(jitter(pil_img))


def apply_center_crop(img: np.ndarray, crop_fraction: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = img[top:top + ch, left:left + cw]
    # resize back so downstream model input size stays consistent
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


TRANSFORM_FNS = {
    "jpeg_compression": apply_jpeg_compression,
    "gaussian_blur": apply_gaussian_blur,
    "resize": apply_resize_roundtrip,
    "gaussian_noise": apply_gaussian_noise,
    "color_jitter": apply_color_jitter,
    "center_crop": apply_center_crop,
}


# ---------------------------------------------------------------------------
# Random training-time transform (robustness augmentation)
# ---------------------------------------------------------------------------

def random_train_transform(img: np.ndarray, config: dict) -> np.ndarray:
    """Apply ONE randomly chosen required transform at a random severity level,
    simulating real-world redistribution during training. Also applies
    train-only extras (rotation / extra jitter) if enabled in config, per the
    workshop's shortcut-prevention guidance — these are NOT part of official
    evaluation.
    """
    t_cfg = config["transforms"]
    choice = random.choice(list(TRANSFORM_FNS.keys()))

    if choice == "jpeg_compression":
        q = random.choice(t_cfg["jpeg_compression"]["qualities"])
        img = apply_jpeg_compression(img, q)
    elif choice == "gaussian_blur":
        s = random.choice(t_cfg["gaussian_blur"]["sigmas"])
        img = apply_gaussian_blur(img, s)
    elif choice == "resize":
        s = random.choice(t_cfg["resize"]["scales"])
        img = apply_resize_roundtrip(img, s)
    elif choice == "gaussian_noise":
        s = random.choice(t_cfg["gaussian_noise"]["sigmas"])
        img = apply_gaussian_noise(img, s)
    elif choice == "color_jitter":
        cj = t_cfg["color_jitter"]
        img = apply_color_jitter(img, cj["brightness"], cj["contrast"], cj["saturation"])
    elif choice == "center_crop":
        img = apply_center_crop(img, t_cfg["center_crop"]["crop_fraction"])

    extras = config.get("train_only_augmentation", {})
    if extras.get("random_rotation_degrees", 0) > 0:
        angle = random.uniform(-extras["random_rotation_degrees"], extras["random_rotation_degrees"])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if extras.get("extra_color_jitter", False):
        img = apply_color_jitter(img, 0.1, 0.1, 0.1)

    return img


# ---------------------------------------------------------------------------
# Fixed, named eval transforms — used to build the robustness table
# ---------------------------------------------------------------------------

NAMED_EVAL_TRANSFORMS = {
    "clean": lambda img: img,
    "jpeg_q90": lambda img: apply_jpeg_compression(img, 90),
    "jpeg_q70": lambda img: apply_jpeg_compression(img, 70),
    "jpeg_q50": lambda img: apply_jpeg_compression(img, 50),
    "jpeg_q30": lambda img: apply_jpeg_compression(img, 30),
    "blur_sigma0.5": lambda img: apply_gaussian_blur(img, 0.5),
    "blur_sigma1": lambda img: apply_gaussian_blur(img, 1.0),
    "blur_sigma2": lambda img: apply_gaussian_blur(img, 2.0),
    "resize_0.5x": lambda img: apply_resize_roundtrip(img, 0.5),
    "resize_0.25x": lambda img: apply_resize_roundtrip(img, 0.25),
    "noise_0.02": lambda img: apply_gaussian_noise(img, 0.02),
    "noise_0.05": lambda img: apply_gaussian_noise(img, 0.05),
    "noise_0.10": lambda img: apply_gaussian_noise(img, 0.10),
    "color_jitter": lambda img: apply_color_jitter(img),
    "crop_80": lambda img: apply_center_crop(img, 0.8),
}


def named_eval_transform(name: str, img: np.ndarray) -> np.ndarray:
    if name not in NAMED_EVAL_TRANSFORMS:
        raise ValueError(f"Unknown eval transform '{name}'. Options: {list(NAMED_EVAL_TRANSFORMS.keys())}")
    return NAMED_EVAL_TRANSFORMS[name](img)
