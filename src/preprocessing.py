import cv2
import numpy as np
from PIL import Image


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
        img: Input image as a numpy array (RGB, uint8).

    Returns:
        Image with CLAHE applied (uint8).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img


def preprocess_image(img: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Standardizes image preprocessing: resize, CLAHE, and normalization.

    Args:
        img: PIL Image object.
        target_size: Desired output size.

    Returns:
        Preprocessed image as a numpy array, normalized to [0, 1].
    """
    img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)

    img_array = apply_clahe(img_array)
    img_array = img_array.astype(np.float32)

    return img_array