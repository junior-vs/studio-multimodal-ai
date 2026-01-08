"""
Image preprocessing utilities

This module contains functions for image preprocessing and enhancement.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (224, 224), normalize: bool = True
) -> np.ndarray:
    """
    Preprocess an image for analysis.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, target_size)

    # Normalize if requested
    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


def enhance_image(
    image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0
) -> np.ndarray:
    """
    Enhance image brightness and contrast.

    Args:
        image: Input image
        brightness: Brightness factor
        contrast: Contrast factor

    Returns:
        Enhanced image
    """
    enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return enhanced
