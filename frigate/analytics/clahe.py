"""Contrast-Limited Adaptive Histogram Equalization (CLAHE) for low-light CCTV/LPR feeds."""

import cv2
import numpy as np


def enhance_license_plate_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhance localized contrast of low-light or glare-degraded license plate crops.

    Args:
        image: Numpy array of shape (H, W, 3) BGR/RGB or (H, W) Grayscale.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced image array with identical shape and dtype uint8.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:
        # Grayscale image
        return clahe.apply(image)

    # 3-channel color image: convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L-channel only to avoid color shifting
    cl = clahe.apply(l_channel)

    # Merge channels back
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
