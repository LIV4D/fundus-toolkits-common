from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


def fundus_ROI(
    fundus: np.ndarray,
    blur_radius=5,
    red_threshold: int = 30,
    morphological_clean=False,
    smoothing_radius=0,
    final_erosion=5,
    check=True,
) -> npt.NDArray[np.bool_]:
    """Compute the region of interest (ROI) of a fundus image by thresholding its red channel.

    Parameters:
    -----------
    fundus:
        The fundus image. Expect dimensions: (rgb, H, W).

    blur_radius:
        The radius of the median blur filter applied on the red channel.

        By default: 5.

    red_threshold:
        The threshold value for the red channel.

        By default: 30.

    morphological_clean:
        Whether to perform morphological cleaning. (Small objects removal and filling of the holes not adjacent to the image borders.)

        By default: False.

    smoothing_radius:
        The radius of the Gaussian smoothing of the threshold mask.

        By default: 0.

    final_erosion:
        The radius of the disk used for a final erosion of the mask.

        By default: 4.

    Returns:
        The ROI mask.

    """  # noqa: E501
    from .safe_import import import_cv2

    cv2 = import_cv2()

    padding = blur_radius + smoothing_radius
    fundus_red = np.pad(fundus[0], ((padding, padding), (padding, padding)), mode="constant")
    if fundus_red.dtype != np.uint8:
        if fundus_red.min() >= 0 and fundus_red.max() <= 1:
            fundus_red *= 255
        fundus_red = fundus_red.astype(np.uint8)
    fundus_red = cv2.medianBlur(fundus_red, blur_radius * 2 - 1)
    _, mask = cv2.threshold(fundus_red, red_threshold, 255, type=cv2.THRESH_BINARY_INV)

    if morphological_clean:
        _, labels_map, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Remove small objects
        small_labels = stats[:, cv2.CC_STAT_AREA] > 5000
        small_labels[0] = True
        inv_mask = np.where(small_labels[labels_map], np.uint8(0), np.uint8(255))  # type: ignore[assignment]

        # Select dark components adjacent to the image borders
        _, labels_map, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
        border_labels = np.unique([labels_map[0, :], labels_map[-1, :], labels_map[:, 0], labels_map[:, -1]])
        if border_labels[0] == 0:
            border_labels = border_labels[1:]

        # The final mask is black for those components, and white otherwise
        mask = np.where(np.isin(labels_map, border_labels), np.uint8(0), np.uint8(255))

    # Take the largest connected component
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if stats.shape[0] > 1:
        mask = labels == (np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        mask = np.where(mask, np.uint8(0), np.uint8(255))

    if smoothing_radius > 0:
        smooth_mask = cv2.GaussianBlur(
            mask.astype(np.uint8) * 255,
            (smoothing_radius * 6 + 1, smoothing_radius * 6 + 1),
            smoothing_radius,
            borderType=cv2.BORDER_CONSTANT,
        )
        _, mask = cv2.threshold(smooth_mask, 125, 255, type=cv2.THRESH_BINARY_INV)

    if final_erosion > 0:
        erosion_ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_erosion * 2 + 1, final_erosion * 2 + 1))
        mask = cv2.erode(mask, erosion_ele)

    # Convert to boolean mask
    mask = mask > 125

    if check:
        if mask.sum() < mask.size * 0.6:
            warnings.warn(
                "The computed ROI mask is smaller than 60% of the image size and might be invalid.",
                RuntimeWarning,
                stacklevel=2,
            )

    return mask[padding:-padding, padding:-padding]


def gaussian_preprocess_torch(img: torch.Tensor) -> torch.Tensor:
    """Preprocess the fundus image using Gaussian filtering.
    Parameters:
    -----------
    img: np.ndarray
        The input image with shape (3, H, W) or (B, 3, H, W).

    Returns:
    --------
    np.ndarray
        The preprocessed image with shape (3, H, W) or (B, 3, H, W).
    """
    import torch
    from torch.nn.functional import conv2d

    sigma = np.max(img.shape) / 60
    radius = int(6.5 * sigma + 0.5)
    xx, yy = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device),
        torch.arange(-radius, radius + 1, dtype=torch.float32, device=img.device),
    )

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel[None, None, :, :].repeat(img.shape[1], 1, 1, 1)

    batched = img.ndim == 4

    if not batched:
        img = img.unsqueeze(0)  # Add batch dimension if not present
    blur = conv2d(img, kernel, padding=radius, groups=img.shape[1])
    if not batched:
        img = img.squeeze(0)
        blur = blur.squeeze(0)

    return (img - blur - 0.0022501) / 0.02771
