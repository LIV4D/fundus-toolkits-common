from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt

from .color import parse_color
from .geometric import ABSENT, Point
from .safe_import import import_cv2, is_torch_tensor

if TYPE_CHECKING:
    import torch

    from .color import ColorSpec
    from .safe_import import TensorOrArray
    from .typing import PathLike


########################################################################################################################
#   === READ IMAGE ===
########################################################################################################################
@overload
def read_image(
    path: PathLike,
    *,
    binarize: Literal[True],
    resize: Optional[Tuple[int, int]] = None,
    crop_pad: Optional[Tuple[int, int]] = None,
    cast_to_float: bool = True,
    keep_alpha: bool = False,
) -> npt.NDArray[np.bool_]: ...
@overload
def read_image(
    path: PathLike,
    *,
    binarize: Literal[False] = False,
    resize: Optional[Tuple[int, int]] = None,
    crop_pad: Optional[Tuple[int, int]] = None,
    cast_to_float: Literal[True] = True,
    keep_alpha: bool = False,
) -> npt.NDArray[np.float32]: ...
@overload
def read_image(
    path: PathLike,
    *,
    binarize: Literal[False] = False,
    resize: Optional[Tuple[int, int]] = None,
    crop_pad: Optional[Tuple[int, int]] = None,
    cast_to_float: Literal[False],
    keep_alpha: bool = False,
) -> npt.NDArray[np.uint8]: ...
def read_image(
    path: PathLike,
    *,
    binarize: bool = False,
    resize: Optional[Tuple[int, int]] = None,
    crop_pad: Optional[Tuple[int, int]] = None,
    cast_to_float: bool = True,
    keep_alpha: bool = False,
) -> npt.NDArray[np.float32 | np.bool_ | np.uint8]:
    """Read an image from a file and convert it to a numpy array.
    The image is returned with channels first (C, H, W) if it has multiple channels.

    Parameters
    ----------
    path : PathLike
        The path to the image file.

    binarize : bool, optional
        If True, the image is binarized (converted to a binary mask).

        Default: False.

    cast_to_float : bool, optional
        If True, the image is cast to float and normalized to [0, 1], otherwise it's left as uint8.
        This parameter is ignored if ``binarize`` is True.

        Default: True.

    resize : Tuple[int, int], optional
        If not None, the image is resized to this shape.

        Default: None.

    crop_pad : int | Tuple[int,int], optional
        If not None, the image is cropped or padded with zeros to this shape.
        (If both ``crop_pad`` and ``resize`` are not None, the image is resized first and then cropped/padded.)

        Default: None.

    keep_alpha : bool, optional
        If True, the alpha channel is kept if present in the image.
        If False, the alpha channel is always discarded.

    Returns
    -------
    npt.NDArray[np.float_ | np.bool_ | np.uint8]
        The image as a numpy array.
        If ``binarize`` is True, the image is a binary mask (dtype: bool).
        Otherwise if ``cast_to_float`` is True, the image is a float array (dtype: float).
        Otherwise, the image is a uint8 array (dtype: uint8).
    """
    from .image import resize as resize_image
    from .safe_import import import_cv2, is_cv2_available

    if is_cv2_available():
        cv2 = import_cv2()
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image

        img = Image.open(path)
        img = np.array(img, dtype=np.uint8)

    if img.ndim == 3:
        img = img.transpose((2, 0, 1))  # HWC to CHW
        if img.shape[0] == 4 and not keep_alpha:
            img = img[:, :, :3]

    if img is None:
        raise ValueError(f"Could not load image from {path}")

    if resize is not None:
        img = resize_image(img, resize)

    if binarize:
        img = img.mean(axis=2) > 127 if img.ndim == 3 else img > 127
    elif cast_to_float:
        img = img.astype(np.float32) / np.float32(255)

    if crop_pad is not None:
        img = crop_pad_center(img, crop_pad)

    return img


def read_label_image(
    path: PathLike,
    labels: Mapping[int, ColorSpec] | Sequence[Optional[ColorSpec]],
    *,
    resize: Optional[Tuple[int, int]] = None,
    crop_pad: Optional[Tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    img = read_image(path, binarize=False, resize=resize, crop_pad=crop_pad, cast_to_float=False)
    return rgb_map_to_label(img, labels)


########################################################################################################################
#   === WRITE IMAGE ===
########################################################################################################################
def write_image(
    img: torch.Tensor | npt.NDArray,
    path: PathLike,
    *,
    default_filename: Optional[str] = None,
    on_exists: Literal["raise", "overwrite", "ignore"] = "raise",
) -> None:
    """Write an image to a file.

    Parameters
    ----------
    img : torch.Tensor | npt.NDArray
        The image to write as a [H, W] or [C, H, W] array or tensor.
    path : PathLike
        The path to the output image file.

    default_filename : Optional[str], optional
        If the path is a directory, the image will be saved to this filename in the directory.
        If None and the path is a directory, an error will be raised.

        Default: None.

    on_exists : Literal["overwrite", "ignore", "raise"], optional
        What to do if the output file already exists:
        - "raise": raise an error.
        - "overwrite": overwrite the existing file;
        - "ignore": do nothing;
    """
    from .safe_import import import_cv2, is_cv2_available

    # -- Convert image to uint8 numpy array --
    if is_torch_tensor(img):
        img = img.numpy(force=True)
    img = typing.cast(npt.NDArray, img)

    if img.ndim == 3 and img.shape[2] not in {1, 3, 4} and img.shape[0] in {1, 3, 4}:
        img = np.transpose(img, (2, 0, 1))  # CHW to HWC
    if (np.issubdtype(img.dtype, np.floating) and (img.max() <= 1.0 and img.min() >= 0.0)) or img.dtype == np.bool_:
        img = img * 255
    img = img.astype(np.uint8)

    # -- Ensure path exist --
    path = Path(path)
    if path.is_dir():
        if default_filename is None:
            raise ValueError(f"Output path {path} is a directory, but no default filename was provided.")
        path = path / default_filename

    if path.exists():
        match on_exists:
            case "raise":
                raise FileExistsError(f"Output file {path} already exists.")
            case "ignore":
                return
            case "overwrite":
                pass
            case _:
                raise ValueError(f"Invalid value for on_exists: {on_exists}")

    path.parent.mkdir(parents=True, exist_ok=True)

    # -- Save image --
    if is_cv2_available():
        cv2 = import_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), img)
    else:
        from PIL import Image

        Image.fromarray(img).save(path)


########################################################################################################################
#   === COLOR LABELS MAPS ===
########################################################################################################################
def rgb_map_to_label(
    img: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | Mapping[int, ColorSpec] | Sequence[Optional[ColorSpec]],
) -> npt.NDArray[np.uint8]:
    """Convert a RGB image to a label map.
    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        The RGB image(s) as a [..., H, W] array.

    labels : npt.NDArray[np.uint8] | Mapping[int, ColorSpec] | Sequence[Optional[ColorSpec]]
        The colors corresponding to each label as:
        - a (N, 3) array of RGB colors (dtype: uint8);
        - a mapping from label to color specification (string, RGB tuple or array);
        - a sequence of color specifications (string, RGB tuple or array), where the index in the sequence is the label.

    Returns
    -------
    npt.NDArray[np.uint8]
        The label map as a [..., H, W] array (dtype: uint8).
    """

    if isinstance(labels, Mapping):
        colors = []
        labels_mapping = []
        for label, color in labels.items():
            if color is not None:
                colors.append(parse_color(color))
                labels_mapping.append(label)
        labels_mapping = np.array(labels_mapping, dtype=np.uint8)
        colors = np.array(colors, dtype=np.uint8)
    elif isinstance(labels, Sequence):
        colors = []
        labels_mapping = []
        for i, label in enumerate(labels):
            if label is not None:
                colors.append(parse_color(label))
                labels_mapping.append(i)
        labels_mapping = np.array(labels_mapping, dtype=np.uint8)
        colors = np.array(colors, dtype=np.uint8)
    elif isinstance(labels, np.ndarray):
        if labels.ndim != 2 or labels.shape[1] != 3 or labels.dtype != np.uint8:
            raise ValueError(
                f"Labels array must have shape (N, 3) and dtype uint8, got {labels.shape} and {labels.dtype}"
            )
        colors = labels
        labels_mapping = np.arange(labels.shape[0], dtype=np.uint8)
    else:
        raise ValueError(f"Invalid labels specification: {labels}")
    while colors.ndim < img.ndim:
        colors = colors[None, ...]
    return labels_mapping[np.linalg.norm(img - colors, axis=1).argmin(axis=0)]


def label_map_to_rgb(
    label_map: npt.NDArray[np.integer],
    colors: npt.NDArray[np.uint8] | Mapping[int, ColorSpec] | Sequence[Optional[ColorSpec]],
    background_image: Optional[npt.NDArray[np.uint8]] = None,
):
    """Convert a label map to a RGB image.
    Parameters
    ----------
    label_map : npt.NDArray[np.integer]
        The label map as a [..., H, W] array (dtype: integer).
    colors : npt.NDArray[np.uint8] | Mapping[int, ColorSpec] | Sequence[Optional[ColorSpec]]
        The colors corresponding to each label as:
        - a (N, 3) array of RGB colors (dtype: uint8);
        - a mapping from label to color specification (string, RGB tuple or array);
        - a sequence of color specifications (string, RGB tuple or array), where the index in the sequence is the label.
    background_image : Optional[npt.NDArray[np.uint8]], optional
        If provided, the label colors will be drawn on top of this image.
        Its shape must be (3, H, W) or (..., 3, H, W) and its dtype uint8.
        If None, the background will be black.
        Default: None.

    Returns
    -------
    npt.NDArray[np.uint8]
        The RGB image as a [..., 3, H, W] array (dtype: uint8).
    """
    output_shape = (*label_map.shape[:-2], 3, *label_map.shape[-2:])
    if background_image is None:
        img = np.zeros(output_shape, dtype=np.uint8)
    else:
        assert background_image.dtype == np.uint8, (
            f"Background image must have dtype uint8, got {background_image.dtype}"
        )
        if not background_image.shape == output_shape:
            assert background_image.shape == (3, *label_map.shape[-2:]), (
                f"Background image must have shape (3, H, W) or {output_shape}, got {background_image.shape}"
            )
            for s in reversed(output_shape[:-3]):
                background_image = np.repeat(background_image[None], s, axis=0)
        img = background_image.copy()

    colors_dict: dict[int, npt.NDArray[np.uint8]] = {}
    if isinstance(colors, np.ndarray):
        if colors.ndim != 2 or colors.shape[1] != 3 or colors.dtype != np.uint8:
            raise ValueError(
                f"Colors array must have shape (N, 3) and dtype uint8, got {colors.shape} and {colors.dtype}"
            )
        for i in range(colors.shape[0]):
            colors_dict[i] = colors[i]
    elif isinstance(colors, Sequence):
        for i, color in enumerate(colors):
            if color is not None:
                colors_dict[i] = parse_color(color)
    elif isinstance(colors, Mapping):
        for label, color in colors.items():
            colors_dict[label] = parse_color(color)
    else:
        raise ValueError(f"Invalid colors type: {type(colors)}")

    for label, color in colors_dict.items():
        img[label_map == label] = color

    return img


def smooth_labels(image: npt.NDArray[np.uint8], std: float) -> npt.NDArray[np.uint8]:
    """Smooth the labels of an image using a Gaussian filter.

    Parameters
    ----------
    image : npt.NDArray[np.uint8]
        The image to smooth.
    std : float
        The standard deviation of the Gaussian filter.

    Returns
    -------
    npt.NDArray[np.uint8]
        The smoothed image.
    """
    from .safe_import import import_cv2

    cv2 = import_cv2()
    labels = np.unique(image)
    if len(labels) == 1:
        return image

    ksize = int(2 * std + 1)
    if ksize % 2 == 0:
        ksize += 1

    # -- Smooth labels --
    if labels[0] == 0:
        labels = labels[1:]  # Ignore background
    labels_lookup = np.zeros(256, dtype=np.uint8)
    for i, label in enumerate(labels):
        labels_lookup[i + 1] = label

    image_one_hots = {label: (image == label).astype(np.uint8) * 255 for label in labels}

    for label, one_hot in image_one_hots.items():
        image_one_hots[label] = cv2.GaussianBlur(one_hot, (ksize, ksize), std)
    all_one_hots = np.stack(list(image_one_hots.values()), axis=0)
    smoothed_image = np.argmax(all_one_hots, axis=0).astype(np.uint8) + 1
    smoothed_image = labels_lookup[smoothed_image]

    # -- Smooth background --
    mask = (image != 0).astype(np.uint8) * 255
    smoothed_mask = cv2.GaussianBlur(mask, (ksize, ksize), std)
    smoothed_image[smoothed_mask < 100] = 0

    return smoothed_image


def crop_pad_center(img: TensorOrArray, shape: Tuple[int, ...]) -> TensorOrArray:
    """Crop or pad an image to a given shape, centering the original image.

    Parameters
    ----------
    img : TensorOrArray
        The image to crop or pad as a [H, W] or [C, H, W] array or tensor.
    shape : Tuple[int, ...]
        The target# shape to which the image should be cropped or padded as (height, width).

    Returns
    -------
    TensorOrArray
        The cropped or padded image.

    Raises
    ------
    ValueError
        If the image type is not supported.
    """
    H, W = img.shape[-2:]
    h, w = shape[-2:]

    if isinstance(img, np.ndarray):
        out = np.zeros(shape=img.shape[:-2] + shape, dtype=img.dtype)
    elif isinstance(img, torch.Tensor):
        out = torch.zeros(img.shape[:-2] + shape, dtype=img.dtype, device=img.device)
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    if H < h:
        x0 = (h - H) // 2
        x1 = 0
        h0 = H
    else:
        x0 = 0
        x1 = (H - h) // 2
        h0 = h

    if W < w:
        y0 = (w - W) // 2
        y1 = 0
        w0 = W
    else:
        y0 = 0
        y1 = (W - w) // 2
        w0 = w

    out[..., x0 : x0 + h0, y0 : y0 + w0] = img[..., x1 : x1 + h0, y1 : y1 + w0]  # type: ignore
    return out


def crop_pad(
    img: TensorOrArray, shape, center=(0.5, 0.5), pad_mode="constant", pad_value=0, broadcastable=False
) -> TensorOrArray:
    H, W = img.shape[-2:]
    h, w = shape[-2:]
    y, x = (
        int(round((c % 1) * s)) if isinstance(c, float) and 0 <= c <= 1 else c
        for c, s in zip(center, (H, W), strict=True)
    )

    if H == 1 and broadcastable:
        y1 = 0
        y2 = 1
        pad_y1 = 0
        pad_y2 = 0
    else:
        pad_y1, y1 = 0, y - h // 2
        if y1 < 0:
            pad_y1, y1 = -y1, 0
        y2 = min(y1 + h, H)
        pad_y2 = h - (y2 - (y1 - pad_y1))

    if W == 1 and broadcastable:
        x1 = 0
        x2 = 1
        pad_x1 = 0
        pad_x2 = 0
    else:
        pad_x1, x1 = 0, x - w // 2
        if x1 < 0:
            pad_x1, x1 = -x1, 0
        x2 = min(x1 + w, W)
        pad_x2 = w - (x2 - (x1 - pad_x1))

    img = img[..., y1:y2, x1:x2]
    if pad_x1 or pad_x2 or pad_y1 or pad_y2:
        if is_torch_tensor(img):
            import torch.nn.functional as F

            img = F.pad(img, (pad_x1, pad_x2, pad_y1, pad_y2), mode=pad_mode, value=pad_value)  # type: ignore
        elif isinstance(img, np.ndarray):
            img = np.pad(
                img,
                ((0, 0),) * (img.ndim - 2) + ((pad_y1, pad_y2), (pad_x1, pad_x2)),
                mode=pad_mode,  # type: ignore
                constant_values=pad_value,
            )  # type: ignore
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    return img  # type: ignore


T = TypeVar("T", bound=npt.NDArray[np.float32 | np.uint8 | np.bool_])


def rotate(image: T, angle: float, interpolation: Optional[bool] = None) -> T:
    """Rotate an image by a given angle.

    Parameters
    ----------
    img : npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The image to rotate as a [H, W] or [C, H, W] array.
    angle : float
        The angle by which to rotate the image in degrees.
    interpolation : Optional[bool], optional
        Wether to use interpolation or not:
        - if False, the image is rotated using the nearest neighbor interpolation;
        - if True, the image is rotated using the linear interpolation;
        - if None, the interpolation is automatically selected based on the image type.
        The default is None.

    Returns
    -------
        npt.NDArray: Rotated image.
    """
    from .safe_import import import_cv2

    cv2 = import_cv2()

    if interpolation is None:
        interpolation = not (image.dtype == np.uint8 and image.max() < 10)
    interpol_mode = cv2.INTER_LINEAR if interpolation else cv2.INTER_NEAREST

    if np.issubdtype(image.dtype, np.floating):
        cv2_image = (image * 255).astype(np.uint8)
        image_dtype = "float"
    elif image.dtype == np.bool_:
        cv2_image = (image * 255).astype(np.uint8)
        image_dtype = "bool"
    elif image.dtype == np.uint8:
        cv2_image = image
        image_dtype = "uint8"
    else:
        raise ValueError(f"Unsupported image type: {image.dtype}")
    if cv2_image.ndim == 3:
        cv2_image = cv2_image.transpose((1, 2, 0))  # CHW to HWC

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    cv2_image = cv2.warpAffine(cv2_image, M, (image.shape[1], image.shape[0]), flags=interpol_mode)

    if cv2_image.ndim == 3:
        cv2_image = cv2_image.transpose((2, 0, 1))  # HWC to CHW

    match image_dtype:
        case "float":
            return cv2_image.astype(np.float32) / np.float32(255)
        case "bool":
            return cv2_image > 127
    return cv2_image


def resize(image: T, size: Tuple[int, int], interpolation: Optional[bool] = None) -> T:
    """Resize an image to a given size.

    Parameters
    ----------
    image : npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The image to resize as a [H, W] or [C, H, W] array.
    size : Tuple[int, int]
        The size to which the image should be resized as (height, width).
    interpolation : Optional[bool], optional
        Wether to use interpolation or not:
        - if False, the image is resized using the nearest neighbor interpolation;
        - if True, the image is resized using the linear interpolation for upscaling and the area interpolation for down-sampling;
        - if None, the interpolation is automatically selected based on the image type.
        The default is None.

    Returns
    -------
    npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The resized image.
    """  # noqa: E501
    from .safe_import import import_cv2

    cv2 = import_cv2()

    if np.issubdtype(image.dtype, np.floating):
        cv2_image = (image * 255).astype(np.uint8)
        image_dtype = "float"
    elif image.dtype == np.bool_:
        cv2_image = (image * 255).astype(np.uint8)
        image_dtype = "bool"
    elif image.dtype == np.uint8:
        cv2_image = image
        image_dtype = "uint8"
    else:
        raise ValueError(f"Unsupported image type: {image.dtype}")
    if cv2_image.ndim == 3:
        cv2_image = cv2_image.transpose((1, 2, 0))  # CHW to HWC

    if interpolation is None:
        # If the image is a label map (less than 10 unique values), disable interpolation
        interpolation = not (image_dtype == "uint8" and image.max() < 10)
    interpol_mode = cv2.INTER_NEAREST
    if interpolation:
        interpol_mode = cv2.INTER_LINEAR if size[0] > image.shape[0] or size[1] > image.shape[1] else cv2.INTER_AREA

    cv2_image = cv2.resize(cv2_image, (size[1], size[0]), interpolation=interpol_mode)

    if cv2_image.ndim == 3:
        cv2_image = cv2_image.transpose((2, 0, 1))  # HWC to CHW

    match image_dtype:
        case "float":
            return cv2_image.astype(np.float32) / np.float32(255)
        case "bool":
            return cv2_image > 127
    return cv2_image


@overload
def find_centroid(
    seg: npt.NDArray[np.bool_], fit_ellipse: Literal[False] = False
) -> Tuple[npt.NDArray[np.bool_], Point]: ...
@overload
def find_centroid(
    seg: npt.NDArray[np.bool_], fit_ellipse: Literal[True]
) -> Tuple[npt.NDArray[np.bool_], Point, Point]: ...
def find_centroid(
    seg: npt.NDArray[np.bool_], fit_ellipse: bool = False
) -> Tuple[npt.NDArray[np.bool_], Point] | Tuple[npt.NDArray[np.bool_], Point, Point]:
    """Find the centroid of a connected component in a binary image.

    Parameters
    ----------
    seg : npt.NDArray[np.uint8]
        The binary image.
    fit_ellipse : bool, optional
        Whether to fit an ellipse to the connected component or not.
        If True, the function returns the fitted ellipse parameters.
        If False, the function returns the centroid of the connected component.

    Returns
    -------
    Tuple[npt.NDArray[np.uint8], Point, Point | int]
        The binary image with the connected component labeled as 1,
        the centroid of the connected component,
        and the fitted ellipse parameters or ABSENT if fit_ellipse is False.
    """
    cv2 = import_cv2()
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(seg.astype(np.uint8), 4, cv2.CV_32S)  # type: ignore
    if stats.shape[0] == 1:
        return (seg, ABSENT) if not fit_ellipse else (seg, ABSENT, ABSENT)
    largest_cc = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 if stats.shape[0] > 2 else 1
    if not fit_ellipse:
        return labels == largest_cc, Point(*centroids[1][::-1])
    contours, _ = cv2.findContours((labels == largest_cc).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    return labels == largest_cc, Point(*ellipse[0][::-1]), Point(*ellipse[1][::-1])
