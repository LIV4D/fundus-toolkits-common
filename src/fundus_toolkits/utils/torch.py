from __future__ import annotations

from types import EllipsisType
from typing import NamedTuple, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

# from numba import jit, prange


def torch_interp_bilinear(
    imgs: Tensor,
    y: Tensor,
    x: Tensor,
    batch_idx: Optional[Tensor] = None,
    img_shape: Optional[tuple[int, int]] = None,
    legacy: bool = True,
) -> Tensor:
    """2D bilinear interpolation for a batch of images.

    Parameters
    ----------
    imgs : Tensor
        A batch of images with shape (C, H, W) if batch_idx is None, or (B, C, H, W) otherwise.
    y : Tensor
        A vector of y coordinates with shape (N1, N2, ...).
    x : Tensor
        A vector of x coordinates with shape (N1, N2, ...).
    batch_idx : Optional[Tensor], optional
        A vector of batch indices with shape (N1, N2, ...), by default None. If None, ``imgs`` is expected to have no batch dimension, and all coordinates in ``coord`` are assumed to belong to the same image.

    img_shape: Optional[tuple[int, int]], optional
        The initial shape of the input images (H, W) that the coordinates refer to, by default None. If not None, the input coordinates are rescaled from the original image size to the actual image size in ``imgs``.

    legacy: bool
        If True, use legacy python implementation, otherwise use torch.nn.function.grid_sample.

    Returns
    -------
    Tensor
        A batch of interpolated values with shape (N1, N2, ..., C).

    Example
    -------
    >>> torch.set_printoptions(precision=1, sci_mode=False)
    >>> img = torch.tensor([[[0,0],[1,1]],[[0,1],[0,1]]]).float() # C=2, H=2, W=2
    >>> imgs = torch.stack([img, img+10], dim=0) # B=2, C=2, H=2, W=2
    >>> y = torch.tensor([[0, 0.5, 1.5], [0, 0.5, 1]]) # N=(2,3)
    >>> x = torch.tensor([[0, 0, 1], [0, 0, 1]])
    >>> batch_idx = torch.tensor([[0, 0, 0], [1, 1, 1]])
    >>> torch_interp_bilinear(imgs, y, x, batch_idx, legacy=False)
    tensor([[[ 0.0,  0.0],
             [ 0.5,  0.0],
             [ 1.0,  1.0]],
            [[10.0, 10.0],
             [10.5, 10.0],
             [11.0, 11.0]]])
    """  # noqa: E501
    H, W = imgs.shape[-2:]
    if img_shape is not None and img_shape != imgs.shape[-2:]:
        H_orig, W_orig = img_shape
    elif not y.dtype.is_floating_point and y.dtype.is_floating_point:
        return imgs[batch_idx, :, y, x] if batch_idx is not None else imgs[:, y, x]
    else:
        H_orig, W_orig = H, W

    y = torch.clamp(y, 0, H_orig - 1)
    x = torch.clamp(x, 0, W_orig - 1)

    if legacy:
        y = y * ((H - 1) / (H_orig - 1))
        x = x * ((W - 1) / (W_orig - 1))
        y0 = torch.floor(y).long()
        x0 = torch.floor(x).long()
        y1 = torch.clamp(y0 + 1, max=H - 1)
        x1 = torch.clamp(x0 + 1, max=W - 1)

        dy1 = (y - y0)[..., None]
        dy0 = 1 - dy1
        dx1 = (x - x0)[..., None]
        dx0 = 1 - dx1
        if batch_idx is None:
            img_y0x0 = imgs[:, y0, x0].view(*y0.shape, -1)
            img_y1x0 = imgs[:, y1, x0].view(*y0.shape, -1)
            img_y0x1 = imgs[:, y0, x1].view(*y0.shape, -1)
            img_y1x1 = imgs[:, y1, x1].view(*y0.shape, -1)
        else:
            b = batch_idx.long()
            img_y0x0 = imgs[b, :, y0, x0].view(*y0.shape, -1)
            img_y1x0 = imgs[b, :, y1, x0].view(*y0.shape, -1)
            img_y0x1 = imgs[b, :, y0, x1].view(*y0.shape, -1)
            img_y1x1 = imgs[b, :, y1, x1].view(*y0.shape, -1)

        return img_y0x0 * (dy0 * dx0) + img_y1x0 * (dy1 * dx0) + img_y0x1 * (dy0 * dx1) + img_y1x1 * (dy1 * dx1)
    else:
        Ns = y.shape
        C = imgs.shape[-3]
        if batch_idx is not None:
            B = imgs.shape[0]
            y += B * batch_idx
            H_orig = H * B
            imgs = imgs.permute(1, 0, 2, 3).flatten(1, 2)  # C, B*H, W
        x = x * (2 / (W_orig - 1)) - 1
        y = y * (2 / (H_orig - 1)) - 1
        imgs = imgs.unsqueeze(0)  # 1, C, H, W

        grid = torch.stack((x.flatten(), y.flatten()), dim=-1)[None, None, :, :]  # 1, 1, N, 2
        v = torch.nn.functional.grid_sample(imgs, grid, align_corners=True)  # 1, C, 1, N
        return v.squeeze(0, 2).T.reshape(*Ns, C)


def inverse_displacement(disp: Tensor, src: Tensor, max_iter=50, tol=0.5) -> Tensor:
    """Compute the inverse of a displacement field through fixed-point iteration.

    Parameters
    ----------
    disp : Tensor
        The displacement field with shape (H, W, 2).
    src : Tensor
        The source coordinates with shape (H, W, 2).
    max_iter : int, optional
        The maximum number of iterations, by default 50.
    tol : float, optional
        The tolerance for convergence, by default 0.5.

    Returns
    -------
    Tensor
        The inverse displacement field with shape (H, W, 2).
    """
    try:
        from fundus_vessels_toolkit.utils.cpp_extensions.fvt_cpp import inverse_displacement
    except ImportError:
        raise ImportError("ElasticTransform requires the 'fvt_cpp' extension module to be installed") from None

    return inverse_displacement(disp, src.double(), max_iter, tol)


# @jit("float32[:, :](float32[:, :, :], float32[:, :], int32, float32)", parallel=True)
# def _inverse_displacement(disp, src, max_iter=50, tol=0.5):
#     out = np.zeros_like(src)

#     for i in prange(src.shape[0]):
#         inv_disp_y = 0
#         inv_disp_x = 0
#         for _ in range(max_iter):
#             # Bilinear interpolate
#             disp_interp = _bilinear_interp(disp, src[i, 0] + inv_disp_y, src[i, 1] + inv_disp_x)
#             if (disp_interp[0] + inv_disp_y) ** 2 + (disp_interp[1] + inv_disp_x) ** 2 <= tol**2:
#                 break
#             inv_disp_y = -disp_interp[0]
#             inv_disp_x = -disp_interp[1]
#         out[i, 0] = inv_disp_y
#         out[i, 1] = inv_disp_x
#     return out


# @jit("float32[:](float32[:, :, :], float32, float32)")
# def _bilinear_interp(img, y, x):
#     H, W, C = img.shape
#     y0, x0 = int(np.floor(y)), int(np.floor(x))
#     y0 = np.clip(y0, 0, H - 1)
#     x0 = np.clip(x0, 0, W - 1)
#     y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
#     dy1, dx1 = y - y0, x - x0
#     dy0, dx0 = 1 - dy1, 1 - dx1
#     return (img[y0, x0] * dy0 + img[y1, x0] * dy1) * dx0 + (img[y0, x1] * dy0 + img[y1, x1] * dy1) * dx1


def grid_indices(shape: tuple[int, int], device=None) -> Tensor:
    """Return the grid indices for a given shape.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the grid (H, W).
    device : torch.device, optional
        The device to return the tensors on, by default None.

    Returns
    -------
    Tensor:
        A tensor of shape (H, W, 2) containing the grid indices for each pixel.
    """
    H, W = shape
    y = torch.arange(H, device=device, dtype=torch.int32)
    x = torch.arange(W, device=device, dtype=torch.int32)
    return torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)  # (H, W, 2)


########################################################################################################################
# === TYPING ===
TensorArray = TypeVar("TensorArray", torch.Tensor, npt.NDArray)
DeviceLikeType: TypeAlias = Union[str, torch.device, int]


########################################################################################################################
# === TENSOR SPEC ===
class TensorSpec(NamedTuple):
    name: str
    dim_names: Tuple[str, ...]
    dtype: Optional[torch.dtype] = None
    description: str = ""
    optional: bool = False

    def __str__(self) -> str:
        doc = self.name + f" [{', '.join(self.dim_names)}"
        if self.dtype is not None:
            doc += f" | {self.dtype}"
        doc += "]"

        if self.description:
            doc += f": {self.description}"
        if self.optional:
            doc += " (optional)"

        return doc

    def __repr__(self) -> str:
        return (
            f"TensorSpec(name={self.name}, dim_names={self.dim_names}, dtype={self.dtype}, "
            f"description={self.description})"
        )

    def update(
        self,
        name: str | EllipsisType = ...,
        dim_names: Tuple[str, ...] | EllipsisType = ...,
        dtype: Optional[torch.dtype] | EllipsisType = ...,
        description: str | EllipsisType = ...,
        optional: bool | EllipsisType = ...,
    ) -> TensorSpec:
        """
        Create a new TensorSpec updated with the given keyword arguments.
        """
        return TensorSpec(
            name=name if name is not ... else self.name,
            dim_names=dim_names if dim_names is not ... else self.dim_names,
            dtype=dtype if dtype is not ... else self.dtype,
            description=description if description is not ... else self.description,
            optional=optional if optional is not ... else self.optional,
        )


########################################################################################################################
# === NUMPY/TORCH CASTING ===
def img_to_torch(x: TensorArray, device: Optional[DeviceLikeType] = None, copy=False) -> torch.Tensor:
    """Convert an image to a torch tensor.

    Parameters
    ----------
    x: numpy.ndarray or torch.Tensor
        The input image. Expect dimensions: (H, W, rgb) or (H, W, 1) or (H, W).
    device: torch.DeviceLikeType, optional
        The device to move the tensor to. If None, the tensor will be on the CPU.
    """
    import torch

    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        t = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"Unknown type: {type(x)}.\n Expected numpy.ndarray or torch.Tensor.")
    else:
        t = x if not copy else x.clone()

    match t.shape:
        case s if len(s) == 3:
            if s[2] == 3:
                t = t.permute(2, 0, 1)
            t = t.unsqueeze(0)
        case s if len(s) == 4:
            if s[3] == 3:
                t = t.permute(0, 3, 1, 2)
            assert t.shape[1] == 3, f"Expected 3 channels, got {t.shape[1]}"

    t = t.float()
    if device is not None:
        t = t.to(device)

    return t


def recursive_numpy2torch(x, device=None):
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            x = x.copy()
        try:
            r = torch.from_numpy(x)
        except ValueError:
            r = torch.from_numpy(x.copy())
        if device is not None:
            r = r.to(device)
        return r
    if isinstance(x, dict):
        return {k: recursive_numpy2torch(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_numpy2torch(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple([recursive_numpy2torch(v, device) for v in x])
    return x


def recursive_torch2numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        r = x.cpu().numpy()
        return r
    if isinstance(x, dict):
        return {k: recursive_torch2numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_torch2numpy(v) for v in x]
    if type(x) is tuple:
        return type(x)(recursive_torch2numpy(v) for v in x)
    return x
