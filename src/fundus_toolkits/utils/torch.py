from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType

TensorArray = TypeVar("TensorArray", torch.Tensor, npt.NDArray)


def img_to_torch(x: TensorArray, device: Optional[DeviceLikeType] = None):
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
        t = x

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
