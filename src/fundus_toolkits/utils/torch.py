from __future__ import annotations

from types import EllipsisType
from typing import NamedTuple, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch

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
