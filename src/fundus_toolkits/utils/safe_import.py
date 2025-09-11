from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

_cv2 = None

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    TensorOrArray = TypeVar("TensorOrArray", torch.Tensor, npt.NDArray)


def is_cv2_available() -> bool:
    """Check if cv2 is available in the current python environment.

    Returns
    -------
        bool: True if cv2 is available, False otherwise.
    """
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
            return True
        except ImportError:
            _cv2 = False
            return False
    return _cv2 is not False


def import_cv2() -> ModuleType:
    if is_cv2_available():
        return _cv2  # type: ignore[return-value]
    else:
        raise ImportError(  # noqa: B904
            "cv2 is not available in the current python environment.\n"
            "\t This package is required by the fundus toolkits but "
            "we can't add it to the dependencies to prevent versions conflict.\n"
            "\t Please install it by yourself using `pip install opencv-python-headless`."
        )


def is_torch_tensor(x) -> TypeGuard[torch.Tensor]:
    """Check if the input is a torch.Tensor.

    If the type of the input not named "Tensor", it will return False without importing torch.
    """
    if type(x).__qualname__ == "Tensor":
        import torch

        return isinstance(x, torch.Tensor)
    return False
