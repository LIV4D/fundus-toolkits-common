import numpy as np
import numpy.typing as npt

type ColorSpec = str | tuple[int, int, int] | tuple[float, float, float] | npt.NDArray[np.uint8]


def parse_color(color: ColorSpec) -> npt.NDArray[np.uint8]:
    if isinstance(color, str):
        from coloraide import Color

        return np.array([(c * 255) for c in Color(color).convert("srgb").coords()], dtype=np.uint8)
    elif isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError(f"Color tuple must have 3 elements, got {len(color)}")
        if all(isinstance(c, float) for c in color):
            return (np.clip(color, 0, 1) * 255).astype(np.uint8)
        elif all(isinstance(c, int) for c in color):
            return np.clip(color, 0, 255).astype(np.uint8)
    elif isinstance(color, np.ndarray):
        if color.shape != (3,) or color.dtype != np.uint8:
            raise ValueError(f"Color array must have shape (3,) and dtype uint8, got {color.shape} and {color.dtype}")
        return color
    raise ValueError(f"Invalid color specification: {color}")
