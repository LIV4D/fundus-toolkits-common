from typing import Generator

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


def color_jitter(
    color: ColorSpec,
    hue: float = 0,
    saturation: float = 0,
    value: float = 0,
    rng: np.random.Generator | None = None,
) -> Generator[npt.NDArray[np.uint8], None, None]:
    """Apply random jitter to a color.

    Parameters
    ----------
    color : np.ndarray[np.uint8]
        The input color as an array of shape (3,).
    hue : float
        The amount of hue jitter to apply. Should be in [0, 1].
    saturation : float
        The amount of saturation jitter to apply. Should be in [0, 1].
    value : float
        The amount of value jitter to apply. Should be in [0, 1].
    rng : np.random.Generator, optional
        The random number generator to use. If None, a new generator will be created.

    Returns
    -------
    np.ndarray[np.uint8]
        The jittered color as an array of shape (3,).
    """
    from coloraide import Color

    if rng is None:
        rng = np.random.default_rng()

    color = parse_color(color)
    color = Color("srgb", color.astype(np.float32) / 255.0).convert("hsv")

    while True:
        h, s, v = color.coords()
        h_jitter = rng.uniform(-hue, hue) * 360 if hue > 0 else 0.0
        s_jitter = rng.uniform(-saturation, saturation) if saturation > 0 else 0.0
        v_jitter = rng.uniform(-value, value) if value > 0 else 0.0

        h_new = (h + h_jitter) % 360
        s_new = np.clip(s + s_jitter, 0, 1)
        v_new = np.clip(v + v_jitter, 0, 1)

        jittered_color = Color("hsv", (h_new, s_new, v_new)).convert("srgb")
        yield np.array([(c * 255) for c in jittered_color.coords()], dtype=np.uint8)
