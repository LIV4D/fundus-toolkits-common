from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, TypeAlias, Union

import numpy as np


def load_av(file, av_inverted=False, pad=None):
    from .safe_import import import_cv2

    cv2 = import_cv2()

    av_color = cv2.imread(str(file))
    if av_color is None:
        raise ValueError(f"Could not load image from {file}")

    av = np.zeros(av_color.shape[:2], dtype=np.uint8)  # Unknown
    v = av_color.mean(axis=2) > 10
    a = av_color[:, :, 2] > av_color[:, :, 0]
    if av_inverted:
        a = ~a
    av[v & a] = 1  # Artery
    av[v & ~a] = 2  # Vein

    if pad is not None:
        av = np.pad(av, pad, mode="constant", constant_values=0)
    return av
