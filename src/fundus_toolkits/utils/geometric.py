from __future__ import annotations

import math
from functools import reduce
from types import EllipsisType
from typing import Iterable, List, Literal, NamedTuple, Optional, TypeGuard, overload

import numpy as np
import numpy.typing as npt


class Rect(NamedTuple):
    h: float
    w: float
    y: float = 0
    x: float = 0

    @property
    def center(self) -> Point:
        return Point(self.y + self.h // 2, self.x + self.w // 2)

    @property
    def top_left(self) -> Point:
        return Point(self.y, self.x)

    @property
    def top_right(self) -> Point:
        return Point(self.y, self.x + self.w)

    @property
    def bottom_left(self) -> Point:
        return Point(self.y + self.h, self.x)

    @property
    def bottom_right(self) -> Point:
        return Point(self.y + self.h, self.x + self.w)

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def size(self) -> Point:
        return Point(y=self.h, x=self.w)

    @property
    def shape(self) -> tuple[int, int]:
        return int(math.ceil(self.h)), int(math.ceil(self.w))

    @property
    def area(self) -> float:
        return self.h * self.w

    def corners(self) -> tuple[Point, Point, Point, Point]:
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left

    def to_int(self):
        return Rect(int(math.ceil(self.h)), int(math.ceil(self.w)), int(math.floor(self.y)), int(math.floor(self.x)))

    @classmethod
    def from_tuple(
        cls,
        rect: float
        | int
        | tuple[float | int]
        | tuple[float | int, float | int]
        | tuple[float | int, float | int, float | int, float | int],
    ):
        if np.isscalar(rect):
            if np.issubdtype(type(rect), np.integer):
                rect = (int(rect), int(rect))  # type: ignore[assignment]
            else:
                rect = (float(rect), float(rect))  # type: ignore[assignment]
        elif isinstance(rect, tuple) and len(rect) in (2, 4) and all(np.isscalar(_) for _ in rect):
            pass
        else:
            raise TypeError("Rect can only be created from a float or a tuple of 2 or 4 floats")
        return cls(*rect)

    @classmethod
    def from_size(cls, shape: tuple[float | int, float | int]):
        return cls(shape[0], shape[1])

    @overload
    @classmethod
    def from_points(cls, bottom_right: tuple[float | int, float | int], /) -> Rect: ...
    @overload
    @classmethod
    def from_points(cls, bottom: float | int, right: float | int, /) -> Rect: ...
    @overload
    @classmethod
    def from_points(
        cls,
        top_left: tuple[float | int, float | int],
        bottom_right: tuple[float | int, float | int],
        /,
        *,
        ensure_positive: bool = False,
    ) -> Rect: ...
    @overload
    @classmethod
    def from_points(
        cls,
        top: float | int,
        left: float | int,
        bottom: float | int,
        right: float | int,
        /,
        *,
        ensure_positive: bool = False,
    ) -> Rect: ...
    @overload
    @classmethod
    def from_points(
        cls,
        top_left_bottom_right: tuple[float | int, float | int, float | int, float | int],
        /,
        *,
        ensure_positive: bool = False,
    ) -> Rect: ...
    @classmethod
    def from_points(
        cls,
        *p: float | int | tuple[float | int, float | int] | tuple[float | int, float | int, float | int, float | int],
        ensure_positive: bool = False,
    ) -> Rect:
        match p:
            case ((bottom, right),) if all(np.isscalar(_) for _ in p[0]):
                p2 = Point(bottom, right)
                p1 = Point.origin()
            case (bottom, right) if all(np.isscalar(_) for _ in p):
                p2 = Point(float(bottom), float(right))  # type: ignore[assignment]
                p1 = Point.origin()
            case ((top, left), (bottom, right)) if all(np.isscalar(_) for _ in p[0] + p[1]):
                p1 = Point(top, left)
                p2 = Point(bottom, right)
            case (top, left, bottom, right) if all(np.isscalar(_) for _ in p):
                p1 = Point(top, left)  # type: ignore[assignment]
                p2 = Point(bottom, right)  # type: ignore[assignment]
            case ((top, left, bottom, right),) if all(np.isscalar(_) for _ in p[0]):
                p1 = Point(top, left)
                p2 = Point(bottom, right)
            case _:
                raise TypeError("Rect can only be created from 2 or 4 floats or from 2 tuples of 2 floats")

        if not ensure_positive:
            return cls(abs(p2.y - p1.y), abs(p2.x - p1.x), min(p1.y, p2.y), min(p1.x, p2.x))
        else:
            rect = cls(p2.y - p1.y, p2.x - p1.x, p1.y, p1.x)
            return Rect.empty() if rect.h < 0 or rect.w < 0 else rect

    @classmethod
    def from_center(cls, center: tuple[float, float], shape: float | tuple[float, float]) -> Rect:
        if np.isscalar(shape):
            shape = (float(shape), float(shape))  # type: ignore[assignment]
        return cls(shape[0], shape[1], center[0] - shape[0] // 2, center[1] - shape[1] // 2)  # type: ignore[assignment]

    @classmethod
    def empty(cls) -> Rect:
        return cls(h=0, w=0, y=float("nan"), x=float("nan"))

    @classmethod
    def bounding_box(cls, points: npt.ArrayLike) -> Rect:
        points = np.atleast_2d(points)
        assert points.ndim == 2 and points.shape[-1] == 2, "Array must have shape (n, 2)"
        min_y, min_x = points.min(axis=0)
        max_y, max_x = points.max(axis=0)
        return cls(h=max_y - min_y, w=max_x - min_x, y=min_y, x=min_x)

    def is_empty(self) -> bool:
        return self.w == 0 or self.h == 0

    @classmethod
    def is_empty_rect(cls, rect: Rect | None) -> bool:
        if rect is None:
            return True
        if isinstance(rect, tuple) and len(rect) == 4:
            rect = Rect(*rect)
        return isinstance(rect, tuple) and (rect.w == 0 or rect.h == 0)

    @overload
    def crop_pad_slices(
        src,
        dst: Optional[Rect | tuple[int, int]] = None,
        *,
        prefix_ellipsis: Literal[False] = False,
        src_shape: Optional[tuple[int, int]] = None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[tuple[slice, slice], tuple[slice, slice]]: ...
    @overload
    def crop_pad_slices(
        src,
        dst: Optional[Rect | tuple[int, int]] = None,
        *,
        prefix_ellipsis: Literal[True],
        src_shape: Optional[tuple[int, int]] = None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[tuple[EllipsisType, slice, slice], tuple[EllipsisType, slice, slice]]: ...
    def crop_pad_slices(
        src,
        dst: Optional[Rect | tuple[int, int]] = None,
        *,
        prefix_ellipsis: bool = False,
        src_shape: Optional[tuple[int, int]] = None,
        dst_shape: Optional[tuple[int, int]] = None,
    ) -> (
        tuple[tuple[slice, slice], tuple[slice, slice]]
        | tuple[tuple[EllipsisType, slice, slice], tuple[EllipsisType, slice, slice]]
    ):
        """Provide slices to crop/pad this rectangle to fit into the destination rectangle.

        Parameters
        ----------
        dst : Rect | tuple[int, int] | None
            The destination rectangle or shape (height, width). If None, use the shape of this rectangle.

        Returns
        -------
        src: tuple[slice, slice]
            The slices to crop this rectangle.

        dst: tuple[slice, slice], optional
            The slices to pad into the destination rectangle. If None, assume that the destination rectangle has its top left at (0, 0), and the same size as this rectangle.

        prefix_ellipsis: bool, optional
            If True, add an ellipsis at the beginning of the slices to support images with more than 2 dimensions.

        src_shape: tuple[int, int], optional
            The shape of the source image. If provided, the source rectangle will be clipped to this shape.

        dst_shape: tuple[int, int], optional
            The shape of the destination image. If provided, the destination rectangle will be clipped to this shape.

        Raises
        ------
        TypeError
            If the input is not a Rect or a tuple of 2 integers.
        """
        src_dy, src_dx = max(0, -src.top_left.y), max(0, -src.top_left.x)

        if dst is None:
            h, w = src.h - src_dy, src.w - src_dx
            if src_shape is not None:
                h, w = h - max(0, src.bottom - src_shape[0]), w - max(0, src.right - src_shape[1])

            src = Rect(h, w, src.y + src_dy, src.x + src_dx)
            dst = Rect(h, w, src_dy, src_dx)
        else:
            dst = Rect.from_tuple(dst)
            assert dst.size == src.size, "Source and destination rectangles must have the same size"

            dy, dx = max(-dst.top_left.y, src_dy), max(-dst.top_left.x, src_dx)

            dh, dw = 0, 0
            if src_shape is not None:
                dh, dw = max(dh, src.bottom - src_shape[0]), max(dw, src.right - src_shape[1])
            if dst_shape is not None:
                dh, dw = max(dh, dst.bottom - dst_shape[0]), max(dw, dst.right - dst_shape[1])

            h, w = src.h - dy - dh, src.w - dx - dw
            src = Rect(h, w, src.y + dy, src.x + dx)
            dst = Rect(h, w, dst.y + dy, dst.x + dx)

        src_slice, dst_slice = src.slice(), dst.slice()
        if prefix_ellipsis:
            return (Ellipsis, src_slice[0], src_slice[1]), (Ellipsis, dst_slice[0], dst_slice[1])
        return src_slice, dst_slice

    def crop_pad_image[T: np.generic](
        self,
        image: npt.NDArray[T],
        dst: Optional[Rect | tuple[int, int]] = None,
        dst_shape: Optional[tuple[int, int]] = None,
        *,
        channel_last: bool = False,
    ) -> npt.NDArray[T]:
        """Crop/pad an image to fit into the destination rectangle.

        Parameters
        ----------
        image : npt.NDArray[T]
            The image to crop/pad with shape [..., height, width].

        dst : Rect | tuple[int, int] | None
            The destination rectangle or shape (height, width). If None, use the shape of this rectangle.

        dst_shape : tuple[int, int], optional
            The shape of the destination image. If provided, the destination rectangle will be clipped to this shape.

        channel_last : bool, optional
            If True, the input image should have shape [height, width, C] and the output image will have shape [height, width, C].

        Returns
        -------
        npt.NDArray[T]
            The cropped/padded image.

        Raises
        ------
        TypeError
            If the input is not a Rect or a tuple of 2 integers.
        """
        if channel_last and image.ndim == 3:
            image = image.transpose(2, 0, 1)
        src_slice, dst_slice = self.crop_pad_slices(
            dst, src_shape=image.shape[-2:], dst_shape=dst_shape, prefix_ellipsis=True
        )

        if dst_shape is None:
            if dst is None:
                dst_shape = self.shape
            else:
                if not isinstance(dst, Rect):
                    dst = Rect.from_tuple(dst)
                dst_shape = dst.shape

        dst_image = np.zeros(image.shape[:-2] + dst_shape, dtype=image.dtype)
        dst_image[dst_slice] = image[src_slice]
        if channel_last and dst_image.ndim == 3:
            dst_image = dst_image.transpose(1, 2, 0)
        return dst_image

    def grid_indices(self) -> npt.NDArray[np.int_]:
        """Get the grid indices of the rectangle as an array of shape (h, w, 2)"""
        yy, xx = np.meshgrid(
            np.arange(int(math.ceil(self.h))) + self.y,
            np.arange(int(math.ceil(self.w))) + self.x,
            indexing="ij",
        )
        return np.stack((yy, xx), axis=-1)

    def exclude_bottom_right_edges(self) -> Rect:
        return Rect(self.h - 1, self.w - 1, self.y, self.x)

    @classmethod
    def is_rect(cls, r) -> TypeGuard[Rect]:
        return isinstance(r, Rect) or (isinstance(r, tuple) and len(r) == 4)

    def __repr__(self):
        return "Rect(y={}, x={}, h={}, w={})".format(self.y, self.x, self.h, self.w)

    def __or__(self, other) -> Rect:
        if isinstance(other, Rect):
            if self.is_empty():
                return other
            if other.is_empty():
                return self
            return Rect.from_points(
                (min(self.top, other.top), min(self.left, other.left)),
                (max(self.bottom, other.bottom), max(self.right, other.right)),
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __and__(self, other) -> Rect:
        if isinstance(other, Rect):
            return Rect.from_points(
                (max(self.top, other.top), max(self.left, other.left)),
                (min(self.bottom, other.bottom), min(self.right, other.right)),
                ensure_positive=True,
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __add__(self, other: Point | float) -> Rect:  # type: ignore[override]
        other = Point.from_tuple(other)
        return self.translate(other.y, other.x)

    def __sub__(self, other: Point | float) -> Rect:
        other = Point.from_tuple(other)
        return self.translate(-other.y, -other.x)

    def __mul__(self, other: float) -> Rect:  # type: ignore[override]
        return self.scale(other)

    def __truediv__(self, other: float) -> Rect:
        return self.scale(1 / other)

    def __contains__(self, other: Point | Rect) -> bool:  # type: ignore[override]
        if isinstance(other, Point):
            return self.y <= other.y <= self.y + self.h and self.x <= other.x <= self.x + self.w
        elif isinstance(other, Rect):
            return not Rect.is_empty_rect(self & other)
        else:
            raise TypeError("Rect can only be compared with a Point or a Rect")

    def translate(self, y: float, x: float) -> Rect:
        return Rect(self.h, self.w, self.y + y, self.x + x)

    def scale(self, fy: float, fx: float | None = None) -> Rect:
        if fx is None:
            fx = fy
        return Rect(self.h * fy, self.w * fx, self.y * fy, self.x * fx)

    @overload
    def clip(self, points: Point | tuple[float, float]) -> Point: ...
    @overload
    def clip(self, points: npt.NDArray[float]) -> npt.NDArray[np.float_]: ...
    def clip(self, points: Point | tuple[float, float] | npt.NDArray[float]) -> Rect | npt.NDArray[np.float_]:
        """Clip a point or an array of points to the rectangle

        Parameters
        ----------
        points : Point | tuple[float, float] | npt.NDArray[float]
            The point or array of points to clip

        Returns
        -------
        Rect | npt.NDArray[np.float_]
            The clipped point or array of points

        Raises
        ------
        TypeError
            If the input is not a Point, a tuple of 2 floats or an array of points
        """
        if isinstance(points, np.ndarray):
            is_single = points.ndim == 1
            if is_single:
                points = points[np.newaxis, :]
            assert points.ndim == 2 and points.shape[-1] == 2, "Array must have shape (n, 2)"
            clipped = np.empty_like(points)
            clipped[..., 0] = np.clip(points[..., 0], self.top, self.bottom)
            clipped[..., 1] = np.clip(points[..., 1], self.left, self.right)
            return clipped
        elif isinstance(points, tuple) and len(points) == 2:
            points = Point.from_tuple(points)
        elif not isinstance(points, Point):
            raise TypeError("Rect can only be used to clip a Point or an array of Points")
        return Point(
            min(max(points.y, self.top), self.bottom),
            min(max(points.x, self.left), self.right),
        )

    def clip_to_size(self, shape: tuple[float, float], center: Optional[tuple[float, float]] = None):
        if center is None:
            center = self.center
        h, w = self.shape
        H, W = shape
        x0, y0 = self.top_left
        xC, yC = center
        if h > H:
            y0 = max(xC - H / 2, y0) if xC - y0 > y0 + h - xC else min(xC - H / 2, y0 + h - H)
        if w > W:
            x0 = max(yC - W / 2, x0) if yC - x0 > x0 + w - xC else min(yC - W / 2, x0 + w - W)
        return Rect.from_points((y0, x0), (y0 + min(h, H), x0 + min(w, W)))

    @overload
    def pad(self, pad: float | tuple[float, float], /) -> Rect: ...
    @overload
    def pad(self, vertical: float, horizontal: float, /) -> Rect: ...
    @overload
    def pad(self, top: float, right: float, bottom: float, left: float, /) -> Rect: ...
    def pad(self, *pad: float | tuple[float, float]) -> Rect:
        if len(pad) == 1 and np.isscalar(pad[0]):
            p = float(pad[0])  # type: ignore[assignment]
            pad = (p, p, p, p)
        elif len(pad) == 1 and isinstance(pad[0], tuple) and len(pad[0]) == 2 and all(np.isscalar(_) for _ in pad[0]):
            # case ((vertical, horizontal), )
            h = float(pad[0][0])
            v = float(pad[0][1])
            pad = (h, v, h, v)
        elif len(pad) == 2 and all(np.isscalar(_) for _ in pad):
            # case (vertical, horizontal)
            h = float(pad[0])  # type: ignore[assignment]
            v = float(pad[1])  # type: ignore[assignment]
            pad = (h, v, h, v)
        elif len(pad) == 4 and all(np.isscalar(_) for _ in pad):
            # case (top, right, bottom, left)
            top = float(pad[0])  # type: ignore[assignment]
            right = float(pad[1])  # type: ignore[assignment]
            bottom = float(pad[2])  # type: ignore[assignment]
            left = float(pad[3])  # type: ignore[assignment]
            pad = (top, right, bottom, left)
        else:
            raise TypeError("Rect.pad() only accept 1, 2 or 4 floats as arguments")

        return Rect(self.h + pad[0] + pad[2], self.w + pad[1] + pad[3], self.y - pad[0], self.x - pad[3])

    def box(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def slice(self) -> tuple[slice, slice]:
        r = self.to_int()
        y, x = int(r.y), int(r.x)
        h, w = int(r.h), int(r.w)
        return slice(y, y + h), slice(x, x + w)

    @overload
    def contains(self, other: Point | Rect) -> bool: ...
    @overload
    def contains(self, other: npt.NDArray) -> npt.NDArray[np.bool_]: ...
    def contains(self, other: Point | Rect | npt.ArrayLike) -> bool | npt.NDArray[np.bool_]:
        if isinstance(other, Point):
            return self.y <= other.y <= self.y + self.h and self.x <= other.x <= self.x + self.w
        elif isinstance(other, Rect):
            return (
                (self.y <= other.y)
                & (self.x <= other.x)
                & (self.y + self.h >= other.y + other.h)
                & (self.x + self.w >= other.x + other.w)
            )
        elif isinstance(other, np.ndarray):
            other = np.atleast_2d(other)
            assert other.shape[-1] in (2, 4), "Array must have shape (n, 2) or (n, 4)"
            if other.shape[-1] == 2:
                return (
                    np.isfinite(other).all(axis=-1)
                    & (self.y <= other[..., 0])
                    & (self.x <= other[..., 1])
                    & (self.y + self.h >= other[..., 0])
                    & (self.x + self.w >= other[..., 1])
                )
            return (
                np.isfinite(other).all(axis=-1)
                & (self.y <= other[..., 0])
                & (self.x <= other[..., 1])
                & (self.y + self.h >= other[..., 0] + other[..., 2])
                & (self.x + self.w >= other[..., 1] + other[..., 3])
            )
        else:
            raise TypeError("Rect can only be compared with a Point, a Rect or an array of Points or Rects")

    @staticmethod
    def union(*rects: Iterable[Rect] | Rect) -> Rect:
        if not rects:
            return Rect.empty()
        r = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a | b, r)  # type: ignore

    @staticmethod
    def intersection(*rects: Iterable[Rect] | Rect) -> Rect:
        """Compute the intersection of a list of rectangles.
        If the intersection is empty, return an empty rectangle.

        Parameters
        ----------
        rects: An iterables of rectangles.

        Returns
        -------
        A rectangle representing the intersection of the input rectangles.

        Example
        -------
        >>> r1 = Rect(0, 0, 10, 10)
        >>> r2 = Rect(5, 5, 10, 10)
        >>> r3 = Rect(15, 15, 10, 10)
        >>> Rect.intersection(r1, r2, r3)
        Rect(5, 5, 0, 0)
        >>> Rect.intersection(r1, r2)
        Rect(5, 5, 10, 10)
        >>> Rect.intersection(r1, r3)
        Rect(0, 0, 0, 0)
        """
        r = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a & b, r)  # type: ignore


class Point(NamedTuple):
    y: float
    x: float

    def __str__(self) -> str:
        return f"({self.y:.1f}, {self.x:.1f})"

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def __add__(self, other: tuple[float, float] | float) -> Point:  # type: ignore[override]
        if np.isscalar(other):
            other = float(other)  # type: ignore[assignment]
            return Point(self.y + other, self.x + other)
        y, x = other  # type: ignore[assignment]
        return Point(float(self.y + y), float(self.x + x))

    def __radd__(self, other: tuple[float, float] | float) -> Point:
        return self + other

    def __sub__(self, other: tuple[float, float] | float) -> Point:  # type: ignore[override]
        if np.isscalar(other):
            other = float(other)  # type: ignore[assignment]
            return Point(self.y - other, self.x - other)
        y, x = other  # type: ignore[assignment]
        return Point(float(self.y - y), float(self.x - x))

    def __rsub__(self, other: tuple[float, float] | float):
        return -(self - other)

    def __neg__(self) -> Point:  # type: ignore[override]
        return Point(-self.y, -self.x)

    def __mul__(self, other: tuple[float, float] | float) -> Point:  # type: ignore[override]
        if np.isscalar(other):
            other = float(other)  # type: ignore[assignment]
            return Point(self.y * other, self.x * other)
        y, x = other  # type: ignore[assignment]
        return Point(self.y * float(y), self.x * float(x))

    def __rmul__(self, other: tuple[float, float] | float) -> Point:  # type: ignore[override]
        return self * other

    def __truediv__(self, other: tuple[float, float] | float) -> Point:  # type: ignore[override]
        if np.isscalar(other):
            other = float(other)  # type: ignore[assignment]
            return Point(self.y / other, self.x / other)
        y, x = other  # type: ignore[assignment]
        return Point(self.y / float(y), self.x / float(x))

    def __rtruediv__(self, other: tuple[float, float] | float):
        if np.isscalar(other):
            other = float(other)  # type: ignore[assignment]
            return Point(other / self.y, other / self.x)
        y, x = other  # type: ignore[assignment]
        return Point(float(y) / self.y, float(x) / self.x)

    @classmethod
    def origin(cls):
        return cls(0, 0)

    @classmethod
    def from_tuple(cls, point: float | int | tuple[float | int, float | int]):
        if np.isscalar(point):
            p = float(point)  # type: ignore[assignment]
            return cls(p, p)
        if isinstance(point, tuple) and len(point) == 2 and all(np.isscalar(_) for _ in point):
            return cls(float(point[0]), float(point[1]))  # type: ignore[assignment]
        raise TypeError("Point can only be created from a float or a tuple of 2 floats")

    @classmethod
    def from_array(cls, point: npt.NDArray[np.float_]) -> Point:
        return cls(float(point[0]), float(point[1]))

    def numpy(self) -> np.ndarray:
        return np.array(self)

    @overload
    def distance(self, other: Point) -> float: ...
    @overload
    def distance(self, other: List[Point]) -> List[float]: ...
    @overload
    def distance(self, other: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]: ...
    def distance(self, other: Point | Iterable[Point]) -> float | Iterable[float] | npt.NDArray[np.float_]:
        if isinstance(other, np.ndarray):
            return np.linalg.norm(other - self, axis=-1)
        elif isinstance(other, Point):
            return ((self.y - other.y) ** 2 + (self.x - other.x) ** 2) ** 0.5
        elif isinstance(other, Iterable):
            return [self.distance(p) for p in other]
        raise TypeError("Point can only be compared with a Point or an array of Points")

    def is_nan(self) -> bool:
        return np.isnan(self.y) or np.isnan(self.x)

    def to_int(self) -> Point:
        return Point(int(round(self.y)), int(round(self.x)))

    def to_int_pair(self) -> tuple[int, int]:
        return int(round(self.y)), int(round(self.x))

    def ceil(self) -> Point:
        return Point(int(math.ceil(self.y)), int(math.ceil(self.x)))

    def floor(self) -> Point:
        return Point(int(math.floor(self.y)), int(math.floor(self.x)))

    def clip(self, rect: float | tuple[float, float] | tuple[float, float, float, float]) -> Point:
        rect = Rect.from_tuple(rect)
        return Point(
            min(max(self.y, rect.top), rect.bottom),
            min(max(self.x, rect.left), rect.right),
        )

    @property
    def norm(self) -> float:
        """The norm of the point"""
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def angle(self) -> float:
        """The angle of the point in radians"""
        return math.atan2(self.y, self.x)

    @property
    def max(self) -> float:
        """The maximum of y and x"""
        return max(self.y, self.x)

    @property
    def min(self) -> float:
        """The minimum of y and x"""
        return min(self.y, self.x)

    def normalized(self) -> Point:
        """Return the point normalized to unit length

        Example
        -------
        >>> p = Point(0, 4)
        >>> p.normalized()
        Point(0.0, 1.0)
        >>> p = Point(3, 4)
        >>> p.normalized()
        Point(0.6, 0.8)
        """
        norm = self.norm
        return Point.origin() if norm == 0 else self / norm

    def cross(self, other: Point, normalize=False) -> float:
        if normalize:
            self = self.normalized()
            other = other.normalized()
        return self.x * other.y - self.y * other.x

    def dot(self, other: Point, normalize=False) -> float:
        if normalize:
            self = self.normalized()
            other = other.normalized()
        return self.x * other.x + self.y * other.y

    def rot90(self) -> Point:
        """Rotate the point by 90 degrees counter-clockwise

        Returns
        -------
        Point
            The rotated point
        """
        return Point(-self.x, self.y)

    def rotate(self, angle: float, degrees: bool = False) -> Point:
        """Rotate the point by a given angle in radians counter-clockwise

        Parameters
        ----------
        angle: float
            The angle in radians

        Returns
        -------
        Point
            The rotated point
        """
        angle = math.radians(angle) if degrees else angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Point(
            self.y * cos_a - self.x * sin_a,
            self.y * sin_a + self.x * cos_a,
        )


ABSENT = Point(float("nan"), float("nan"))


def distance_matrix(points_coord: np.ndarray):
    """
    Compute the distance matrix between a set of points.

    Parameters
    ----------
    nodes_coord: A 2D array of shape (nb_nodes, 2) containing the coordinates of the nodes.

    Returns
    -------
    A 2D array of shape (nb_nodes, nb_nodes) containing the distance between each pair of nodes.
    """
    return np.linalg.norm(points_coord[:, None, :] - points_coord[None, :, :], axis=2)
