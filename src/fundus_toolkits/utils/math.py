import numpy as np

from .geometric import Point
from .typing import PointArrayLike, as_points


def ensure_superior_multiple(x, m):
    """
    Return the smallest integer greater than or equal to x that is a multiple of m.
    """
    return m - (x - 1) % m + x - 1


def modulo_pi(x):
    """Wraps an angle in radians to the range [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def fit_circle(points: PointArrayLike) -> tuple[Point, float]:
    """Fit a circle to a set of points using least squares. Returns the center and radius of the fitted circle."""
    points = as_points(points)
    n = points.shape[0]
    y, x = points.T
    y_sqr, x_sqr = np.square(points).T
    x_y_sqr = x_sqr + y_sqr

    Y = y.sum()
    X = x.sum()
    Y_sqr = y_sqr.sum()
    X_sqr = x_sqr.sum()
    XY = (y * x).sum()

    # ⎡ 2 Σxᵢ² , 2 Σxᵢyᵢ, Σxᵢ ⎤   ⎡ a ⎤   ⎡ Σxᵢ(xᵢ² + yᵢ²) ⎤
    # ⎢ 2 Σxᵢyᵢ, 2 Σyᵢ² , Σyᵢ ⎥ * ⎢ b ⎥ = ⎢ Σyᵢ(xᵢ² + yᵢ²) ⎥
    # ⎣ 2 Σxᵢ  , 2 Σyᵢ  ,  n  ⎦   ⎣ c ⎦   ⎣ Σ  (xᵢ² + yᵢ²) ⎦
    A = np.array([[2 * X_sqr, 2 * XY, X], [2 * XY, 2 * Y_sqr, Y], [2 * X, 2 * Y, n]])
    B = np.array([(x * x_y_sqr).sum(), (y * x_y_sqr).sum(), x_y_sqr.sum()])
    a, b, c = np.linalg.solve(A, B)

    return Point(float(b), float(a)), float(np.sqrt(a**2 + b**2 + c))
