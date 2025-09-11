import numpy as np


def ensure_superior_multiple(x, m=32):
    """
    Return the smallest integer greater than or equal to x that is a multiple of m.
    """
    return m - (x - 1) % m + x - 1


def modulo_pi(x):
    """Wraps an angle in radians to the range [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi
