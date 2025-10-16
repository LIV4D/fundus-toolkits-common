__all__ = ["cache_model", "clear_gpu_cache"]
from functools import lru_cache
from collections.abc import Callable
from typing import Any

import torch


def cache_model(
    max_size: int = 2,
) -> Callable[[Callable[..., torch.nn.Module]], Callable[..., torch.nn.Module]]:
    """
    Decorator that caches a model created by a function, so that the model can be de-instantiated using ``clear_gpu_cache()``.
    This method internally uses ``functools.lru_cache`` to cache the model.

    Parameters
    ----------
    max_size : int
        The maximum number of models to cache. Default is 2.

    Exemples
    --------
    >>> @cache_model(max_size=2)
    ... def load_model(model_name: str) -> torch.nn.Module:
    ...     # Load the model here
    ...     return model
    """

    def decorator(f: Callable[..., torch.nn.Module]) -> Callable[..., torch.nn.Module]:
        f_decorated = lru_cache(maxsize=max_size)(f)
        _model_factories.append(f_decorated)
        return f_decorated

    return decorator


_model_factories: list[Any] = []


def clear_gpu_cache(empty_torch_cache: bool = True) -> None:
    """
    Clears the GPU cache.
    """
    for f in _model_factories:
        f.cache_clear()

    if empty_torch_cache:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
