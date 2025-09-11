from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from pickle import UnpicklingError
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
)
from urllib.error import HTTPError

import torch
from torch.serialization import MAP_LOCATION

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType

from .model_pre_post_processings import PrePostProcessing

T = TypeVar("T", bound=str)


class ModelCache[T]:
    def __init__(self):
        self._name: Optional[T] = None
        self._model: Optional[torch.nn.Module] = None
        self._prepostprocessing: Optional[PrePostProcessing] = None
        _model_caches.append(self)

    def __del__(self):
        """
        Clear the model cache when the object is deleted.
        """
        self.clear()
        _model_caches.remove(self)

    def is_cached(self) -> bool:
        """
        Check if a model is loaded in the cache.

        Returns
        -------
        bool
            True if a model is loaded, False otherwise.
        """
        return self._model is not None

    @property
    def name(self) -> T:
        """
        Get the name of the model.
        Raises
        ------
        ValueError
            If the model name is not set.
        """
        if self._name is None:
            raise AttributeError("Model name is not set.")
        return self._name

    @property
    def model(self) -> torch.nn.Module:
        """
        Get the model.
        Raises
        ------
        ValueError
            If the model is not set.
        """
        if self._model is None:
            raise AttributeError("Model is not set.")
        return self._model

    def model_to(self, device: Optional[DeviceLikeType] = None) -> torch.nn.Module:
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the model to.
        """
        if self._model is not None:
            if device is not None:
                self._model.to(device)
            return self._model
        else:
            raise AttributeError("Model is not set.")

    def has_pre_post_processing(self) -> bool:
        """
        Check if the model has a preprocessing method.

        Returns
        -------
        bool
            True if the model has a preprocessing method, False otherwise.
        """
        return self._prepostprocessing is not None

    @property
    def pre_post_processing(self) -> PrePostProcessing:
        """
        Get the pre/post-processing methods.
        Raises
        ------
        ValueError
            If the pre/post-processing methods are not set.
        """
        if self._prepostprocessing is None:
            raise AttributeError("Pre/post-processing methods are not set.")
        return self._prepostprocessing

    def set_model(self, name: T, model: torch.nn.Module, prepostprocessing: PrePostProcessing) -> None:
        """
        Set the model and its pre/post-processing methods.

        Parameters
        ----------
        name : str
            The name of the model.
        model : T
            The model.
        prepostprocessing : PrePostProcessing
            The pre/post-processing methods.
        """
        self._name = name
        self._model = model
        self._prepostprocessing = prepostprocessing

    def clear(self) -> None:
        """
        Clear the model and its pre/post-processing methods.
        """
        self._name = None
        self._model = None
        self._prepostprocessing = None


_model_caches: List[ModelCache] = []


def clear_gpu_cache():
    """
    Clears the GPU cache.
    """
    import torch

    _model_caches.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


MODEL_DIR = Path(torch.hub.get_dir()) / "checkpoints" / "fundus_toolkits"


def download_state_dict(
    url: str,
    model_name: str,
    toolkit: str,
    subdir: Optional[str] = None,
    *,
    map_location: MAP_LOCATION = "cpu",
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Download a state dict from a URL and save it to the local torch hub directory.

    Parameters
    ----------
    url : str
        The URL to download the state dict from.
    model_name : str
        The name of the model.
    toolkit : str
        The name of the toolkit. (e.g. "vessels", "odmac", "lesions")

    subdir : str, optional
        The subdirectory to save the model in. (e.g. "segmentation", "classification", ...)

    force_download : bool, optional
        Whether to force download the model even if it already exists locally.
        By default: False.
    Returns
    -------
    dict
        The state dict of the model.
    Raises
    -------
    ValueError
        If the URL is invalid.
    RuntimeError
        If the model file is corrupted or the download fails.
    """

    # --- Compute the local model path ---
    local_model_dir = MODEL_DIR / toolkit
    if subdir is not None:
        local_model_dir /= subdir
    local_model_path = local_model_dir / (model_name + ".pth")
    short_model_name = f"{toolkit}/{subdir}/{model_name}" if subdir else f"{toolkit}/{model_name}"

    # --- If force_download is False, remove any existing model file ---
    if force_download:
        local_model_path.unlink(missing_ok=True)

    # --- Fetch the model state from the URL or local path ---
    try:
        return torch.hub.load_state_dict_from_url(
            url,
            map_location=map_location,
            file_name=local_model_path.name,
            model_dir=str(local_model_dir),
            check_hash=True,
            progress=True,
        )
    except UnpicklingError:
        local_model_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"The requested model file {short_model_name} is corrupted, it has been removed. Please retry.\n"
            "If the problem persists contact the author: the url might be invalid."
        ) from None
    except HTTPError as e:
        raise RuntimeError(
            f"An error occurred while downloading the model: {e}\n"
            "Please retry. If the problem persists contact the author."
        ) from None
