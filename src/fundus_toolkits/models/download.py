from __future__ import annotations

from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Dict, Optional
from urllib.error import HTTPError

import torch
from torch.serialization import MAP_LOCATION

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
