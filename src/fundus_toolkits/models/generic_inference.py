import functools
import typing
from collections.abc import Callable, Sequence
from typing import Literal, NotRequired, Protocol, TypedDict, TypeGuard, overload

import numpy as np
import numpy.typing as npt
import torch

from ..fundus_data import FundusData
from ..utils.typing import PathLike


class FundusInferenceInternalFunc[**P](Protocol):
    def __call__(
        self,
        fundus: torch.Tensor,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> torch.Tensor: ...


class FundusInferenceFunc[**P](Protocol):
    def __call__(
        self,
        fundus: torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike] | FundusData,
        fundus_mask: Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None = "auto",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> torch.Tensor: ...


class FundusPreprocessInfo(TypedDict):
    is_numpy: NotRequired[bool]
    fundus_data: NotRequired[Sequence[FundusData] | FundusData]
    is_single: NotRequired[bool]


type FundusDataFieldUpdater = Callable[[FundusData, npt.NDArray], None]


class GenericFundusInference[**P]:
    """Use a model to segment a fundus image or a batch of fundus images."""

    def __init__(
        self, infer_func: FundusInferenceInternalFunc[P], fundus_data_field: FundusData.Fields | FundusDataFieldUpdater
    ) -> None:
        self.__wrapped__ = infer_func
        self._fundus_data_field = fundus_data_field
        functools.wraps(infer_func)(self)

    @overload
    def __call__(
        self,
        fundus: torch.Tensor,
        fundus_mask: Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None = "auto",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> torch.Tensor: ...
    @overload
    def __call__(
        self,
        fundus: npt.NDArray | PathLike | Sequence[PathLike] | FundusData,
        fundus_mask: Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None = "auto",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> npt.NDArray: ...
    def __call__(
        self,
        fundus: torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike] | FundusData,
        fundus_mask: Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None = "auto",
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> torch.Tensor | npt.NDArray:
        """Use a model to segment a fundus image or a batch of fundus images.

        Parameters
        ----------
        fundus : torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike] | FundusData
            The fundus image(s) to segment. Must be one of:
            - Fundus image(s) as a tensor or array of shape (3, H, W) or (B, 3, H, W) with pixel values in [0, 1].
            - Path(s) to the fundus image file(s).
            - FundusData object(s) containing the fundus image(s).

            In the last cases the FundusData object(s) will be updated in-place with the segmentation result, and should therefore be mutable.

        fundus_mask : Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None, optional
            Mask(s) used in a post-process step to remove any segmentation artefacts outside the fundus area.

            Must be one of:
            - 'auto' (default): if FundusData object(s) are provided, their fundus_mask attribute will be used if available, otherwise the mask will be inferred from the fundus image;
            - Path(s) to the fundus mask file(s);
            - A binary tensor or array of shape (H, W) or (B, H, W);
            - None: no mask will be applied.

        *args, **kwargs
            Additional arguments passed to the inference function.

        Returns
        -------
        torch.Tensor | npt.NDArray
            The segmentation map(s) as a tensor or array of shape (C, H, W) or (B, C, H, W).

            If the input fundus was provided as a tensor, the output will be a tensor. In all other cases, the output will be a numpy array. If FundusData object(s) were provided, they will be updated in-place with the segmentation result.
        """  # noqa: E501
        with torch.inference_mode():
            fundus, infos = self.preprocess_fundus(fundus)
            fundus_mask = self.preprocess_fundus_mask(fundus, fundus_mask, infos)
            tensor = self.__wrapped__(fundus, *args, **kwargs)
            return self.postprocess_output_map(tensor, infos, fundus_mask)

    @staticmethod
    def preprocess_fundus(
        fundus: torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike] | FundusData | Sequence[FundusData],
    ) -> tuple[torch.Tensor, FundusPreprocessInfo]:
        infos = FundusPreprocessInfo()

        # === CAST TO TENSOR ... ===
        # ... from list of FundusData
        if is_fundus_data_sequence(fundus):
            for f in fundus:
                check_fundus_data(f)
            infos["fundus_data"] = fundus
            fundus = np.stack([f.image for f in fundus], axis=0)
        # ... from list of paths
        elif is_path_sequence(fundus):
            imgs = [FundusData.load_fundus_image(f) for f in fundus]
            assert len(imgs) <= 1 or all(i.shape == imgs[0].shape for i in imgs), (
                "All fundus images must have the same shape. Got " + ", ".join(str(i.shape) for i in imgs)
            )
            fundus = np.stack(imgs, axis=0)
        # ... from FundusData
        elif isinstance(fundus, FundusData):
            assert fundus.mutable, (
                "FundusData passed to the inference function must be mutable."
                " Please call ``fundus.mutable_copy()`` first."
            )
            infos["fundus_data"] = fundus
            fundus = fundus.image
        # ... from path
        elif isinstance(fundus, PathLike):
            fundus = FundusData.load_fundus_image(fundus)

        # ... from numpy array (and all the above cases)
        infos["is_numpy"] = isinstance(fundus, np.ndarray)
        if infos["is_numpy"]:
            fundus = torch.from_numpy(fundus)

        # Check fundus is now a tensor
        assert isinstance(fundus, torch.Tensor), "fundus must be a torch.Tensor, numpy.ndarray, PathLike or FundusData."

        # === CAST TO FLOAT ===
        if not torch.is_floating_point(fundus):
            fundus = fundus.float() / 255.0

        # === ADD BATCH DIMENSION & CHANNEL FIRST ===
        if fundus.shape[-1] == 3:
            reorder = (2, 0, 1) if fundus.ndim == 3 else (0, 3, 1, 2)
            fundus = fundus.permute(reorder)  # HWC -> CHW

        infos["is_single"] = len(fundus.shape) == 3
        if infos["is_single"]:
            fundus = fundus.unsqueeze(0)  # CHW -> BCHW

        assert len(fundus.shape) == 4 and fundus.shape[1] == 3, (
            f"fundus must have shape BCHW with 3 channels. Got {fundus.shape}."
        )

        return fundus, infos

    @staticmethod
    def preprocess_fundus_mask(
        fundus: torch.Tensor,
        fundus_mask: Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None = "auto",
        infos: FundusPreprocessInfo | None = None,
    ) -> torch.Tensor | None:
        if fundus_mask is None:
            return None

        # === LOAD FUNDUS MASK ... ===
        if fundus_mask == "auto":
            fundus_masks: list[None | npt.NDArray[np.bool_]] = [None for _ in range(fundus.shape[0])]
            # ... from FundusData
            fundus_data = None
            if infos is not None and (fundus_data := infos.get("fundus_data") if infos else None) is not None:
                if infos.get("is_single", False):
                    assert isinstance(fundus_data, FundusData), "Mismatch between input and output."
                    if fundus_data.has_fundus_mask:
                        fundus_masks[0] = fundus_data.fundus_mask
                else:
                    assert is_fundus_data_sequence(fundus_data) and len(fundus_data) == len(fundus_masks), (
                        "Mismatch between input and output."
                    )
                    fundus_masks = [f.fundus_mask if f.has_fundus_mask else None for f in fundus_data]
            # ... infer from fundus image
            if any(m is None for m in fundus_masks):
                for i, m in enumerate(fundus_masks):
                    if m is None:
                        fundus_masks[i] = FundusData.load_fundus_mask(fundus[i], from_fundus=True)

            # Update FundusData if provided
            fundus_masks_ = typing.cast(list[npt.NDArray[np.bool_]], fundus_masks)
            if isinstance(fundus_data, FundusData):
                if not fundus_data.has_fundus_mask:
                    fundus_data.update(fundus_mask=fundus_masks_[0], inplace=True)
            elif is_fundus_data_sequence(fundus_data):
                for d, mask in zip(fundus_data, fundus_masks_, strict=True):
                    if not d.has_fundus_mask:
                        d.update(fundus_mask=mask, inplace=True)

            # Stack masks
            fundus_mask = np.stack(fundus_masks_, axis=0)

        # ... from list of paths
        elif is_path_sequence(fundus_mask):
            fundus_mask = np.stack([FundusData.load_fundus_mask(f) for f in fundus_mask], axis=0)
        # ... from path
        elif isinstance(fundus_mask, PathLike):
            fundus_mask = FundusData.load_fundus_mask(fundus_mask)

        # ... from numpy array (and all the above cases)
        if isinstance(fundus_mask, np.ndarray):
            fundus_mask = torch.from_numpy(fundus_mask) > 0.5

        # Check roi_mask is now a tensor
        assert isinstance(fundus_mask, torch.Tensor), (
            "fundus_mask must be a torch.Tensor, numpy.ndarray, PathLike or 'auto'."
        )

        # === CAST TO BOOL ===
        if not fundus_mask.dtype == torch.bool:
            fundus_mask = fundus_mask > 0.5

        # === ADD BATCH DIMENSION ===
        if fundus_mask.ndim == 2:
            fundus_mask = fundus_mask.unsqueeze(0)  # HW -> BHW

        assert fundus_mask.ndim == 3, f"fundus_mask must have shape BHW. Got {fundus_mask.shape}."
        assert fundus_mask.shape[0] in (1, fundus.shape[0]), (
            f"fundus_mask must have the same batch size as fundus. Got {fundus_mask.shape[0]} and {fundus.shape[0]}."
        )
        assert fundus_mask.shape[-2:] == fundus.shape[-2:], (
            f"fundus_mask must have the same height and width as fundus. Got {fundus_mask.shape[-2:]} and {fundus.shape[-2:]}."
        )

        return fundus_mask

    def postprocess_output_map(
        self,
        map: torch.Tensor,
        infos: FundusPreprocessInfo,
        fundus_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | npt.NDArray:
        # === MASK ROI ===
        if fundus_mask is not None:
            assert fundus_mask.ndim == 3, f"fundus_mask must have shape BHW. Got {fundus_mask.shape}."
            assert fundus_mask.shape[0] in (1, map.shape[0]), (
                f"fundus_mask must have the same batch size as output map. Got {fundus_mask.shape[0]} and {map.shape[0]}."
            )
            assert fundus_mask.shape[-2:] == map.shape[-2:], (
                f"fundus_mask must have the same height and width as output map. Got {fundus_mask.shape[-2:]} and {map.shape[-2:]}."
            )
            fundus_mask = fundus_mask.to(map.device)
            if map.ndim == 4:
                fundus_mask = fundus_mask.unsqueeze(1)  # BHW -> B1HW
            map = map * fundus_mask

        # === REMOVE BATCH DIMENSION ===
        if is_single := infos.get("is_single", False):
            map = map.squeeze(0)  # BCHW -> CHW

        # === CAST TO ORIGINAL DTYPE ===
        if infos.get("is_numpy", True):
            # Cast to numpy ...
            map_np = map.numpy(force=True)

            if (fundus_data := infos.get("fundus_data")) is not None:
                # ... and update FundusData if provided
                mismatch_error = "Mismatch between input and output"
                if self.func_name:
                    mismatch_error += f" in {self.func_name}()"

                if is_single:
                    assert isinstance(fundus_data, FundusData), "Mismatch between input and output."
                    if callable(self._fundus_data_field):
                        self._fundus_data_field(fundus_data, map_np)  # type: ignore[arg-type]
                    else:
                        fundus_data.update(**{self._fundus_data_field: map_np}, inplace=True)  # type: ignore[arg-type]
                else:
                    assert is_fundus_data_sequence(fundus_data) and len(fundus_data) == map_np.shape[0], (
                        "Mismatch between input and output."
                    )
                    for f, m in zip(fundus_data, map_np, strict=True):
                        if callable(self._fundus_data_field):
                            self._fundus_data_field(f, m)  # type: ignore[arg-type]
                        else:
                            f.update(**{self._fundus_data_field: m}, inplace=True)  # type: ignore[arg-type]

            return map_np
        else:
            return map

    @property
    def func_name(self) -> str | None:
        return getattr(self.__wrapped__, "__name__", None)


def fundus_inference[**P](
    fundus_data_field: FundusData.Fields | FundusDataFieldUpdater,
) -> Callable[[FundusInferenceInternalFunc[P]], FundusInferenceFunc[P]]:
    def decorator(infer_func: FundusInferenceInternalFunc[P]) -> FundusInferenceFunc[P]:
        return GenericFundusInference(infer_func, fundus_data_field)

    return decorator


def is_path_sequence(x: object) -> TypeGuard[Sequence[PathLike]]:
    """Check if the input is a sequence of PathLike."""
    if not isinstance(x, str) and isinstance(x, Sequence):
        return all(isinstance(p, PathLike) for p in x)
    return False


def is_fundus_data_sequence(x: object) -> TypeGuard[Sequence[FundusData]]:
    """Check if the input is a sequence of FundusData."""
    if isinstance(x, Sequence):
        return all(isinstance(f, FundusData) for f in x)
    return False


def check_fundus_data(f: FundusData) -> None:
    """Check if the input is a FundusData."""
    assert f.mutable, (
        "FundusData passed to the inference function must be mutable. Please call ``fundus.mutable_copy()`` first."
    )
    assert f.image is not None, "FundusData passed to the inference function must contains a fundus image."
