from __future__ import annotations

import typing
from copy import copy
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, Optional, Self, Tuple, TypeAlias, TypedDict, overload, Generator

import numpy as np
import numpy.typing as npt

from .transform import AffineTransform, Bool2DArray, ResizeTranslation, Transform
from .utils.data_io import most_common_image_ext
from .utils.geometric import Point, Rect
from .utils.image import (
    crop_pad_center,
    find_centroid,
    label_map_to_rgb,
    read_image,
    resize,
    write_image,
    read_image_shape,
)
from .utils.math import fit_circle
from .utils.safe_import import is_torch_tensor
from .utils.typing import Bool1DArray, PointArrayLike, as_points

if TYPE_CHECKING:
    import torch

    from .utils.typing import PathLike

    type ImageSource = torch.Tensor | npt.NDArray | PathLike  # | Sequence[PathLike]

ABSENT = Point(float("nan"), float("nan"))


class ReshapeMethod(str, Enum):
    """Enum class for the shape resolution method."""

    #: The image is resized to the given shape.
    RESIZE = "resize"

    #: The image is cropped or padded to the given shape.
    CROP = "crop"

    #: Raise an error if the shape is not the same as the given shape.
    RAISE = "raise"

    @classmethod
    def parse(cls, value: ReshapeMethods) -> Self:
        """Parse a string to a ReshapeMethod enum."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid reshape method: {value}.\nValid options are: {', '.join(m.value for m in ReshapeMethod)}"
            ) from e


ReshapeMethods: TypeAlias = Literal["resize", "crop", "raise"] | ReshapeMethod


class AVLabel(IntEnum):
    """Enum class for the labels of the arteries and veins in the fundus image."""

    #: Background
    BKG = 0

    #: Artery
    ART = 1

    #: Vein
    VEI = 2

    #: Both
    BOTH = 3

    #: Unknown
    UNK = 4

    @classmethod
    def select_label(
        cls, artery: Optional[bool] = None, vein: Optional[bool] = None, unknown: Optional[bool] = None
    ) -> Tuple[Self, ...]:
        """Select the label corresponding to the given conditions.

        Parameters
        ----------
        artery : bool, optional
            If True, the label is an artery.
        vein : bool, optional
            If True, the label is a vein.
        unkown : bool, optional
            If True, the label is unknown.

        Returns
        -------
        Tuple[AVLabel, ...]
            The labels corresponding to the given conditions.
        """
        if all(not v for v in (artery, vein, unknown)):
            return (AVLabel.ART, AVLabel.VEI, AVLabel.BOTH, AVLabel.UNK)  # type: ignore
        labels = set()
        if artery is True:
            labels.update((AVLabel.ART, AVLabel.BOTH))
        if vein is True:
            labels.update((AVLabel.VEI, AVLabel.BOTH))
        if unknown is True:
            labels.add(AVLabel.UNK)
        return tuple(labels)


class FundusData:
    def __init__(
        self,
        image=None,
        *,
        roi_mask=None,
        roi_specs: Optional[FundusROISpecs] = None,
        vessels=None,
        av=None,
        od=None,
        od_center=None,
        od_size=None,
        macula=None,
        macula_center=None,
        scale: Optional[float] = None,
        name: Optional[str] = None,
        check_validity: bool = True,
        shape: Optional[Tuple[int, int] | ImageSource] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        immutable: bool = False,
    ):
        if name is None:
            if isinstance(image, (str, Path)):
                name = Path(image).stem
            elif isinstance(vessels, (str, Path)):
                name = Path(vessels).stem
            elif isinstance(av, (str, Path)):
                name = Path(av).stem
            elif isinstance(od, (str, Path)):
                name = Path(od).stem
            elif isinstance(macula, (str, Path)):
                name = Path(macula).stem
        self._name = name

        if isinstance(shape, (Path, str)):
            shape_ = read_image_shape(shape)
        elif isinstance(shape, np.ndarray) or is_torch_tensor(shape):
            shape_ = tuple(shape.shape[-2:])  # type: ignore[assignment]
        else:
            shape_: Optional[Tuple[int, int]] = shape  # type: ignore
        assert shape_ is None or (isinstance(shape_, tuple) and len(shape_) == 2), "Shape must be a tuple of (H, W)."

        if image is not None:
            if check_validity:
                image = self.load_fundus_image(image, shape_, reshape_method=reshape_method)
            else:
                assert isinstance(image, np.ndarray), "The image must be a numpy array."
            self._image = image
            if shape_ is None:
                shape_ = self._image.shape[-2:]  # type: ignore[assignment]
        else:
            self._image = None

        assert roi_specs is None or isinstance(roi_specs, FundusROISpecs), (
            "The roi_specs must be an instance of ROISpecs."
        )
        self._roi_specs = roi_specs

        if roi_mask is not None:
            if check_validity:
                roi_mask = self.load_fundus_mask(
                    roi_mask, from_fundus=False, target_shape=shape_, reshape_method=reshape_method
                )
            else:
                assert isinstance(roi_mask, np.ndarray), "The fundus mask must be a numpy array."
            self._roi_mask = roi_mask
            if shape_ is None:
                shape_ = roi_mask.shape[-2:]  # type: ignore[assignment]
        else:
            self._roi_mask = None

        if vessels is not None:
            if check_validity:
                vessels = self.load_vessels(vessels, target_shape=shape_, reshape_method=reshape_method)
            else:
                assert isinstance(vessels, np.ndarray), "The vessels segmentation must be a numpy array."
            self._vessels = vessels
            if shape_ is None:
                shape_ = vessels.shape[-2:]  # type: ignore[assignment]
        else:
            self._vessels = None
        self._bin_vessels = None

        if av is not None:
            if check_validity:
                av = self.load_av(av, target_shape=shape_, reshape_method=reshape_method)
            else:
                assert isinstance(av, np.ndarray), "The artery/vein segmentation must be a numpy array."
            self._av = av
            if shape_ is None:
                shape_ = av.shape[-2:]  # type: ignore[assignment]
        else:
            self._av = None

        if od is not None:
            if check_validity:
                self._od, self._od_center, self._od_size = self.load_od_macula(
                    od, shape_, reshape_method=reshape_method, fit_ellipse=True
                )  # type: ignore
                if shape_ is None:
                    shape_ = self._od.shape[-2:]  # type: ignore[assignment]
            else:
                assert isinstance(od, np.ndarray), "The optic disc segmentation must be a numpy array."
                self._od, self._od_center, self._od_size = od, od_center, od_size
        else:
            self._od = None
            self._od_center = None if od_center is None else Point.parse(od_center)
            self._od_size = None if od_size is None else Point.parse(od_size)

        if macula is not None:
            if check_validity:
                self._macula, self._macula_center = self.load_od_macula(macula, shape_, reshape_method=reshape_method)
                if shape_ is None:
                    shape_ = self._macula.shape[-2:]  # type: ignore[assignment]
            else:
                assert isinstance(macula, np.ndarray), "The macula segmentation must be a numpy array."
                self._macula, self._macula_center = macula, macula_center
        else:
            self._macula = None
            self._macula_center = None if macula_center is None else Point.parse(macula_center)

        if shape_ is None:
            raise ValueError("No data was provided to initialize the FundusData.")
        self._shape = shape_
        self._scale = scale

        self._immutable = immutable

    @classmethod
    def empty_like(cls, data: ImageSource, immutable: bool = False) -> Self:
        """Convert the given data to a fundus image format (numpy array of shape (3, H, W) and type float32)."""
        name = None
        if isinstance(data, str):
            name = Path(data).stem
        elif isinstance(data, Path):
            name = data.stem
        return cls(shape=data, name=name, immutable=immutable)

    type Fields = Literal["image", "fundus_mask", "vessels", "av", "od", "macula"]

    def update(
        self,
        image: ImageSource | EllipsisType = ...,
        roi_mask: ImageSource | EllipsisType = ...,
        vessels: ImageSource | EllipsisType = ...,
        av: ImageSource | EllipsisType = ...,
        od: ImageSource | EllipsisType = ...,
        od_center: Point | EllipsisType = ...,
        od_size: Point | EllipsisType = ...,
        macula: ImageSource | EllipsisType = ...,
        macula_center: Point | EllipsisType = ...,
        name: str | EllipsisType = ...,
        *,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        roi: Optional[Rect] = None,
        inplace: bool = False,
    ) -> Self:
        """Return a copy of this FundusData with updated fields. The fields that are not specified are kept unchanged.
           The dimensions of the updated fields are checked to match the dimensions of the current FundusData and `reshape_method` is used to resolve any mismatch.

        Parameters
        ----------
        image : ImageSource | EllipsisType, optional
            If specified, update the fundus image.
        roi_mask : ImageSource | EllipsisType, optional
            If specified, update the fundus mask.
        vessels : ImageSource | EllipsisType, optional
            If specified, update the vessels image.
        av : ImageSource | EllipsisType, optional
            If specified, update the artery/vein segmentation image.
        od : ImageSource | EllipsisType, optional
            If specified, update the optic disc image.
        od_center : Point | EllipsisType, optional
            If specified, update the optic disc center.
        od_size : Point | EllipsisType, optional
            If specified, update the optic disc size.
        macula : ImageSource | EllipsisType, optional
            If specified, update the macula image.
        macula_center : Point | EllipsisType, optional
            If specified, update the macula center.
        name : str | EllipsisType, optional
            If specified, update the name of the FundusData.
        reshape_method : ReshapeMethods, optional
            The method to use for reshaping the images. Options are:
            - "resize": resize the image to the target shape.
            - "crop": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.
        crop_pad : Optional[Rect], optional
            If specified, crop or pad the images to the given rectangle before resizing.
        inplace : bool, optional
            If True, update this FundusData in place and return it. Otherwise, return a new FundusData with the updated fields.

        Returns
        -------
        Self
            A copy of this FundusData with the updated fields.

        Raises
        ------
        RuntimeError
            If inplace is True and this FundusData is immutable.
        """  # noqa: E501
        if inplace:
            if self._immutable:
                raise RuntimeError("This FundusData instance is immutable.")
            other = self
        else:
            other = copy(self)

        class ShapeOpts(TypedDict):
            crop_pad: Optional[Rect]
            reshape_method: ReshapeMethods
            target_shape: Tuple[int, int]

        shape_opts = ShapeOpts(
            crop_pad=crop_pad, reshape_method=reshape_method, target_shape=self.shape if roi is None else roi.shape
        )

        def roi_crop(img):
            if roi is None:
                return img
            return Rect.from_size(self.shape).crop_pad_image(img, copy=False)

        if image is not ...:
            other._image = roi_crop(other.load_fundus_image(image, **shape_opts))
        if roi_mask is not ...:
            other._roi_mask = roi_crop(other.load_fundus_mask(roi_mask, **shape_opts))
            if other._roi_specs is not None:
                other._roi_specs = None  # Invalidate the roi_specs
        if vessels is not ...:
            other._vessels = roi_crop(other.load_vessels(vessels, **shape_opts))
        if av is not ...:
            other._av = roi_crop(other.load_av(av, **shape_opts))
        if od is not ...:
            seg_map, center, other._od_size = other.load_od_macula(od, **shape_opts, fit_ellipse=True)
            other._od = roi_crop(seg_map)
            other._od_center = center if roi is None else center + roi.top_left
        if macula is not ...:
            seg_map, center = other.load_od_macula(macula, **shape_opts)
            other._macula = roi_crop(seg_map)
            other._macula_center = center if roi is None else center + roi.top_left

        if od_center is not ...:
            other._od_center = Point.parse(od_center)
        if od_size is not ...:
            other._od_size = Point.parse(od_size)
        if macula_center is not ...:
            other._macula_center = Point.parse(macula_center)
        if name is not ...:
            other._name = name
        return other

    def to_immutable(self) -> Self:
        """Return an immutable copy of this FundusData."""
        if self._immutable:
            return self
        other = copy(self)
        other._set_immutable_flag(True)
        return other

    def _set_immutable_flag(self, immutable: bool) -> None:
        """Set the immutable flag to True."""
        self._immutable = immutable
        if self._image is not None:
            self._image.setflags(write=not immutable)
        if self._roi_mask is not None:
            self._roi_mask.setflags(write=not immutable)
        if self._vessels is not None:
            self._vessels.setflags(write=not immutable)
        if self._av is not None:
            self._av.setflags(write=not immutable)
        if self._od is not None:
            self._od.setflags(write=not immutable)
        if self._macula is not None:
            self._macula.setflags(write=not immutable)

    def copy(self, mutable: bool | None = None) -> Self:
        """Return a mutable copy of this FundusData.
        Parameters
        ----------
        mutable : bool | None, optional
            If True, the returned FundusData is mutable.
            If False, the returned FundusData is immutable.
            If None, the returned FundusData has the opposite mutability of this FundusData.
            by default None.

        Returns
        -------
        FundusData
            A copy of this FundusData.

        """
        if mutable is False and self._immutable:
            return self
        other = copy(self)
        if mutable is None:
            mutable = not self._immutable
        other._set_immutable_flag(not mutable)
        return other

    @classmethod
    def from_folders(
        cls,
        image: PathLike | EllipsisType = ...,
        roi_mask: PathLike | EllipsisType = ...,
        vessels: PathLike | EllipsisType = ...,
        av: PathLike | EllipsisType = ...,
        od: PathLike | EllipsisType = ...,
        macula: PathLike | EllipsisType = ...,
    ) -> Generator[Self]:
        """Load fundus data from folders containing the different modalities. Each folder must contain files with the same name.

        Parameters
        ----------
        image : PathLike | Ellipsis, optional
            Folder containing the fundus images.

        roi_mask : PathLike | Ellipsis, optional
            Folder containing the fundus masks.

        vessels : PathLike | Ellipsis, optional
            Folder containing the vessels segmentations.

        av : PathLike | Ellipsis, optional
            Folder containing the artery/vein segmentations.

        od : PathLike | Ellipsis, optional
            Folder containing the optic disc segmentations.

        macula : PathLike | Ellipsis, optional
            Folder containing the macula segmentations.

        Yields
        ------
        Iterator[Self]
            An iterator over the loaded FundusData instances.
        """

        def list_files(folder: PathLike) -> dict[str, Path]:
            return {Path(p).stem: p for p in Path(folder).glob(f"*{most_common_image_ext(folder)}")}

        paths = {
            modality: list_files(folder)
            for modality, folder in dict(
                image=image, roi_mask=roi_mask, vessels=vessels, av=av, od=od, macula=macula
            ).items()
            if folder is not ...
        }

        common_names = set.intersection(*(set(p.keys()) for p in paths.values()))
        for name in common_names:
            yield cls(**{modality: paths[modality][name] for modality in paths.keys()} | {"name": name})

    def remove_od_from_vessels(self, shrink_factor: float = 0, *, mask_roi: bool = True, inplace=False) -> Self:
        updated_data = {}
        mask = self.od

        if shrink_factor > 0:
            from .utils.safe_import import cv2

            mask = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) > shrink_factor * self.od_diameter
        if mask_roi:
            mask |= ~self.roi_mask

        if self._vessels is not None:
            vessels = self._vessels.copy()
            vessels[mask] = False
            updated_data["vessels"] = vessels
        if self._av is not None:
            av = self._av.copy()
            av[mask] = AVLabel.BKG
            updated_data["av"] = av
        return self.update(**updated_data, inplace=inplace)

    def apply_roi_mask(self, inplace=False) -> Self:
        mask = ~self.roi_mask
        mutable = not self._immutable

        data = self if inplace else self.copy(mutable=True)
        if data._image is not None:
            data._image[:, mask] = 0
        if data._vessels is not None:
            data._vessels[mask] = False
        if data._av is not None:
            data._av[mask] = AVLabel.BKG
        if data._od is not None:
            data._od[mask] = False
        if data._macula is not None:
            data._macula[mask] = False

        if mutable:
            data._set_immutable_flag(False)
        return data

    ####################################################################################################################
    #    === CHECK METHODS ===
    ####################################################################################################################
    @classmethod
    def load_fundus_image(
        cls,
        fundus: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
    ) -> npt.NDArray[np.float32]:
        """Load a fundus image from various sources.
        The image is converted to a numpy array of type float32, with channel first (3, H, W) and values in [0, 1].

        Parameters
        ----------
        fundus : ImageSource
            The fundus image. It must be one of:
            - A path to an image file containing the fundus image.
            - A numpy array (or torch Tensor) of shape (H, W, 3) or (3, H, W) containing the fundus image.
        target_shape : Optional[Tuple[int, int]], optional
            The target shape of the image, by default None
        reshape_method : ReshapeMethods, optional
            The method to use to resolve shape mismatches between the image and the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.
        crop_pad : Optional[Rect], optional
            If specified, crop or pad the image to the given rectangle before resizing, by default None.

        Returns
        -------
        npt.NDArray[np.float32]
            The loaded fundus image as a numpy array of shape (3, H, W) and type float32.

        Raises
        ------
        TypeError
            If the fundus image is not a path, a numpy array or a torch tensor.
        ValueError
            If the fundus image is not a color image.
        """  # noqa: E501
        # --- Load the image ---
        if isinstance(fundus, (str, Path)):
            fundus_ = read_image(fundus)
        elif is_torch_tensor(fundus):
            fundus_ = fundus.numpy(force=True)
        elif isinstance(fundus, np.ndarray):
            fundus_ = fundus
        else:
            raise TypeError("The fundus image must be a path, a numpy array or a torch tensor.")

        # --- Check image ---
        assert fundus_.ndim == 3, "The image must be a color image."
        if fundus_.shape[0] != 3 and fundus_.shape[2] == 3:
            fundus_ = fundus_.transpose(2, 0, 1)  # HWC -> CHW
        assert fundus_.shape[0] == 3, "The image must be a RGB color image."

        # --- Format data ---
        if fundus_.dtype != np.float32:
            fundus_ = fundus_.astype(np.float32)

        # --- Extract ROI ---
        if crop_pad is not None:
            fundus_ = crop_pad.crop_pad_image(fundus_, copy=False)

        # -- Resize the image ---
        if target_shape is not None and target_shape != fundus_.shape[-2:]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    t = ResizeTranslation((target_shape[0] / fundus_.shape[-2], target_shape[1] / fundus_.shape[-1]))
                    fundus_ = t.warp(fundus_, channel_last=False, warped_domain=Rect.from_size(target_shape))
                case ReshapeMethod.CROP:
                    fundus_ = crop_pad_center(fundus_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus_ image shape {fundus_.shape} differs from the target shape {target_shape}."
                    )
        return fundus_  # type: ignore

    @classmethod
    def load_fundus_mask(
        cls,
        fundus_mask: ImageSource,
        from_fundus: bool = False,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        name: Optional[str] = None,
    ) -> npt.NDArray[np.bool_]:
        """Load a fundus mask from various sources.

        Parameters
        ----------
        fundus_mask : ImageSource
            The fundus mask or fundus image. It must be one of:
            - A path to an image file containing the fundus mask or fundus image.
            - A numpy array (or torch Tensor) of shape (H, W) containing the fundus mask or of shape (H, W, 3) or (3, H, W) containing the fundus image.
        from_fundus : bool, optional
            If True, compute the fundus mask from the fundus image, by default False.
        target_shape : Optional[Tuple[int, int]], optional
            The target shape of the image, by default None.
        reshape_method : ReshapeMethods, optional
            The method to use to resolve shape mismatches between the image and the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.
        crop_pad : Optional[Rect], optional
            If specified, crop or pad the image to the given rectangle before resizing, by default None.
        """
        if from_fundus:
            # --- Compute from fundus mask ---
            from .utils.fundus import fundus_ROI

            fundus = cls.load_fundus_image(fundus_mask, target_shape, reshape_method=reshape_method, crop_pad=crop_pad)
            mask_ = fundus_ROI(fundus, check=True if name is None else name)  # Compute the fundus mask
        else:
            # --- Load the image ---
            if isinstance(fundus_mask, (str, Path)):
                mask_ = read_image(fundus_mask, cast_to_float=False)
            elif is_torch_tensor(fundus_mask):
                mask_ = fundus_mask.numpy(force=True)
            elif isinstance(fundus_mask, np.ndarray):
                mask_ = fundus_mask
            else:
                raise TypeError("The fundus mask must be a path, a numpy array or a torch tensor.")

            # --- Check image ---
            if mask_.ndim != 2:
                raise ValueError("Invalid fundus mask")

            if crop_pad is not None:
                mask_ = crop_pad.crop_pad_image(mask_, copy=False)

            if mask_.dtype != bool:
                mask_ = mask_ > 127 if mask_.dtype == np.uint8 else mask_ > 0.5

        # -- Resize the image ---
        if target_shape is not None and target_shape != mask_.shape[-2:]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    shape = mask_.shape[-2:]
                    t = ResizeTranslation((target_shape[0] / shape[-2], target_shape[1] / shape[-1]))
                    mask_ = t.warp(mask_, channel_last=False, warped_domain=Rect.from_size(target_shape))
                case ReshapeMethod.CROP:
                    mask_ = crop_pad_center(mask_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {mask_.shape} differs from the target shape {target_shape}."
                    )
        return mask_  # type: ignore

    @classmethod
    def load_vessels(
        cls,
        vessels: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
    ) -> npt.NDArray[np.bool_]:
        """Load a vessels segmentation.

        Parameters
        ----------
        vessels : ImageSource
            The vessels segmentation. It must be one of:
            - A path to an image file containing the vessels segmentation.
            - A numpy array (or torch Tensor) of shape (H, W) containing the vessels segmentation.
        target_shape : Optional[Tuple[int, int]], optional
            The target shape of the image, by default None.
        reshape_method : ReshapeMethods, optional
            The method to use to resolve shape mismatches between the image and the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.
        crop_pad : Optional[Rect], optional
            If specified, crop or pad the image to the given rectangle before resizing, by default None.
        Returns
        -------
        npt.NDArray[np.bool_]
            The loaded vessels segmentation as a binary numpy array.
        """  # noqa: E501
        # --- Load the image ---
        if isinstance(vessels, (str, Path)):
            vessels_ = read_image(vessels, cast_to_float=False)
        elif is_torch_tensor(vessels):
            vessels_ = vessels.numpy(force=True)
        elif isinstance(vessels, np.ndarray):
            vessels_ = vessels
        else:
            raise TypeError("The vessels segmentation must be a path, a numpy array or a torch tensor.")

        # --- Check image ---
        if vessels_.ndim == 3:
            if vessels_.shape[0] not in (1, 3) and vessels_.shape[2] in (1, 3):
                vessels_ = vessels_.transpose(2, 0, 1)  # HWC -> CHW
            # Check if the image is a binary image
            MAX = 255 if vessels_.dtype == np.uint8 else 1
            vessels_ = np.any(vessels_ > MAX / 2, axis=0)
        elif vessels_.ndim != 2:
            raise ValueError("Invalid fundus mask")

        if vessels_.dtype != bool:
            vessels_ = vessels_ > 127 if vessels_.dtype == np.uint8 else vessels_ > 0.5

        # --- Crop/PAD ---
        if crop_pad is not None:
            vessels_ = crop_pad.crop_pad_image(vessels_, copy=False)

        # -- Resize the image ---
        if target_shape is not None and target_shape != vessels_.shape[-2:]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    t = ResizeTranslation(s=(target_shape[0] / vessels_.shape[0], target_shape[1] / vessels_.shape[1]))
                    vessels_ = cls.warp_vessels(vessels_, t, dst_domain=Rect.from_size(target_shape))
                case ReshapeMethod.CROP:
                    vessels_ = crop_pad_center(vessels_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {vessels_.shape} differs from the target shape {target_shape}."
                    )
        return vessels_  # type: ignore

    @classmethod
    def load_av(
        cls,
        av: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        *,
        invert_av: bool = False,
        ensure_valid_av: bool = False,
    ) -> npt.NDArray[np.uint8]:
        """Load the arteries and veins map.

        Parameters
        ----------
        av :
            The arteries and veins map. It must be one of:

            - A path to an image file containing the arteries and veins map. Its colors are interpreted as: blue (vein), red (artery), magenta (both), green (unknown).
            - A numpy array (or torch Tensor) of shape (H, W, 3) of such image.
            - A numpy array of shape (H, W) containing the arteries and veins map using the `AVLabel` convention.

        target_shape : Tuple[int, int], optional
            The target shape of the image, by default None.

        reshape_method : ShapeResolve, optional
            The method used when the shape of the image is different from the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.

        crop_pad : Optional[Rect], optional
            If specified, crop or pad the image to the given rectangle before resizing, by default None.

        invert_av : bool, optional
            If True, the red channel is interpreted as the vein and the blue channel as the artery.

        ensure_valid_av : bool, optional
            If True, ensure that the loaded arteries and veins map contains at least one artery and one vein and raise an error otherwise.

        Returns
        -------
        npt.NDArray[np.uint8]
            An array of shape (H, W) containing the arteries and veins map.
            The values are defined in the `AVLabel` enum:

            - 0: background
            - 1: artery
            - 2: vein
            - 3: both
            - 4: unknown

        """  # noqa: E501
        # --- Load the image ---
        if isinstance(av, (str, Path)):
            av_ = read_image(av, cast_to_float=False)
        elif is_torch_tensor(av):
            av_ = av.numpy(force=True)
        elif isinstance(av, np.ndarray):
            av_ = av
        else:
            raise TypeError("The vessels segmentation must be a path, a numpy array or a torch tensor.")

        # --- Interpret image ---
        assert isinstance(av_, np.ndarray), "The vessels map must be a numpy array."
        if av_.ndim == 3:
            if av_.shape[0] not in (1, 3) and av_.shape[2] in (1, 3):
                av_ = av_.transpose(2, 0, 1)  # HWC -> CHW

            # Check if the image is a binary image
            MAX = 255 if av_.dtype == np.uint8 else 1
            av_ = av_ > MAX / 2

            if np.all(np.all(av_, axis=2)):
                av_ = (av_[0] * AVLabel.UNK).astype(np.uint8)
            else:
                av_map = np.zeros(av_.shape[-2:], dtype=np.uint8)
                if invert_av:
                    av_ = av_[::-1]  # Invert the channels: red (artery) and blue (vein)
                av_map[av_[0]] = AVLabel.ART
                av_map[av_[2]] = AVLabel.VEI
                av_map[av_[1]] = AVLabel.BOTH
                av_map[av_[2] & av_[0]] = AVLabel.BOTH
                av_ = av_map
        elif av_.ndim == 2:
            if av_.dtype == bool:
                if ensure_valid_av:
                    raise ValueError("The provided map is a binary image, which is not a valid arteries/veins map.")
                av_ = (av_ * AVLabel.UNK).astype(np.uint8)
            elif np.issubdtype(av_.dtype, np.integer):
                assert av_.min() >= 0 and av_.max() <= AVLabel.UNK, "Invalid vessels map"
                av_ = av_.astype(np.uint8)
            else:
                raise ValueError("The vessels map must be a binary image or a label map using the AVLabel convention.")
        if ensure_valid_av:
            if not np.all(np.isin([AVLabel.BKG, AVLabel.ART, AVLabel.VEI], av_)):
                raise ValueError("The provided arteries/veins map does not contain at least one artery and one vein.")

        # --- Crop/PAD ---
        if crop_pad is not None:
            av_ = crop_pad.crop_pad_image(av_, copy=False)

        # --- Resize the image ---
        if target_shape is not None and av_.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    t = ResizeTranslation(s=(target_shape[0] / av_.shape[0], target_shape[1] / av_.shape[1]))
                    av_ = cls.warp_vessels(av_, t, dst_domain=Rect.from_size(target_shape))
                case ReshapeMethod.CROP:
                    av_ = crop_pad_center(av_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The artery/vein maps shape {av_.shape} differs from the fundus image shape {target_shape}."
                    )

        return av_  # type: ignore

    @overload
    @classmethod
    def load_od_macula(
        cls,
        seg: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        *,
        fit_ellipse: Literal[False] = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point]: ...
    @overload
    @classmethod
    def load_od_macula(
        cls,
        seg: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        *,
        fit_ellipse: Literal[True],
    ) -> Tuple[npt.NDArray[np.bool_], Point, Point]: ...
    @classmethod
    def load_od_macula(
        cls,
        seg: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        crop_pad: Optional[Rect] = None,
        *,
        fit_ellipse: bool = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point] | Tuple[npt.NDArray[np.bool_], Point, Point]:
        """Load an optic disc or macula segmentation.
        Parameters
        ----------
        seg : ImageSource
            The optic disc or macula segmentation. It must be one of:
            - A path to an image file containing the segmentation.
            - A numpy array (or torch Tensor) of shape (H, W) containing the segmentation.
        target_shape : Optional[Tuple[int, int]], optional
            The target shape of the image, by default None.
        reshape_method : ReshapeMethods, optional
            The method to use to resolve shape mismatches between the image and the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.
        crop_pad : Optional[Rect], optional
            If specified, crop or pad the image to the given rectangle before resizing, by default None.
        fit_ellipse : bool, optional
            If True, fit an ellipse to the optic disc segmentation and return its size as well.
        Returns
        -------
        binary_map : npt.NDArray[np.bool_]
            The binary segmentation of the optic disc or macula.
        center : Point
            The center of the optic disc or macula.
        size : Point, optional
            If `fit_ellipse` is True, the size of the fitted ellipse (major_axis, minor_axis).
        """  # noqa: E501

        # --- Load the image ---
        if isinstance(seg, (str, Path)):
            seg_ = read_image(seg, cast_to_float=False)
        elif is_torch_tensor(seg):
            seg_ = seg.numpy(force=True)
        elif isinstance(seg, np.ndarray):
            seg_ = seg
        else:
            raise TypeError("The optic disc segmentation must be a path, a numpy array or a torch tensor.")

        # --- Interpret image ---
        if seg_.ndim == 3:
            if seg_.shape[0] not in (1, 3) and seg_.shape[2] in (1, 3):
                seg_ = seg_.transpose(2, 0, 1)  # HWC -> CHW
            # Check if the image is a binary image
            MAX = 255 if seg_.dtype == np.uint8 else 1
            seg_ = np.any(seg_ > MAX / 2, axis=0)
        assert seg_.ndim == 2, "The optic disc map must be a grayscale image."

        if seg_.dtype != bool:
            MAX = 255 if seg_.dtype == np.uint8 else 1
            seg_ = seg_ > MAX / 2

        # --- Crop/PAD ---
        if crop_pad is not None:
            seg_ = crop_pad.crop_pad_image(seg_, copy=False)

        # --- Resize the image ---
        if target_shape is not None and seg_.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    seg_ = resize(seg_, target_shape, interpolation=True)
                case ReshapeMethod.CROP:
                    seg_ = crop_pad_center(seg_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The optic disc / macula maps shape {seg_.shape} differs from the fundus image shape {target_shape}."
                    )

        # --- Find the centroid ---
        return find_centroid(seg_.astype(np.bool_), fit_ellipse=fit_ellipse)

    def infer_scale(self, method: Literal["od_macula", "od_diameter", "width"] | None = None) -> float:
        """Infer the scale of the fundus image in μm per pixel using optic disc, macula or vessels segmentation.

        - "od_macula": use the distance between the optic disc and macula centers, assuming an average distance of 4500 μm.
        - "od_diameter": use the diameter of the optic disc, assuming an average diameter of 1.8 mm.
        - "width": use the width of the fundus image, assuming an average width of 15 mm (45°).

        Returns
        -------
        float
            The inferred scale in microns per pixel.
        """
        if method is None:
            if self.has_od and self.has_macula and self.od_center is not None and self.macula_center is not None:
                method = "od_macula"
            elif self.has_od and self.od_diameter is not None:
                method = "od_diameter"
            else:
                method = "width"

        match method:
            case "od_macula":
                if not (self.has_od and self.has_macula) or self.od_center is None or self.macula_center is None:
                    raise ValueError("Cannot infer scale from optic disc and macula centers.")
                return 4500 / self.od_center.distance(self.macula_center)
            case "od_diameter":
                if not self.has_od or self.od_diameter is None:
                    raise ValueError("Cannot infer scale from optic disc diameter.")
                return 1800 / self.od_diameter
            case "width":
                return 15000 / (self.roi_specs.radius * 2)  # Assuming a fundus width of 14 mm (45°)
            case _:
                raise ValueError(f"Unknown scale inference method: {method}")

    ####################################################################################################################
    #    === PROPERTY ACCESSORS ===
    ####################################################################################################################
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape  # type: ignore

    @property
    def has_name(self) -> bool:
        return self._name is not None

    @property
    def has_image(self) -> bool:
        return self._image is not None

    @property
    def image(self) -> npt.NDArray[np.float32]:
        """The fundus image with shape (3, H, W) and in RGB format. Values are float32 in the range [0, 1]."""
        if self._image is None:
            raise AttributeError("The fundus image was not provided.")
        return self._image

    @property
    def has_roi_mask(self) -> bool:
        return self._roi_mask is not None

    @property
    def roi_mask(self) -> npt.NDArray[np.bool_]:
        if self._roi_mask is None:
            if self._roi_specs is not None:
                self._roi_mask = self._roi_specs.to_mask(self.shape)
            elif self._image is not None:
                self._roi_mask = self.load_fundus_mask(self._image, from_fundus=True, target_shape=self.shape)
            else:
                raise AttributeError("The fundus ROI mask was not provided.")
            if self.immutable:
                self._roi_mask.setflags(write=False)
        return self._roi_mask

    @property
    def roi_specs(self) -> FundusROISpecs:
        if self._roi_specs is None:
            if self._roi_mask is None and self._image is None:
                raise AttributeError("The fundus ROI mask was not provided.")
            self._roi_specs = FundusROISpecs.from_mask(self.roi_mask)
        return self._roi_specs

    @property
    def has_vessels(self) -> bool:
        return self._vessels is not None

    @property
    def vessels(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the vessels with shape (H, W).

        Raises
        ------
        AttributeError
            If the vessels segmentation was not provided.
        """
        if self._vessels is not None:
            return self._vessels
        if self._av is not None:
            return self._av > 0
        raise AttributeError(
            "The vessels segmentation was not provided.\n"
            "See `fundus_vessels_toolkit.segment_vessels()` for automatic vessels segmentation models."
        )

    @property
    def has_av(self) -> bool:
        return self._av is not None

    @property
    def av(self) -> npt.NDArray[np.uint8]:
        """The arteries and veins labels of the vessels segmentation. The labels are defined in the `AVLabel` enum.

        This method returns the arteries and veins segmentation masked by the fundus and vessel mask.

        Raises
        ------
        AttributeError
            If the vessels segmentation was not provided.
        """
        if (av := self._av) is None:
            raise AttributeError("The arteries and veins segmentation was not provided.")
        return av * self.vessels if self._vessels is not None else av

    @property
    def artery_map(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the arteries.

        Raises
        ------
        AttributeError
            If the arteries and veins segmentation was not provided.
        """
        return np.isin(self.av, (AVLabel.ART, AVLabel.BOTH))

    @property
    def vein_map(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the veins.

        Raises
        ------
        AttributeError
            If the arteries and veins segmentation was not provided.
        """
        return np.isin(self.av, (AVLabel.VEI, AVLabel.BOTH))

    @property
    def av_not_masked(self) -> npt.NDArray[np.uint8]:
        """The arteries and veins labels of the vessels segmentation. The labels are defined in the `AVLabel` enum.

        This method returns the arteries and veins segmentation without applying the fundus and vessel mask.

        Raises
        ------
        AttributeError
            If the vessels segmentation was not provided.
        """
        if (av := self._av) is None:
            raise AttributeError("The arteries and veins segmentation was not provided.")
        return av

    @property
    def has_od(self) -> bool:
        return self._od is not None

    @property
    def od(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the optic disc.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        if self._od is None:
            raise AttributeError("The optic disc segmentation was not provided.")
        return self._od

    @property
    def has_od_center(self) -> bool:
        return self._od_center is not None

    @property
    def od_center(self) -> Optional[Point]:
        """The center of the optic disc or None if the optic disc is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        if self._od_center is None:
            if self._od is None:
                raise AttributeError("The optic disc segmentation was not provided.")
            else:
                _, self._od_center, self._od_size = self.load_od_macula(self._od, self.shape, fit_ellipse=True)
        return None if self._od_center.is_nan() else self._od_center

    def od_region(
        self,
        outer_radius: Optional[float] = None,
        inner_radius: Optional[float] = None,
        *,
        multiply_by_od_diameter: bool = True,
        offset_by_od_radius: bool = True,
        mask_roi: Optional[bool] = None,
        exclude_od: bool = True,
    ) -> npt.NDArray[np.bool_]:
        """Generate the mask of a region between an inner circle and an outer circle centered on the optic disc.

        Parameters
        ----------
        outer_radius : Optional[float], optional
            Radius of the outer circle of the region.

        inner_radius : float, optional
            Radius of the inner circle of the region­.

        multiply_by_od_diameter : bool, optional
            If true (by default), the radius of the inner and outer circle are multiplied by the optic disc radius.

        offset_by_od_radius : bool, optional
            If true (by default), the radius of the inner and outer circle are offset by the optic disc radius, effectively removing the optic disc from the region if `inner_radius` is 0.

        mask_roi : bool, optional
            If true, the returned region is masked by the fundus ROI mask.
            By default, true if the fundus ROI mask is available, false otherwise.

        exclude_od : bool, optional
            If true (by default), the optic disc is excluded from the region, even if `inner_radius` is inside the optic disc.

        Returns
        -------
        npt.NDArray[bool]
            A binary mask of the same shape as the fundus image, where True values correspond to pixels in the defined region.
        """  # noqa: E501
        if mask_roi is None:
            mask_roi = self.has_roi_mask
        mask = np.ones(self.shape, bool) if not mask_roi else self.roi_mask
        if offset_by_od_radius:
            mask &= ~self.od

        od = self.od_center
        od_diameter = self.od_diameter
        assert od is not None and od_diameter is not None, (
            "Impossible to define a region around the optic: it's absent from the provided image."
        )
        H, W = self.shape
        y0, x0 = od
        dist_map = np.linalg.norm(np.stack(np.mgrid[-y0 : H - y0, -x0 : W - x0], axis=0), axis=0)  # type: ignore

        def scale_radius(radius: float) -> float:
            if multiply_by_od_diameter:
                if offset_by_od_radius:
                    radius += 0.5
                radius *= self.od_diameter / 2
            elif offset_by_od_radius:
                radius += self.od_diameter / 2
            return radius

        if inner_radius is not None and (inner_radius := scale_radius(inner_radius)) > 0:
            mask &= dist_map >= inner_radius
        if outer_radius is not None:
            mask &= dist_map <= scale_radius(outer_radius)
        return mask

    @property
    def has_od_size(self) -> bool:
        return self._od_size is not None and self._od_size is not ABSENT

    @property
    def od_size(self) -> Point:
        """The size (width, height) of the optic disc or None if the optic disc is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        if self._od_size is None:
            if self._od is None:
                raise AttributeError("The optic disc segmentation was not provided.")
            else:
                _, self._od_center, self._od_size = self.load_od_macula(self._od, self.shape, fit_ellipse=True)
        if self._od_size is ABSENT:
            raise AttributeError("The optic disc is not visible in this fundus.")
        return self._od_size

    @property
    def has_od_diameter(self) -> bool:
        return self.has_od_size

    @property
    def od_diameter(self) -> float:
        """The diameter of the optic disc or -1 if the optic disc is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        return -1 if self._od_size is None else self._od_size.max

    @property
    def has_macula(self) -> bool:
        return self._macula is not None

    @property
    def macula(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the macula.

        Raises
        ------
        AttributeError
            If the macula segmentation was not provided.
        """
        if self._macula is None:
            raise AttributeError("The macula segmentation was not provided.")
        return self._macula

    @property
    def has_macula_center(self) -> bool:
        return self._macula_center is not None

    @property
    def macula_center(self) -> Optional[Point]:
        """The center of the macula or None if the macula is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the macula segmentation was not provided.
        """
        if self._macula_center is None:
            if self._macula is None:
                raise AttributeError("The macula segmentation was not provided.")
            else:
                _, self._macula_center = self.load_od_macula(self._macula, self.shape)
        return None if self._macula_center.is_nan() else self._macula_center

    def inferred_macula_center(self) -> Optional[Point]:
        """The center of the macula or the center of the fundus if the optic disc is not visible."""
        if self._macula_center is not None and not self._macula_center.is_nan():
            return self._macula_center
        if self._od_center is None:
            return None

        half_weight = self.shape[1] * 0.5  # Assume 45° fundus image
        if self._od_center[1] > half_weight:
            half_weight = -half_weight
        return Point(self._od_center.y, x=self._od_center.x + half_weight)

    @property
    def scale(self) -> float:
        """The scale of the fundus image in μm per pixel.
        If not provided, estimates its value using ``FundusData.infer_scale()``."""
        if self._scale is None:
            self._scale = self.infer_scale()
        return self._scale

    @property
    def name(self) -> str:
        """The name of the fundus image.

        Raises
        ------
        AttributeError
            If the name of the fundus image was not provided.
        """
        if self._name is None:
            raise AttributeError("The name of the fundus image was not provided.")

        return self._name

    @property
    def mutable(self) -> bool:
        """Whether this FundusData instance is immutable."""
        return not self._immutable

    @property
    def immutable(self) -> bool:
        """Whether this FundusData instance is immutable."""
        return self._immutable

    ####################################################################################################################
    #    === EXPORT METHODS ===
    ####################################################################################################################
    def write_image(
        self,
        image: Optional[PathLike] = None,
        od: Optional[PathLike] = None,
        macula: Optional[PathLike] = None,
        vessels: Optional[PathLike] = None,
        av: Optional[PathLike] = None,
        *,
        on_exists: Literal["raise", "warn", "skip", "overwrite"] = "warn",
    ) -> None:
        """Save the fundus image and the provided segmentations to files.

        Parameters
        ----------
        image : PathLike
            If provided, save the fundus image at the given path. If the path correspond to a directory, the image is saved in a file named `<name>.jpg` in that directory.

        od : PathLike
            If provided, save the optic disc segmentation at the given path. If the path correspond to a directory, the segmentation is saved in a file named `<name>.png` in that directory.
        macula : PathLike
            If provided, save the macula segmentation at the given path. If the path correspond to a directory, the segmentation is saved in a file named `<name>.png` in that directory.
        vessels : PathLike
            If provided, save the vessels segmentation at the given path. If the path correspond to a directory, the segmentation is saved in a file named `<name>.png` in that directory.
        av : PathLike
            If provided, save the arteries and veins segmentation at the given path. If the path correspond to a directory, the segmentation is saved in a file named `<name>.png` in that directory.

        on_exists : Literal["raise", "warn", "skip", "overwrite"] = "warn",
            What to do if the output file already exists.
        Raises
        ------
        AttributeError
            If one of the requested data to save was not provided.
        """  # noqa: E501
        if image is not None:
            write_image(self.image, image, default_filename=f"{self.name}.jpg", on_exists=on_exists)
        if od is not None:
            write_image(self.od, od, default_filename=f"{self.name}.png", on_exists=on_exists)
        if macula is not None:
            write_image(self.macula, macula, default_filename=f"{self.name}.png", on_exists=on_exists)
        if vessels is not None:
            write_image(self.vessels, vessels, default_filename=f"{self.name}.png", on_exists=on_exists)
        if av is not None:
            av_color = label_map_to_rgb(self.av_not_masked, typing.cast(dict[int, np.ndarray], FundusData.AV_COLORS))
            write_image(av_color, av, default_filename=f"{self.name}.png", on_exists=on_exists)

    ####################################################################################################################
    #    === VISUALISATION TOOLS ===
    ####################################################################################################################
    AV_COLORS = {
        AVLabel.ART: np.array([255, 0, 0], np.uint8),  # Red
        AVLabel.VEI: np.array([0, 0, 255], np.uint8),  # Blue
        AVLabel.BOTH: np.array([255, 0, 255], np.uint8),  # Magenta
        AVLabel.UNK: np.array([0, 255, 0], np.uint8),  # Green
    }

    def draw(self, labels_opacity=0.5, *, vessels_on_top=False, view=None):
        from jppype import Mosaic, View2D

        if isinstance(view, Mosaic):
            for v in view.views:
                self.draw(labels_opacity, view=v)
            return view
        elif view is None:
            view = View2D()
        view.add_image(self._image, "fundus")

        COLORS = {
            AVLabel.ART: "coral",
            AVLabel.VEI: "cornflowerblue",
            AVLabel.BOTH: "darkorchid",
            AVLabel.UNK: "gray",
            10: "white",
            11: "teal",
        }

        if self._av is not None:
            labels = self.av.copy()
        elif self._vessels is not None:
            labels = self.vessels * np.uint8(AVLabel.UNK)
        else:
            labels = np.zeros(self.shape, dtype=np.uint8)

        if self._od is not None:
            od = self._od & ~self.vessels if vessels_on_top else self._od
            labels[od] = 10

        if self._macula is not None:
            macula = self._macula & ~self.vessels if vessels_on_top else self._macula
            labels[macula] = 11

        view.add_label(labels, "AnatomicalData", opacity=labels_opacity, colormap=COLORS)
        return view

    def show(self, labels_alpha=0.5):
        self.draw(labels_alpha).show()

    ####################################################################################################################
    #    === Utilities ===
    ####################################################################################################################
    def transform(
        self,
        transform: Transform,
        src_top_left: Point | tuple[int, int] = (0, 0),
        dst_domain: Rect | Literal["full", "same"] = "full",
    ) -> tuple[Self, Rect]:
        """Apply a geometric transformation to the fundus image and the vessels segmentation."""
        other = copy(self)
        dst_domain = transform.warped_domain(self.shape, src_top_left, dst_domain)
        other._shape = dst_domain.shape

        if self._image is not None:
            other._image, _ = transform.warp(self._image, src_top_left, dst_domain, channel_last=False)
        if self._roi_mask is not None:
            other._roi_mask, _ = transform.warp(self._roi_mask, src_top_left, dst_domain)
        if self._roi_specs is not None:
            other._roi_specs = self._roi_specs.transform(transform, src_top_left, dst_domain)
        if self._vessels is not None:
            other._vessels, _ = transform.warp(self._vessels, src_top_left, dst_domain)
        if self._av is not None:
            other._av = self.warp_vessels(self._av, transform, src_top_left, dst_domain)

        if self._od is not None:
            other._od, _ = transform.warp(self._od, src_top_left, dst_domain)
        if self._macula is not None:
            other._macula, _ = transform.warp(self._macula, src_top_left, dst_domain)
        if self._od_center not in (None, ABSENT):
            other._od_center = Point.parse(transform.transform(self._od_center + src_top_left)[0]) - dst_domain.top_left
        if self._macula_center not in (None, ABSENT):
            other._macula_center = (
                Point.parse(transform.transform(self._macula_center + src_top_left)[0]) - dst_domain.top_left
            )

        if not transform.is_identity():
            if isinstance(transform, AffineTransform):
                s = transform.scaling
                if self._od_size not in (None, ABSENT):
                    other._od_size = other._od_size * s
                if other._scale is not None:
                    other._scale = other._scale * s
            else:
                other._od_size = None
                other._scale = None

        return other, dst_domain

    @classmethod
    def warp_vessels[DTYPE: np.uint8 | np.bool_](
        cls,
        av: npt.NDArray[DTYPE],
        transform: Transform,
        src_top_left: Point | tuple[int, int] = (0, 0),
        dst_domain: Rect | Literal["full", "same"] = "full",
    ) -> npt.NDArray[DTYPE]:
        if av.dtype != np.bool_:
            art = np.isin(av, [AVLabel.ART, AVLabel.BOTH]).astype(np.float32)
            art, _ = transform.warp(art, src_top_left, dst_domain)
            vei = np.isin(av, [AVLabel.VEI, AVLabel.BOTH]).astype(np.float32)
            vei, _ = transform.warp(vei, src_top_left, dst_domain)

            av_ = np.zeros_like(art, dtype=np.uint8)
            threshold = 0.5
            if isinstance(transform, AffineTransform) and abs(transform.scaling) < 1:
                threshold *= transform.scaling  # Ensure vessels 1px wide remain visible after down-scaling
            av_[art > threshold] = np.uint8(AVLabel.ART)
            av_[vei > threshold] += np.uint8(AVLabel.VEI)

            unk = np.isin(av, [AVLabel.UNK]).astype(np.float32)
            if unk.any():
                unk, _ = transform.warp(unk, src_top_left, dst_domain)
                av_[unk > threshold] = np.uint8(AVLabel.UNK)
            return av_  # type: ignore
        else:
            av_, _ = transform.warp(av.astype(np.float32), src_top_left, dst_domain)
            threshold = 0.5
            if isinstance(transform, AffineTransform) and abs(transform.scaling) < 1:
                threshold *= transform.scaling  # Ensure vessels 1px wide remain visible after down-scaling
            return av_ > threshold  # type: ignore

    def rotate(self, angle: float) -> Self:
        """Rotate the fundus image and the vessels segmentation by a given angle.
        Parameters
        ----------
        angle : float
            The angle by which to rotate the image in degrees.
        Returns
        -------
            FundusData
                The rotated fundus image and vessels segmentation.
        """
        h, w = self.shape
        return self.transform(AffineTransform.rotate(theta=angle, center=(h / 2, w / 2)), dst_domain="same")[0]

    def resize(self, size: float | int | Tuple[int, int]) -> Self:
        """Rescale the fundus image and the vessels segmentation to a given size.

        Parameters
        ----------
        size : float | int | Tuple[int, int]
            The size to which to rescale the image. If a single value is provided, the image is rescaled to that value.
            If a tuple of two values is provided, the image is rescaled to that size.

        Returns
        -------
            FundusData
                The rescaled fundus image and vessels segmentation.
        """
        if isinstance(size, tuple):
            shape = size
        ratio = self.shape[0] / self.shape[1]
        if isinstance(size, int):
            shape = (round(size * ratio), size)
        elif isinstance(size, float):
            shape = self.shape
            shape = int(round(shape[0] * size)), int(round(shape[1] * size))
        else:
            raise

        s = Point(shape[0] / self.shape[0], shape[1] / self.shape[1])

        return self.transform(ResizeTranslation.resize(s=s), dst_domain=Rect.from_size(shape))[0]

    def crop(self, roi: Rect) -> Self:
        """Crop the fundus image and the vessels segmentation to a given rectangle.

        Parameters
        ----------
        rect : Rect
            The rectangle to which to crop the image.

        Returns
        -------
            FundusData
                The cropped fundus image and vessels segmentation.
        """

        def crop_roi(array: npt.NDArray) -> npt.NDArray:
            return roi.crop_pad_image(array, copy=False)

        updated_data = {}
        if self._image is not None:
            updated_data["image"] = crop_roi(self._image)
        if self._roi_mask is not None:
            updated_data["roi_mask"] = crop_roi(self._roi_mask)
        if self._roi_specs is not None:
            updated_data["roi_specs"] = self._roi_specs.transform(dst_domain=roi)
        if self._vessels is not None:
            updated_data["vessels"] = crop_roi(self._vessels)
        if self._av is not None:
            updated_data["av"] = crop_roi(self._av)
        if self._od is not None:
            updated_data["od"] = crop_roi(self._od)
        if self._macula is not None:
            updated_data["macula"] = crop_roi(self._macula)
        if self._od_center not in (None, ABSENT):
            updated_data["_od_center"] = self._od_center - roi.top_left
        if self._macula_center not in (None, ABSENT):
            updated_data["_macula_center"] = self._macula_center - roi.top_left
        return self.__class__(**updated_data, name=self._name, immutable=self._immutable)

    def uncrop(self, roi: Rect, dst_shape: tuple[int, int]) -> Self:
        """Uncrop the fundus image and the vessels segmentation from a given rectangle to a given shape.

        Parameters
        ----------
        roi : Rect
            The rectangle from which to uncrop the image.
        dst_shape : tuple[int, int]
            The shape to which to uncrop the image.

        Returns
        -------
            FundusData
                The uncropped fundus image and vessels segmentation.
        """

        def uncrop_roi(array: npt.NDArray) -> npt.NDArray:
            return Rect.from_size(dst_shape).crop_pad_image(array, origin=-roi.top_right, copy=False)

        updated_data = {}
        if self._image is not None:
            updated_data["image"] = uncrop_roi(self._image)
        if self._roi_mask is not None:
            updated_data["roi_mask"] = uncrop_roi(self._roi_mask)
        if self._roi_specs is not None:
            updated_data["roi_specs"] = self._roi_specs.transform(src_top_left=roi.top_left)
        if self._vessels is not None:
            updated_data["vessels"] = uncrop_roi(self._vessels)
        if self._av is not None:
            updated_data["av"] = uncrop_roi(self._av)
        if self._od is not None:
            updated_data["od"] = uncrop_roi(self._od)
        if self._macula is not None:
            updated_data["macula"] = uncrop_roi(self._macula)
        if self._od_center not in (None, ABSENT):
            updated_data["_od_center"] = self._od_center + roi.top_left
        if self._macula_center not in (None, ABSENT):
            updated_data["_macula_center"] = self._macula_center + roi.top_left
        return self.__class__(**updated_data, name=self._name, immutable=self._immutable)

    @overload
    def crop_to_roi(self, *, pad: float = 0.01, ensure_square=True, return_roi: Literal[False] = False) -> Self: ...
    @overload
    def crop_to_roi(self, *, pad: float = 0.01, ensure_square=True, return_roi: Literal[True]) -> Tuple[Self, Rect]: ...
    def crop_to_roi(
        self, *, pad: float = 0.01, ensure_square=True, return_roi: bool = False
    ) -> Self | Tuple[Self, Rect]:
        """Crop all data to the bounding box of the ROI mask.

        Parameters
        ----------
        pad : float, optional
            The padding to add to the bounding box, as a fraction of the bounding-box width, by default 0.01.

        Returns
        -------
        FundusData
            The cropped FundusData.
        """
        try:
            roi_specs = self.roi_specs
        except AttributeError:
            raise ValueError("Cannot crop to ROI: no ROI mask available.") from None

        r = roi_specs.to_rect(ensure_square=ensure_square, pad=pad)
        fundus = self.crop(r)
        return (fundus, r) if return_roi else fundus

    type ROISpecs = FundusROISpecs


@dataclass
class FundusROISpecs:
    """Specifications of a region of interest (ROI) in a fundus image."""

    center: Point
    """The center of the ROI in pixels."""

    radius: float
    """The radius of the ROI in pixels."""

    top: int | None
    """The first not-null row."""

    bottom: int | None
    """The last not-null row."""

    @classmethod
    def from_mask(cls, roi_mask: npt.NDArray[np.bool_]) -> Self:
        """Create ROISpecs from a binary ROI mask."""

        h, w = roi_mask.shape
        half_h, half_w = h / 2, w / 2
        # 1. Approximate y_min and y_max by looking at the center vertical axis
        center_col = roi_mask[:, roi_mask.shape[1] // 2]
        y_min = np.argmax(center_col)
        y_max = h - 1 - np.argmax(np.flip(center_col))

        # 2. Sample rows to estimate the center and radius of the ROI
        y = np.linspace(y_min, y_max, min(100, y_max - y_min), dtype=int)
        rows = roi_mask[y]
        x_mins = np.argmax(rows, axis=1)
        invalid_rows = ((x_mins == 0) & (~roi_mask[y, 0])) | (x_mins > half_w)
        if np.any(invalid_rows):
            y, rows, x_mins = y[~invalid_rows], rows[~invalid_rows], x_mins[~invalid_rows]
        x_maxs = w - 1 - np.argmax(np.flip(rows, axis=1), axis=1)
        invalid_rows = x_maxs < half_w
        if np.any(invalid_rows):
            y, x_mins, x_maxs = y[~invalid_rows], x_mins[~invalid_rows], x_maxs[~invalid_rows]

        yx = np.stack((np.tile(y, (2,)), np.concatenate([x_mins, x_maxs])), axis=-1)
        center, radius = fit_circle(yx)

        # 3. If y_max and y_min doesn't match the estimated radius and center check for crop band on the top and bottom
        if abs(y_min - center.y + radius) > radius * 0.1 or abs(y_max - center.y - radius) > radius * 0.1:
            x = np.linspace(half_h * 0.8, half_h * 1.2, min(20, int(half_h * 0.4)), dtype=int)
            cols = roi_mask[:, x]
            y_mins = np.argmax(cols, axis=0)
            invalid_cols = ((y_mins == 0) & (~roi_mask[0, x])) | (y_mins > half_h)
            if np.any(invalid_cols):
                x, cols, y_mins = x[~invalid_cols], cols[:, ~invalid_cols], y_mins[~invalid_cols]
            y_maxs = h - np.argmax(np.flip(cols, axis=0), axis=0)
            invalid_cols = y_maxs < half_h
            if np.any(invalid_cols):
                x, y_mins, y_maxs = x[~invalid_cols], y_mins[~invalid_cols], y_maxs[~invalid_cols]
            top = int(np.floor(np.percentile(y_mins, 95)))
            bottom = int(np.ceil(np.percentile(y_maxs, 5)))
        else:
            top = None
            bottom = None

        return cls(center, radius, top, bottom)

    def to_mask(self, shape: Tuple[int, int], disk_only: bool = False) -> Bool2DArray:
        """Convert the ROISpecs to a binary mask of the given shape."""
        from skimage.morphology import disk

        x0 = int(np.floor(self.center.x - self.radius))
        y0 = int(np.floor(self.center.y - self.radius))
        disk_mask: Bool2DArray = disk(self.radius, dtype=np.bool_)  # type: ignore
        mask = Rect.from_size(shape).crop_pad_image(disk_mask, origin=(-y0, -x0), copy=False)

        if not disk_only:
            if self.top is not None:
                mask[: self.top] = False
            if self.bottom is not None:
                mask[self.bottom :] = False
        return mask  # type: ignore

    def to_rect(self, ensure_square: bool = False, pad: float = 0) -> Rect:
        """Convert the ROISpecs to a Rect."""
        r = Rect.from_center(self.center, (self.radius * 2, self.radius * 2))

        if not ensure_square:
            top = self.top if self.top is not None else r.top
            bottom = self.bottom if self.bottom is not None else r.bottom
            r = Rect.from_points(top, r.left, bottom, r.right)

        if pad:
            r = r.pad(int(pad * r.w))
        return r

    def shrink(self, px: float) -> Self:
        """Shrink the ROI by removing the crop bands on the top and bottom if they exist."""
        top = self.top + int(px) if self.top is not None else None
        bottom = self.bottom - int(px) if self.bottom is not None else None
        return self.__class__(self.center, self.radius - px, top, bottom)

    @overload
    def is_inside(self, point: Point | tuple[int, int]) -> bool: ...
    @overload
    def is_inside(self, point: PointArrayLike) -> Bool1DArray: ...
    def is_inside(self, point: Point | tuple[int, int] | PointArrayLike) -> bool | Bool1DArray:
        """Check if a point or an array of points is inside the ROI."""
        is_point = isinstance(point, Point) or (isinstance(point, tuple) and len(point) == 2)
        points = as_points(point)
        inside = self.center.distance(points) <= self.radius
        if self.top is not None:
            inside &= points[..., 0] >= self.top
        if self.bottom is not None:
            inside &= points[..., 0] <= self.bottom
        return bool(inside[0]) if is_point else inside  # type: ignore

    def transform(
        self,
        transform: Transform | None = None,
        src_top_left: Point | tuple[int, int] = (0, 0),
        dst_domain: Rect | None = None,
    ) -> Self | None:
        """Apply a geometric transformation to the ROISpecs.
        Returns None if the ROI doesn't follow the standard circular shape with possible crop bands on the top and bottom."""
        dt = Point.parse(src_top_left)
        if dst_domain is not None:
            dt -= dst_domain.top_left

        if transform is None or transform.is_identity():
            top = int(np.floor(self.top + dt.y)) if self.top is not None else None
            bottom = int(np.ceil(self.bottom + dt.y)) if self.bottom is not None else None
            return self.__class__(self.center + dt, self.radius, top, bottom)

        transform = transform.simplify()
        if isinstance(transform, AffineTransform):
            if (t := transform.as_translation()) is not None:
                # If transform is a translation
                t.t += dt
                center = Point(*t.transform(self.center)[0])
                top = int(np.floor(self.top + dt.y)) if self.top is not None else None
                bottom = int(np.ceil(self.bottom + dt.y)) if self.bottom is not None else None
                return self.__class__(center, self.radius, top, bottom)
            if (t := transform.as_resize_translation()) is not None and np.isclose(abs(t.s[0]), abs(t.s[1])):
                # If transform is a form of isotropic scaling + translation
                t.t += dt
                center = Point(*t.transform(self.center)[0])
                radius = self.radius * abs(t.s[0])
                top = int(np.floor(t.s[0] * self.top + t.t[0])) if self.top is not None else None
                bottom = int(np.ceil(t.s[0] * self.bottom + t.t[0])) if self.bottom is not None else None
                return self.__class__(center, radius, top, bottom)
            if self.top is None and self.bottom is None and (t := transform.as_similarity_transform()) is not None:
                # If transform is a Similarity transform (combination of rotation, isotropic scaling and translation) we can keep the circular shape of the ROI
                t.t += dt
                center = Point(*t.transform(self.center)[0])
                radius = self.radius * t.s
                return self.__class__(center, radius, None, None)
        return None
