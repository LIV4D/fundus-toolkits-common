from __future__ import annotations

import typing
from collections.abc import Sequence
from copy import copy
from enum import Enum, IntEnum
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, Optional, Self, Tuple, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from .utils.geometric import Point
from .utils.image import crop_pad_center, find_centroid, label_map_to_rgb, read_image, resize, write_image
from .utils.safe_import import is_torch_tensor

if TYPE_CHECKING:
    import torch

    from .utils.typing import PathLike

    type ImageSource = torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike]

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
        vessels=None,
        av=None,
        od=None,
        od_center=None,
        od_size=None,
        macula=None,
        macula_center=None,
        name: Optional[str] = None,
        check_validity: bool = True,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        immutable: bool = False,
    ):
        shape: Optional[Tuple[int, int]] = target_shape
        if image is not None:
            if check_validity:
                image = self.load_fundus_image(image, target_shape, reshape_method=reshape_method)
            self._image = image
            if shape is None:
                shape = self._image.shape[-2:]  # type: ignore[assignment]
        else:
            self._image = None

        if roi_mask is not None:
            if check_validity:
                roi_mask = self.load_fundus_mask(
                    roi_mask, from_fundus=False, target_shape=shape, reshape_method=reshape_method
                )
            self._fundus_mask = roi_mask
            if shape is None:
                shape = roi_mask.shape[-2:]  # type: ignore[assignment]
        elif self._image is not None:
            self._fundus_mask = self.load_fundus_mask(self._image, from_fundus=True)

        if vessels is not None:
            if check_validity:
                vessels = self.load_vessels(vessels, target_shape=shape, reshape_method=reshape_method)
            self._vessels = vessels
            if shape is None:
                shape = vessels.shape[-2:]  # type: ignore[assignment]
        else:
            self._vessels = None
        self._bin_vessels = None

        if av is not None:
            if check_validity:
                av = self.load_av(av, target_shape=shape, reshape_method=reshape_method)
            self._av = av
            if shape is None:
                shape = av.shape[-2:]  # type: ignore[assignment]
        else:
            self._av = None

        if od is not None:
            if check_validity:
                self._od, self._od_center, self._od_size = self.load_od_macula(
                    od, shape, reshape_method=reshape_method, fit_ellipse=True
                )  # type: ignore
                if shape is None:
                    shape = self._od.shape[-2:]  # type: ignore[assignment]
            else:
                self._od, self._od_center, self._od_size = od, od_center, od_size
        else:
            self._od, self._od_center, self._od_size = None, od_center, od_size

        if macula is not None:
            if check_validity:
                self._macula, self._macula_center = self.load_od_macula(macula, shape, reshape_method=reshape_method)
                if shape is None:
                    shape = self._macula.shape[-2:]  # type: ignore[assignment]
            else:
                self._macula, self._macula_center = macula, macula_center
        else:
            self._macula, self._macula_center = None, macula_center

        if shape is None:
            raise ValueError("No data was provided to initialize the FundusData.")
        self._shape = shape

        if name is None:
            if isinstance(image, (str, Path)):
                name = Path(image).stem
            elif isinstance(vessels, (str, Path)):
                name = Path(vessels).stem
            elif isinstance(od, (str, Path)):
                name = Path(od).stem
            elif isinstance(macula, (str, Path)):
                name = Path(macula).stem
        self._name = name
        self._immutable = immutable

    type Fields = Literal["image", "fundus_mask", "vessels", "av", "od", "macula"]

    def update(
        self,
        image: ImageSource | EllipsisType = ...,
        fundus_mask: ImageSource | EllipsisType = ...,
        vessels: ImageSource | EllipsisType = ...,
        av: ImageSource | EllipsisType = ...,
        od: ImageSource | EllipsisType = ...,
        macula: ImageSource | EllipsisType = ...,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        name: str | EllipsisType = ...,
        inplace: bool = False,
    ) -> Self:
        if inplace:
            if self._immutable:
                raise RuntimeError("This FundusData instance is immutable.")
            other = self
        else:
            other = copy(self)

        shape = self.shape if target_shape is None else target_shape
        if image is not ...:
            other._image = other.load_fundus_image(image, target_shape=shape, reshape_method=reshape_method)
        if fundus_mask is not ...:
            other._fundus_mask = other.load_fundus_mask(fundus_mask, target_shape=shape, reshape_method=reshape_method)
        if vessels is not ...:
            other._vessels = other.load_vessels(vessels, target_shape=shape, reshape_method=reshape_method)
        if av is not ...:
            other._av = other.load_av(av, target_shape=shape, reshape_method=reshape_method)

        if od is not ...:
            other._od, other._od_center, other._od_size = other.load_od_macula(
                od, target_shape=shape, reshape_method=reshape_method, fit_ellipse=True
            )
        if macula is not ...:
            other._macula, other._macula_center = other.load_od_macula(
                macula, target_shape=shape, reshape_method=reshape_method
            )
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
        if self._fundus_mask is not None:
            self._fundus_mask.setflags(write=not immutable)
        if self._vessels is not None:
            self._vessels.setflags(write=not immutable)
        if self._av is not None:
            self._av.setflags(write=not immutable)
        if self._od is not None:
            self._od.setflags(write=not immutable)
        if self._macula is not None:
            self._macula.setflags(write=not immutable)

    def mutable_copy(self) -> Self:
        """Return a mutable copy of this FundusData."""
        other = copy(self)
        other._set_immutable_flag(False)
        return other

    def remove_od_from_vessels(self):
        updated_data = {}
        if self._vessels is not None:
            vessels = self._vessels.copy()
            vessels[self.od] = False
            updated_data["vessels"] = vessels
        if self._av is not None:
            av = self._av.copy()
            av[self.od] = AVLabel.UNK
            updated_data["av"] = av
        return self.update(**updated_data)

    ####################################################################################################################
    #    === CHECK METHODS ===
    ####################################################################################################################
    @classmethod
    def load_fundus_image(
        cls,
        fundus: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
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
            The method to use for reshaping the image, by default ReshapeMethod.RAISE. If target_shape is None, this parameter is ignored.

        Returns
        -------
        npt.NDArray[np.float32]
            The loaded fundus image as a numpy array.

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

        # -- Resize the image ---
        if target_shape is not None and target_shape != fundus_.shape[-2:]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    fundus_ = resize(fundus_, target_shape, interpolation=True)
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
    ) -> npt.NDArray[np.bool_]:
        if from_fundus:
            # --- Compute from fundus mask ---
            from .utils.fundus import fundus_ROI

            fundus = cls.load_fundus_image(fundus_mask, target_shape, reshape_method=reshape_method)
            fundus_mask_ = fundus_ROI(fundus)  # Compute the fundus mask from the fundus image
        else:
            # --- Load the image ---
            if isinstance(fundus_mask, (str, Path)):
                fundus_mask_ = read_image(fundus_mask, cast_to_float=False)
            elif is_torch_tensor(fundus_mask):
                fundus_mask_ = fundus_mask.numpy(force=True)
            elif isinstance(fundus_mask, np.ndarray):
                fundus_mask_ = fundus_mask
            else:
                raise TypeError("The fundus mask must be a path, a numpy array or a torch tensor.")

            # --- Check image ---
            if fundus_mask_.ndim != 2:
                raise ValueError("Invalid fundus mask")

            if fundus_mask_.dtype != bool:
                fundus_mask_ = fundus_mask_ > 127 if fundus_mask_.dtype == np.uint8 else fundus_mask_ > 0.5

        # -- Resize the image ---
        if target_shape is not None and target_shape != fundus_mask_.shape[-2:]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    fundus_mask_ = resize(fundus_mask_, target_shape, interpolation=False)
                case ReshapeMethod.CROP:
                    fundus_mask_ = crop_pad_center(fundus_mask_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {fundus_mask_.shape} differs from the target shape {target_shape}."
                    )
        return fundus_mask_  # type: ignore

    @classmethod
    def load_vessels(
        cls,
        vessels: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
    ) -> npt.NDArray[np.bool_]:
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

        # -- Resize the image ---
        if target_shape is not None and target_shape != vessels_.shape[-2]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    vessels_ = resize(vessels_, target_shape, interpolation=True)
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
        invert_av: bool = False,
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
            The target shape of the image, by default None.<

        reshape_method : ShapeResolve, optional
            The method used when the shape of the image is different from the target shape:
            - "resize": resize the image to the target shape.
            - "crop_pad": crop or pad the image to the target shape.
            - "raise": (by default) raise an error if the shape is different from the target shape.

        invert_av : bool, optional
            If True, the red channel is interpreted as the vein and the blue channel as the artery.

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
                av_ = (av_ * AVLabel.UNK).astype(np.uint8)
            elif np.issubdtype(av_.dtype, np.integer):
                assert av_.min() >= 0 and av_.max() <= AVLabel.UNK, "Invalid vessels map"
                av_ = av_.astype(np.uint8)
            else:
                raise ValueError("The vessels map must be a binary image or a label map using the AVLabel convention.")

        # --- Resize the image ---
        if target_shape is not None and av_.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    vessel_mask = av_ > 0
                    vessel_mask = resize(vessel_mask, target_shape, interpolation=True)
                    av_ = resize(av_, target_shape, interpolation=False) * vessel_mask
                    # TODO: Smooth av_ boundaries (right now, only vessel_mask is smoothed)
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
        *,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        fit_ellipse: Literal[False] = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point]: ...
    @overload
    @classmethod
    def load_od_macula(
        cls,
        seg: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        *,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        fit_ellipse: Literal[True],
    ) -> Tuple[npt.NDArray[np.bool_], Point, Point]: ...
    @classmethod
    def load_od_macula(
        cls,
        seg: ImageSource,
        target_shape: Optional[Tuple[int, int]] = None,
        *,
        reshape_method: ReshapeMethods = ReshapeMethod.RAISE,
        fit_ellipse: bool = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point] | Tuple[npt.NDArray[np.bool_], Point, Point]:
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

        # --- Resize the image ---
        if target_shape is not None and seg_.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    seg_ = resize(seg_, target_shape, interpolation=True)
                case ReshapeMethod.CROP:
                    seg_ = crop_pad_center(seg_, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The artery/vein maps shape {seg_.shape} differs from the fundus image shape {target_shape}."
                    )

        # --- Find the centroid ---
        return find_centroid(seg_.astype(np.bool_), fit_ellipse=fit_ellipse)

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
        """The fundus image in RGB format. Values are float32 in the range [0, 1]."""
        if self._image is None:
            raise AttributeError("The fundus image was not provided.")
        return self._image

    @property
    def has_fundus_mask(self) -> bool:
        return self._fundus_mask is not None

    @property
    def fundus_mask(self) -> npt.NDArray[np.bool_]:
        if self._fundus_mask is None:
            raise AttributeError("The fundus ROI mask was not provided.")
        return self._fundus_mask

    @property
    def has_vessels(self) -> bool:
        return self._vessels is not None

    @property
    def vessels(self) -> npt.NDArray[np.bool_]:
        """The binary segmentation of the vessels.

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
        return None if self._od_center is ABSENT else self._od_center

    def od_region(
        self,
        min_radius: float = 0,
        max_radius: Optional[float] = None,
        apply_mask: Optional[bool] = None,
        exclude_od: bool = True,
    ) -> npt.NDArray[np.bool_]:
        """Generate the mask of a region between an inner circle and an outer circle centered on the optic disc.

        Parameters
        ----------
        min_radius : float, optional
            Radius of the inner circle of the region. `min_radius` is multiplied by the optic disc diameter.

        max_radius : Optional[float], optional
            Radius of the outer circle of the region. `max_radius` is multiplied by the optic disc diameter.

        apply_mask : bool, optional
            If true, the fundus_mask is


        Returns
        -------
        npt.NDArray[bool]
            _description_
        """
        if apply_mask is None:
            apply_mask = self.has_fundus_mask
        mask = np.ones(self.shape, bool) if not apply_mask else self.fundus_mask
        if exclude_od:
            mask &= ~self.od

        if min_radius == 0 and max_radius is None:
            return mask

        od = self.od_center
        od_diameter = self.od_diameter
        assert od is not None and od_diameter is not None, (
            "Impossible to define a region around the optic: it's absent from the provided image."
        )
        H, W = self.shape
        y0, x0 = od
        dist_map = np.linalg.norm(np.stack(np.mgrid[-y0 : H - y0, -x0 : W - x0], axis=0), axis=0)  # type: ignore

        def scale_radius(radius: float) -> float:
            if exclude_od:
                radius += 0.5
            return radius * od_diameter

        if min_radius > (-0.5 if exclude_od else 0):
            mask &= dist_map >= scale_radius(min_radius)
        if max_radius is not None:
            mask &= dist_map <= scale_radius(max_radius)
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
        return None if self._macula_center is ABSENT else self._macula_center

    def infered_macula_center(self) -> Optional[Point]:
        """The center of the macula or the center of the fundus if the macula is not visible."""
        if self._macula_center is not None:
            return None if self._macula_center is ABSENT else self._macula_center
        if self._od_center is None:
            return None

        half_weight = self._shape[1] * 0.4  # Assume 45° fundus image
        if self._od_center[1] > half_weight:
            half_weight = -half_weight
        return Point(self._od_center.y, x=self._od_center.x + half_weight)

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
        overwrite: bool = False,
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

        overwrite : bool, optional
            Whether to overwrite the files if they already exist, by default False.
        Raises
        ------
        AttributeError
            If one of the requested data to save was not provided.
        """  # noqa: E501
        on_exists = "overwrite" if overwrite else "ignore"
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

    def draw(self, labels_opacity=0.5, *, view=None):
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
            labels[self._od] = 10

        if self._macula is not None:
            labels[self._macula] = 11

        view.add_label(labels, "AnatomicalData", opacity=labels_opacity, colormap=COLORS)
        return view

    def show(self, labels_alpha=0.5):
        self.draw(labels_alpha).show()

    ####################################################################################################################
    #    === Utilities ===
    ####################################################################################################################
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
        from .utils.image import rotate

        other = copy(self)

        if self._image is not None:
            other._image = rotate(self._image, angle)
        if self._fundus_mask is not None:
            other._fundus_mask = rotate(self._fundus_mask, angle)
        if self._vessels is not None:
            other._vessels = rotate(self._vessels, angle)
        if self._od is not None:
            other._od = rotate(self._od, angle)
        if self._macula is not None:
            other._macula = rotate(self._macula, angle)
        return other

    def rescale(self, size: float | int | Tuple[int, int]) -> Self:
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
        from .utils.image import resize

        if isinstance(size, tuple):
            shape = size
        if isinstance(size, int):
            shape = (size, size)
        elif isinstance(size, float):
            shape = self.shape
            shape = int(round(shape[0] * size)), int(round(shape[1] * size))
        else:
            raise

        other = copy(self)
        if self._image is not None:
            other._image = resize(self._image, shape, interpolation=True)
        if self._fundus_mask is not None:
            other._fundus_mask = resize(self._fundus_mask, shape, interpolation=True)
        if self._vessels is not None:
            other._vessels = resize(self._vessels, shape, interpolation=True)
        if self._av is not None:
            vessel_mask = self._av > 0
            other._av = resize(self._av, shape, interpolation=False) * vessel_mask
            # TODO: Smooth av boundaries (right now, only vessel_mask is smoothed)

        if self._od is not None:
            other._od = resize(self._od, shape, interpolation=True)
        if self._macula is not None:
            other._macula = resize(self._macula, shape, interpolation=True)
        return other
