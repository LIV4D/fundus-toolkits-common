from copy import copy
from enum import Enum, IntEnum
from pathlib import Path
from typing import Literal, Optional, Self, Tuple, overload

import numpy as np
import numpy.typing as npt

from .utils.geometric import Point
from .utils.image import crop_pad_center, find_centroid, read_image, resize
from .utils.safe_import import is_torch_tensor

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
    def parse(cls, value: str) -> Self:
        """Parse a string to a ReshapeMethod enum."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid reshape method: {value}.\nValid options are: {', '.join(m.value for m in ReshapeMethod)}"
            ) from e


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
        fundus=None,
        *,
        fundus_mask=None,
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
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
    ):
        shape: Optional[Tuple[int, int]] = target_shape
        if fundus is not None:
            if check_validity:
                fundus = self.load_fundus_image(fundus, target_shape, reshape_method=reshape_method)
            self._fundus = fundus
            if shape is None:
                shape = self._fundus.shape[:2]  # type: ignore[assignment]
        else:
            self._fundus = None

        if fundus_mask is not None:
            if check_validity:
                fundus_mask = self.load_fundus_mask(
                    fundus_mask, from_fundus=False, target_shape=shape, reshape_method=reshape_method
                )
            self._fundus_mask = fundus_mask
            if shape is None:
                shape = fundus_mask.shape[:2]  # type: ignore[assignment]
        elif self._fundus is not None:
            self._fundus_mask = self.load_fundus_mask(self._fundus, from_fundus=True)

        if vessels is not None:
            if check_validity:
                vessels = self.load_vessels(vessels, target_shape=shape, reshape_method=reshape_method)
            self._vessels = vessels
            if shape is None:
                shape = vessels.shape[:2]  # type: ignore[assignment]
        else:
            self._vessels = None
        self._bin_vessels = None

        if av is not None:
            if check_validity:
                av = self.load_av(av, target_shape=shape, reshape_method=reshape_method)
            self._av = av
            if shape is None:
                shape = av.shape[:2]  # type: ignore[assignment]
        else:
            self._av = None

        if od is not None:
            if check_validity:
                self._od, self._od_center, self._od_size = self.load_od_macula(
                    od, shape, reshape_method=reshape_method, fit_ellipse=True
                )  # type: ignore
                if shape is None:
                    shape = self._od.shape[:2]  # type: ignore[assignment]
            else:
                self._od, self._od_center, self._od_size = od, od_center, od_size
        else:
            self._od, self._od_center, self._od_size = None, od_center, od_size

        if macula is not None:
            if check_validity:
                self._macula, self._macula_center = self.load_od_macula(macula, shape, reshape_method=reshape_method)
                if shape is None:
                    shape = self._macula.shape[:2]  # type: ignore[assignment]
            else:
                self._macula, self._macula_center = macula, macula_center
        else:
            self._macula, self._macula_center = None, macula_center

        if shape is None:
            raise ValueError("No data was provided to initialize the FundusData.")
        self._shape = shape

        if name is None:
            if isinstance(fundus, (str, Path)):
                name = Path(fundus).stem
            elif isinstance(vessels, (str, Path)):
                name = Path(vessels).stem
            elif isinstance(od, (str, Path)):
                name = Path(od).stem
            elif isinstance(macula, (str, Path)):
                name = Path(macula).stem
        self._name = name

    def update(
        self,
        fundus=None,
        fundus_mask=None,
        vessels=None,
        av=None,
        od=None,
        macula=None,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
        name: Optional[str] = None,
    ) -> Self:
        other = copy(self)
        shape = self.shape
        if fundus is not None:
            other._fundus = other.load_fundus_image(fundus, target_shape=shape, reshape_method=reshape_method)
        if fundus_mask is not None:
            other._fundus_mask = other.load_fundus_mask(fundus_mask, target_shape=shape, reshape_method=reshape_method)
        if vessels is not None:
            other._vessels = other.load_vessels(vessels, target_shape=shape, reshape_method=reshape_method)
        if av is not None:
            other._av = other.load_av(av, target_shape=shape, reshape_method=reshape_method)

        if od is not None:
            other._od, other._od_center, other._od_size = other.load_od_macula(
                od, target_shape=shape, reshape_method=reshape_method, fit_ellipse=True
            )
        if macula is not None:
            other._macula, other._macula_center = other.load_od_macula(
                macula, target_shape=shape, reshape_method=reshape_method
            )
        if name is not None:
            other._name = name
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
        fundus,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
    ) -> npt.NDArray[np.float32]:
        # --- Load the image ---
        if isinstance(fundus, (str, Path)):
            fundus = read_image(fundus)
        elif is_torch_tensor(fundus):
            fundus = fundus.numpy(force=True)

        # --- Check image ---
        assert isinstance(fundus, np.ndarray), "The image must be a numpy array."
        assert fundus.ndim == 3 and fundus.shape[2] == 3, "The image must be a color image."

        # --- Format data ---
        if fundus.dtype != np.float32:
            fundus = fundus.astype(np.float32)

        # -- Resize the image ---
        if target_shape is not None and target_shape != fundus.shape[:2]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    fundus = resize(fundus, target_shape, interpolation=True)
                case ReshapeMethod.CROP:
                    fundus = crop_pad_center(fundus, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {fundus.shape} differs from the target shape {target_shape}."
                    )
        return fundus  # type: ignore

    @classmethod
    def load_fundus_mask(
        cls,
        fundus_mask,
        from_fundus: bool = False,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
    ) -> npt.NDArray[np.bool_]:
        if from_fundus:
            # --- Compute from fundus mask ---
            from .utils.fundus import fundus_ROI

            fundus = cls.load_fundus_image(fundus_mask, target_shape, reshape_method=reshape_method)
            fundus_mask = fundus_ROI(fundus)  # Compute the fundus mask from the fundus image
        else:
            # --- Load the image ---
            if isinstance(fundus_mask, (str, Path)):
                fundus_mask = read_image(fundus_mask, cast_to_float=False)
            elif is_torch_tensor(fundus_mask):
                fundus_mask = fundus_mask.numpy(force=True)

            # --- Check image ---
            assert isinstance(fundus_mask, np.ndarray), "The image must be a numpy array."
            if fundus_mask.ndim != 2:
                raise ValueError("Invalid fundus mask")

            if fundus_mask.dtype != bool:
                fundus_mask = fundus_mask > 127 if fundus_mask.dtype == np.uint8 else fundus_mask > 0.5

        # -- Resize the image ---
        if target_shape is not None and target_shape != fundus_mask.shape[:2]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    fundus_mask = resize(fundus_mask, target_shape, interpolation=False)
                case ReshapeMethod.CROP:
                    fundus_mask = crop_pad_center(fundus_mask, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {fundus_mask.shape} differs from the target shape {target_shape}."
                    )
        return fundus_mask  # type: ignore

    @classmethod
    def load_vessels(
        cls,
        vessels,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
    ) -> npt.NDArray[np.bool_]:
        # --- Load the image ---
        if isinstance(vessels, (str, Path)):
            vessels = read_image(vessels, cast_to_float=False)
        elif is_torch_tensor(vessels):
            vessels = vessels.numpy(force=True)

        # --- Check image ---
        assert isinstance(vessels, np.ndarray), "The image must be a numpy array."
        if vessels.ndim == 3:
            # Check if the image is a binary image
            MAX = 255 if vessels.dtype == np.uint8 else 1
            vessels = np.any(vessels > MAX / 2, axis=2)
        elif vessels.ndim != 2:
            raise ValueError("Invalid fundus mask")

        if vessels.dtype != bool:
            vessels = vessels > 127 if vessels.dtype == np.uint8 else vessels > 0.5

        # -- Resize the image ---
        if target_shape is not None and target_shape != vessels.shape[:2]:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    vessels = resize(vessels, target_shape, interpolation=True)
                case ReshapeMethod.CROP:
                    vessels = crop_pad_center(vessels, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The fundus image shape {vessels.shape} differs from the target shape {target_shape}."
                    )
        return vessels  # type: ignore

    @classmethod
    def load_av(
        cls,
        av,
        target_shape: Optional[Tuple[int, int]] = None,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
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
            av = read_image(av, cast_to_float=False)
        elif is_torch_tensor(av):
            av = av.numpy(force=True)

        # --- Interpret image ---
        assert isinstance(av, np.ndarray), "The vessels map must be a numpy array."
        if av.ndim == 3:
            # Check if the image is a binary image
            MAX = 255 if av.dtype == np.uint8 else 1
            av = av > MAX / 2

            if np.all(np.all(av, axis=2)):
                av = (av[:, :, 0] * AVLabel.UNK).astype(np.uint8)
            else:
                av_map = np.zeros(av.shape[:2], dtype=np.uint8)
                if invert_av:
                    av = av[:, :, ::-1]  # Invert the channels: red (artery) and blue (vein)
                av_map[av[:, :, 0]] = AVLabel.ART
                av_map[av[:, :, 2]] = AVLabel.VEI
                av_map[av[:, :, 1]] = AVLabel.BOTH
                av_map[av[:, :, 2] & av[:, :, 0]] = AVLabel.BOTH
                av = av_map
        elif av.ndim == 2:
            if av.dtype == bool:
                av = (av * AVLabel.UNK).astype(np.uint8)
            elif np.issubdtype(av.dtype, np.integer):
                assert av.min() >= 0 and av.max() <= AVLabel.UNK, "Invalid vessels map"
                av = av.astype(np.uint8)
            else:
                raise ValueError("The vessels map must be a binary image or a label map using the AVLabel convention.")

        # --- Resize the image ---
        if target_shape is not None and av.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    vessel_mask = av > 0
                    vessel_mask = resize(vessel_mask, target_shape, interpolation=True)
                    av = resize(av, target_shape, interpolation=False) * vessel_mask
                    # TODO: Smooth av boundaries (right now, only vessel_mask is smoothed)
                case ReshapeMethod.CROP:
                    av = crop_pad_center(av, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The artery/vein maps shape {av.shape} differs from the fundus image shape {target_shape}."
                    )

        return av  # type: ignore

    @overload
    @classmethod
    def load_od_macula(
        cls,
        seg,
        target_shape: Optional[Tuple[int, int]] = None,
        *,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
        fit_ellipse: Literal[False] = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point]: ...
    @overload
    @classmethod
    def load_od_macula(
        cls,
        seg,
        target_shape: Optional[Tuple[int, int]] = None,
        *,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
        fit_ellipse: Literal[True],
    ) -> Tuple[npt.NDArray[np.bool_], Point, Point]: ...
    @classmethod
    def load_od_macula(
        cls,
        seg,
        target_shape: Optional[Tuple[int, int]] = None,
        *,
        reshape_method: ReshapeMethod = ReshapeMethod.RAISE,
        fit_ellipse: bool = False,
    ) -> Tuple[npt.NDArray[np.bool_], Point] | Tuple[npt.NDArray[np.bool_], Point, Point]:
        # --- Load the image ---
        if isinstance(seg, (str, Path)):
            seg = read_image(seg, cast_to_float=False)
        elif is_torch_tensor(seg):
            seg = seg.detach().cpu().numpy()

        # --- Interpret image ---
        assert isinstance(seg, np.ndarray), "The optic disc or macula map must be a numpy array."
        if seg.ndim == 3:
            # Check if the image is a binary image
            MAX = 255 if seg.dtype == np.uint8 else 1
            seg = np.any(seg > MAX / 2, axis=2)
        assert seg.ndim == 2, "The optic disc map must be a grayscale image."

        if seg.dtype != bool:
            MAX = 255 if seg.dtype == np.uint8 else 1
            seg = seg > MAX / 2

        # --- Resize the image ---
        if target_shape is not None and seg.shape != target_shape:
            match ReshapeMethod.parse(reshape_method):
                case ReshapeMethod.RESIZE:
                    seg = resize(seg, target_shape, interpolation=True)
                case ReshapeMethod.CROP:
                    seg = crop_pad_center(seg, target_shape)
                case ReshapeMethod.RAISE:
                    raise ValueError(
                        f"The artery/vein maps shape {seg.shape} differs from the fundus image shape {target_shape}."
                    )

        # --- Find the centroid ---
        return find_centroid(seg, fit_ellipse=fit_ellipse)

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
        return self._fundus is not None

    @property
    def image(self) -> npt.NDArray[np.float32]:
        """The fundus image in RGB format. Values are float32 in the range [0, 1]."""
        if self._fundus is None:
            raise AttributeError("The fundus image was not provided.")
        return self._fundus

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
    def od_size(self) -> Optional[Point]:
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
        return None if self._od_size is ABSENT else self._od_size

    @property
    def od_diameter(self) -> Optional[float]:
        """The diameter of the optic disc or None if the optic disc is not visible in this fundus.

        Raises
        ------
        AttributeError
            If the optic disc segmentation was not provided.
        """
        return None if self._od_size is None else self._od_size.max

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
            return None if self._macula_center.is_nan() else self._macula_center
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

    ####################################################################################################################
    #    === VISUALISATION TOOLS ===
    ####################################################################################################################
    def draw(self, labels_opacity=0.5, *, view=None):
        from jppype import Mosaic, View2D

        if isinstance(view, Mosaic):
            for v in view.views:
                self.draw(labels_opacity, view=v)
            return view
        elif view is None:
            view = View2D()
        view.add_image(self._fundus, "fundus")

        COLORS = {
            AVLabel.ART: "coral",
            AVLabel.VEI: "cornflowerblue",
            AVLabel.BOTH: "darkorchid",
            AVLabel.UNK: "gray",
            10: "white",
            11: "white",
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

        if self._fundus is not None:
            other._fundus = rotate(self._fundus, angle)
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
        if self._fundus is not None:
            other._fundus = resize(self._fundus, shape, interpolation=True)
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
