from __future__ import annotations

__all__ = ["PrePostProcessing", "basic_fundus_pre_postprocessing", "TensorSpec"]

import warnings
from collections.abc import Callable
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Tuple,
    TypeAlias,
)

import torch

from .image import crop_pad
from .math import ensure_superior_multiple
from .torch import TensorArray, img_to_torch

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


class TensorSpec(NamedTuple):
    name: str
    dim_names: Tuple[str, ...]
    dtype: Optional[torch.dtype] = None
    description: str = ""
    optional: bool = False

    def __str__(self) -> str:
        doc = self.name + f" [{', '.join(self.dim_names)}"
        if self.dtype is not None:
            doc += f" | {self.dtype}"
        doc += "]"

        if self.description:
            doc += f": {self.description}"
        if self.optional:
            doc += " (optional)"

        return doc

    def __repr__(self) -> str:
        return (
            f"TensorSpec(name={self.name}, dim_names={self.dim_names}, dtype={self.dtype}, "
            f"description={self.description})"
        )

    def update(
        self,
        name: str | EllipsisType = ...,
        dim_names: Tuple[str, ...] | EllipsisType = ...,
        dtype: Optional[torch.dtype] | EllipsisType = ...,
        description: str | EllipsisType = ...,
        optional: bool | EllipsisType = ...,
    ) -> TensorSpec:
        """
        Create a new TensorSpec updated with the given keyword arguments.
        """
        return TensorSpec(
            name=name if name is not ... else self.name,
            dim_names=dim_names if dim_names is not ... else self.dim_names,
            dtype=dtype if dtype is not ... else self.dtype,
            description=description if description is not ... else self.description,
            optional=optional if optional is not ... else self.optional,
        )


P = ParamSpec("P")


PreprocessingMethod: TypeAlias = Callable[P, Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]]


class PostProcessingMethod(Protocol):
    def __call__(self, *model_outputs: torch.Tensor, preprocessing_info: Dict[str, Any]) -> Tuple[Any, ...]: ...


class PrePostProcessing(NamedTuple):
    """Preprocessing and postprocessing methods for a model.

    This class encapsulates the preprocessing and postprocessing methods for a model, along with documentation about the expected inputs and outputs.

    It contains the following attributes:
    - ``preprocess``: A callable that takes the input tensors and returns a tuple of the models inputs and a dictionary containing the necessary information to perform the postprocessing.
    - ``postprocess``: A callable that takes the model outputs and the preprocessing information and returns the final outputs.
    - ``input_info``: A tuple of ``TensorSpec`` describing the expected inputs to the ``preprocess`` method.
    - ``model_input_info``: A tuple of ``TensorSpec`` describing the inputs to the model (or the output of the ``preprocess`` method).
    - ``model_output_info``: A tuple of ``TensorSpec`` describing the outputs of the model.
    - ``output_info``: A tuple of ``TensorSpec`` describing the outputs of the ``postprocess`` method.

    Example
    -------
    PrePostProcessing class are meant to be used as follows:
    >>> pre_post: PrePostProcessing = basic_fundus_pre_post_processing("my_model", standard_resolution=1024)
    >>> # 1. Preprocess the input image (e.g. resize and normalize)
    >>> model_inputs, preprocessing_info = pre_post.preprocess(fundus_image, device="cuda")
    >>> # 2. Run the model
    >>> model_outputs = model(*model_inputs)
    >>> # 3. Postprocess the model outputs (e.g. resize back to original size, thresholding, etc.)
    >>> final_outputs = pre_post.postprocess(*model_outputs, preprocessing_info=preprocessing_info)

    Documentation on the expected inputs and outputs can be accessed as follows:
    >>> pre_post.print_infos()
    Inputs to preprocess:
       - fundus [C, H, W]: The fundus image to segment.
    Inputs to model:
       - fundus [C, H, W | torch.float32]: The fundus image to segment.
    Outputs from model:
       - vessels [H, W | torch.float32]: The segmentation output (without logits).
    Outputs from postprocess:
       - vessels [H, W | torch.float32]: The segmentation output (without logits).
    """  # noqa: E501

    preprocess: PreprocessingMethod
    postprocess: PostProcessingMethod
    input_info: Tuple[TensorSpec, ...]
    model_input_info: Tuple[TensorSpec, ...]
    model_output_info: Tuple[TensorSpec, ...]
    output_info: Tuple[TensorSpec, ...]

    def bind_preprocess(
        self, device: Optional[DeviceLikeType] = None, **kwargs: Any
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, Any]]:
        """
        Bind the preprocess method with the given arguments.
        """
        import inspect

        # Bind input arguments
        args = []
        for arg in self.input_info:
            if arg.optional and arg.name not in kwargs:
                args.append(None)
            elif (t := kwargs.pop(arg.name, None)) is not None:
                args.append(t)
            else:
                raise ValueError(f"Missing required argument: {arg.name}")

        # Bind the device argument if provided
        if device is not None:
            kwargs["device"] = device

        # Bind the remaining arguments
        kwargs = {k: kwargs[k] for k in set(inspect.signature(self.preprocess).parameters).intersection(kwargs.keys())}

        return self.preprocess(*args, **kwargs)

    def print_infos(self) -> None:
        """Print the input and output information."""
        print("Inputs to preprocess:")
        for info in self.input_info:
            print("  -", info)

        print("Inputs to model:")
        for info in self.model_input_info:
            print("  -", info)

        print("Outputs from model:")
        for info in self.model_output_info:
            print("  -", info)

        print("Outputs from postprocess:")
        for info in self.output_info:
            print("  -", info)


########################################################################################################################
def basic_fundus_pre_postprocessing(
    standard_resolution: Optional[int] = 1024, *, model_name: Optional[str] = None, auto_resize=True
) -> PrePostProcessing:
    """Create a basic pre/post-processing pipeline for fundus images.

    Parameters
    ----------
    standard_resolution : Optional[int], optional
        The standard resolution to use for resizing, by default 1024.
    model_name : Optional[str], optional
        The name of the model. Only used for warning messages, by default None.
    auto_resize : bool, optional
        Whether to automatically resize the input image, by default True.

    Returns
    -------
    PrePostProcessing
        The pre/post-processing pipeline.
    """
    if model_name is None:
        model_name = "this model"

    def preprocess(
        fundus: TensorArray, device: Optional[DeviceLikeType] = None
    ) -> Tuple[Tuple[torch.Tensor], Dict[str, Any]]:
        x = img_to_torch(fundus, device=device)
        final_shape = tuple(x.shape[-2:])
        if fundus.ndim == 4:
            final_shape = (x.shape[0],) + final_shape
        preprocessing_info: Dict[str, Any] = {"final_shape": final_shape}

        if standard_resolution is not None and not (
            standard_resolution * 0.75 < x.shape[-1] < standard_resolution * 1.4
        ):
            if auto_resize:
                f = standard_resolution / x.shape[-1]  # Assume the image is cropped
                x = torch.nn.functional.interpolate(x, scale_factor=(f,) * 2, mode="bilinear")
                preprocessing_info["scale_factor"] = f
            else:
                warnings.warn(
                    f"Image size {x.shape[-2:]} is not optimal for {model_name}.\n"
                    f"Consider resizing the image to a size close to 1024x1024.",
                    stacklevel=2,
                )

        x = torch.flip(x, [1])  # RGB to BGR
        padded_shape = [ensure_superior_multiple(s, 32) for s in x.shape]
        x = crop_pad(x, padded_shape)

        if 1.0 < x.max() <= 255:
            x = x / 255.0

        return (x,), preprocessing_info

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        (y,) = model_outputs
        if y.ndim == 3:
            y.unsqueeze_(1)

        # --- Rescale the output if necessary ---
        if "scale_factor" in preprocessing_info:
            f = preprocessing_info["scale_factor"]
            y = torch.nn.functional.interpolate(y, scale_factor=(1 / f,) * 2, mode="bilinear")
        y = torch.argmax(y, dim=1) if y.shape[1] > 1 else y.unsqueeze(1) > 0.5

        # --- Crop or pad the output to the final shape ---
        final_shape = preprocessing_info.get("final_shape", None)
        if final_shape is not None:
            y = crop_pad(y, final_shape[-2:])
            if len(final_shape) == 2:
                y = y[0]

        return (y,)

    input_info = TensorSpec("fundus", ("C", "H", "W"), description="The fundus image to segment.")
    output_info = TensorSpec(
        "vessels", ("H", "W"), description="The segmentation output (without logits).", dtype=torch.float32
    )
    return PrePostProcessing(
        preprocess=preprocess,
        postprocess=postprocess,
        input_info=(input_info,),
        model_input_info=(input_info.update(dtype=torch.float32),),
        model_output_info=(output_info,),
        output_info=(output_info,),
    )
