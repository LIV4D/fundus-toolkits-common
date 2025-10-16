from __future__ import annotations

__all__ = ["PrePostProcessing", "basic_fundus_pre_postprocessing", "TensorSpec"]

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from types import EllipsisType
from typing import Any, Literal, NamedTuple, Optional, Protocol

import torch

from ..utils.image import crop_pad
from ..utils.math import ensure_superior_multiple
from ..utils.torch import DeviceLikeType


class TensorSpec(NamedTuple):
    name: str
    dim_names: tuple[str, ...]
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
        dim_names: tuple[str, ...] | EllipsisType = ...,
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


type PreprocessingMethod[**P] = Callable[P, tuple[tuple[torch.Tensor, ...], dict[str, Any]]]


class PostProcessingMethod(Protocol):
    def __call__(self, *model_outputs: torch.Tensor, preprocessing_info: dict[str, Any]) -> tuple[Any, ...]: ...


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
    input_info: tuple[TensorSpec, ...]
    model_input_info: tuple[TensorSpec, ...]
    model_output_info: tuple[TensorSpec, ...]
    output_info: tuple[TensorSpec, ...]

    def bind_preprocess(
        self, device: Optional[DeviceLikeType] = None, **kwargs: Any
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
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

    def update(
        self,
        preprocess: PreprocessingMethod | EllipsisType = ...,
        postprocess: PostProcessingMethod | EllipsisType = ...,
        input_info: tuple[TensorSpec, ...] | EllipsisType = ...,
        model_input_info: tuple[TensorSpec, ...] | EllipsisType = ...,
        model_output_info: tuple[TensorSpec, ...] | EllipsisType = ...,
        output_info: tuple[TensorSpec, ...] | EllipsisType = ...,
    ) -> PrePostProcessing:
        """
        Create a new PrePostProcessing updated with the given keyword arguments.
        """
        return PrePostProcessing(
            preprocess=preprocess if preprocess is not ... else self.preprocess,
            postprocess=postprocess if postprocess is not ... else self.postprocess,
            input_info=input_info if input_info is not ... else self.input_info,
            model_input_info=model_input_info if model_input_info is not ... else self.model_input_info,
            model_output_info=model_output_info if model_output_info is not ... else self.model_output_info,
            output_info=output_info if output_info is not ... else self.output_info,
        )


########################################################################################################################
def basic_fundus_pre_postprocessing(
    standard_resolution: Optional[int] = 1024,
    normalize_mean: tuple[float, float, float] | bool = False,
    normalize_std: tuple[float, float, float] | bool = False,
    *,
    model_name: Optional[str] = None,
    segmented_structure_name: Optional[str] = None,
    output_channels: bool | Sequence[str] = True,
    rgb_to_bgr: bool = False,
    pad_to_multiple: Optional[int] = None,
    auto_resize=True,
    final_activation: Optional[Literal["softmax", "sigmoid"]] = None,  # noqa: F821
) -> PrePostProcessing:
    """Create a basic pre/post-processing pipeline for fundus images.

    Parameters
    ----------
    standard_resolution : Optional[int], optional
        The standard resolution to use for resizing, by default 1024.
    normalize_mean : tuple[float, float, float] | bool, optional
        The mean to use for normalization. If set to True, uses ImageNet mean, by default False.
    normalize_std : tuple[float, float, float] | bool, optional
        The standard deviation to use for normalization. If set to True, uses ImageNet std, by default False.
    model_name : Optional[str], optional
        The name of the model. Only used for warning messages, by default None.
    rgb_to_bgr : bool, optional
        Whether to convert the input image from RGB to BGR, by default False.
    pad_to_multiple : Optional[int], optional
        If provided, pad the input image to be a multiple of this value, by default None.
    auto_resize : bool, optional
        Whether to automatically resize the input image, by default True.

    Returns
    -------
    PrePostProcessing
        The pre/post-processing pipeline.
    """
    if model_name is None:
        model_name = "this model"

    ## === PREPROCESSING === ##
    normalize_mean_std = None
    if not (normalize_mean is False and normalize_std is False):
        if normalize_mean is True:
            mean = (0.485, 0.456, 0.406)  # ImageNet mean
        elif isinstance(normalize_mean, Sequence) and len(normalize_mean) == 3:
            mean = normalize_mean
        else:
            raise ValueError("normalize_mean must be a tuple of 3 floats or True.")

        if normalize_std is True:
            std = (0.229, 0.224, 0.225)  # ImageNet std
        elif isinstance(normalize_std, Sequence) and len(normalize_std) == 3:
            std = normalize_std
        else:
            raise ValueError("normalize_std must be a tuple of 3 floats or True.")

        normalize_mean_std = (mean, std)

    def preprocess(
        fundus: torch.Tensor, device: Optional[DeviceLikeType] = None
    ) -> tuple[tuple[torch.Tensor], dict[str, Any]]:
        x = fundus

        # --- Preprocess torch tensor ---
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3 or 4 dimensions, got {x.ndim}.")
        assert x.shape[1] in (1, 3), f"Fundus image must have 1 or 3 channels, got {x.shape[1]}."
        if device is not None:
            x = x.to(device)

        final_shape = tuple(x.shape[-2:])
        if fundus.ndim == 4:
            final_shape = (x.shape[0],) + final_shape
        preprocessing_info: dict[str, Any] = {"final_shape": final_shape}

        # --- Resize if necessary ---
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

        # --- Flip RGB to BGR, if necessary ---
        if rgb_to_bgr:
            x = torch.flip(x, [1])

        # --- Normalize, if necessary ---
        if normalize_mean_std is not None:
            mean, std = [
                torch.as_tensor(_, device=x.device, dtype=x.dtype).view(1, -1, 1, 1) for _ in normalize_mean_std
            ]
            x = (x - mean) / std

        # --- Ensure x shape is a multiple of pad_to_multiple ---
        if pad_to_multiple is not None:
            padded_shape = [ensure_superior_multiple(s, pad_to_multiple) for s in x.shape]
            x = crop_pad(x, padded_shape)

        return (x,), preprocessing_info

    ## === POSTPROCESSING === ##
    match final_activation:
        case "softmax":
            final_activation_f = partial(torch.nn.functional.softmax, dim=1)
        case "sigmoid":
            final_activation_f = torch.sigmoid
        case None:
            final_activation_f = None
        case _:
            raise ValueError(f"Unknown activation: {final_activation}, should be one of [None, 'softmax', 'sigmoid'].")

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        (y,) = model_outputs

        # --- Apply activation if necessary ---
        if final_activation_f is not None:
            y = final_activation_f(y)

        # --- Rescale the output if necessary ---
        if "scale_factor" in preprocessing_info:
            f = preprocessing_info["scale_factor"]
            y = torch.nn.functional.interpolate(y, scale_factor=(1 / f,) * 2, mode="bilinear")

        # --- Crop or pad the output to the final shape ---
        final_shape = preprocessing_info.get("final_shape", None)
        if final_shape is not None:
            y = crop_pad(y, final_shape[-2:])
            if len(final_shape) == 2:
                y = y[0]

        return (y,)

    ## === DOCUMENTATION === ##
    input_info = TensorSpec("fundus", ("C", "H", "W"), description="The fundus image to segment.")

    output_info_name = segmented_structure_name if segmented_structure_name is not None else "segmentation_probability"
    output_model_info = TensorSpec(
        output_info_name,
        (() if not output_channels else ("C",)) + ("H", "W"),
        description="The segmentation " + ("probabilities" if final_activation_f is None else "logits") + ".",
        dtype=torch.float32,
    )

    if isinstance(output_channels, Sequence):
        desc = "The segmentation probabilities. Channels definition: [" + ", ".join(output_channels) + "]."
        output_info = output_model_info.update(description=desc)
    else:
        output_info = output_model_info.update(description="The segmentation probabilities.")

    return PrePostProcessing(
        preprocess=preprocess,
        postprocess=postprocess,
        input_info=(input_info,),
        model_input_info=(input_info.update(dtype=torch.float32),),
        model_output_info=(output_model_info,),
        output_info=(output_info,),
    )
