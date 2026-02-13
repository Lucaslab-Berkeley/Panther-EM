"""Custom warp transformation functions.

The `scikit-image` package includes a function for a forward cartesian-to-polar
transformation on 2D images, but the inverse polar-to-cartesian transformation is
absent. This module provides the inverse transformation as well as namespace imports
for the `scikit-image` polar transformation functions.
"""

from typing import Any, Literal

import numpy as np
import torch

from .coordinates import (
    inverse_cartesian_to_offset_polar_mapping,
    inverse_offset_polar_to_cartesian_mapping,
)
from .warp_backends import (
    detect_device,
    ensure_device,
    get_warp_function,
)

_TRANSFORMER_CACHE = {}


def _get_bhw_of_image(
    image: np.ndarray | torch.Tensor,
) -> tuple[int | None, int, int]:
    """Utility function to get the batch, height, and width of an image."""
    if image.ndim == 3:
        b, w, h = image.shape
        return b, w, h
    elif image.ndim == 2:
        w, h = image.shape
        return None, w, h
    else:
        raise ValueError(f"Input image must be 2D or 3D (batched), got {image.ndim}D.")


def warp_offset_polar(
    image: np.ndarray | torch.Tensor,
    num_angle: int,
    num_radius: int,
    center: tuple[float, float],
    radius: float,
    **kwargs: dict[Any, Any],
) -> np.ndarray | torch.Tensor:
    """Wrapper around the OffsetPolarTransform class for the forward transformation.

    This function automatically detects whether the input is a NumPy array or PyTorch
    tensor and uses the appropriate backend (CPU or GPU).

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image in cartesian coordinates.
    num_angle : int
        Number of angular samples in the output polar image.
    num_radius : int
        Number of radial samples in the output polar image.
    center : tuple[float, float]
        Center of the transformation (row, col).
    radius : float
        Maximum radius for the transformation.
    **kwargs
        Additional arguments passed to the warp function.

    Returns
    -------
    np.ndarray or torch.Tensor
        Warped image in offset polar coordinates (same type as input).
    """
    # Auto-detect device from input
    device = detect_device(image)
    _, h, w = _get_bhw_of_image(image)

    # Create cache key that includes device
    key = (device, center, radius, num_angle, num_radius, (h, w))
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = OffsetPolarTransform(
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=h,
            width=w,
            device=device,
        )

    # Obtain the transformer from the cache and call the to_offset_polar method
    transformer = _TRANSFORMER_CACHE[key]
    return transformer.to_offset_polar(image, **kwargs)


def warp_offset_polar_inverse(
    image: np.ndarray | torch.Tensor,
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    **kwargs: dict[Any, Any],
) -> np.ndarray | torch.Tensor:
    """Wrapper around the OffsetPolarTransform class for the inverse transformation.

    This function automatically detects whether the input is a NumPy array or PyTorch
    tensor and uses the appropriate backend (CPU or GPU).

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image in offset polar coordinates.
    height : int
        Height of the output cartesian image.
    width : int
        Width of the output cartesian image.
    center : tuple[float, float]
        Center of the transformation (row, col).
    radius : float
        Maximum radius for the transformation.
    **kwargs
        Additional arguments passed to the warp function.

    Returns
    -------
    np.ndarray or torch.Tensor
        Warped image in cartesian coordinates (same type as input).
    """
    # Auto-detect device from input
    device = detect_device(image)
    _, na, nr = _get_bhw_of_image(image)

    # Create cache key that includes device
    key = (device, center, radius, na, nr, (height, width))
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = OffsetPolarTransform(
            center=center,
            radius=radius,
            num_angle=na,
            num_radius=nr,
            height=height,
            width=width,
            device=device,
        )

    transformer = _TRANSFORMER_CACHE[key]
    return transformer.to_cartesian(image, **kwargs)


class OffsetPolarTransform:
    """Manages offset polar coordinate transformations with coordinate caching.

    Parameters
    ----------
    center : tuple[float, float]
        Center of the transformation (row, col) in cartesian coordinates.
    radius : float
        Maximum radius, in units of pixels, for the transformation.
    num_angle : int
        Number of angular samples in the offset polar image.
    num_radius : int
        Number of radial samples in the offset polar image.
    height : int
        Height of the cartesian image.
    width : int
        Width of the cartesian image.
    device : {"numpy", "cuda"}, optional
        Computational device. If "cuda", all input arrays/tensors must be
        PyTorch CUDA tensors. If "numpy", inputs must be NumPy arrays or
        CPU PyTorch tensors. By default "numpy".

    Methods
    -------
    from_image(
        image_shape,
        num_angle=360,
        num_radius=None,
        center=None,
        radius=None,
        device="numpy"
    ) -> OffsetPolarTransform
        Class method to create an OffsetPolarTransform instance from image shape and
        other associated parameters.
    to_offset_polar(image, order=5, mode='symmetric', **kwargs)
        Warp a cartesian image to offset polar coordinates. Accepts 2D or 3D (batched)
        images. By default, uses the maximum order-5 interpolation and symmetric
        padding. Additional keyword arguments are passed to the warp function.
    to_cartesian(image, order=5, mode='symmetric', **kwargs)
        Warp an offset polar image to cartesian coordinates. Accepts 2D or 3D (batched)
        images. By default, uses the maximum order-5 interpolation and symmetric
        padding. Additional keyword arguments are passed to the warp function.
    clear_cache()
        Clear the cached coordinate mappings.
    """

    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        num_angle: int,
        num_radius: int,
        height: int,
        width: int,
        device: Literal["numpy", "cuda"] = "numpy",
    ) -> None:
        """Initialize OffsetPolarTransform."""
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width
        self.device = device

        # Get the appropriate warp function for this device
        self._warp_fn = get_warp_function(device)

        # Cache the coordinate mappings for efficiency
        self._cartesian_to_offset_polar_coords: np.ndarray | torch.Tensor | None = None
        self._offset_polar_to_cartesian_coords: np.ndarray | torch.Tensor | None = None

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_angle: int = 360,
        num_radius: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
        device: Literal["numpy", "cuda"] = "numpy",
    ) -> "OffsetPolarTransform":
        """Create an OffsetPolarTransform instance from image shape and parameters."""
        height, width = image_shape

        if center is None:
            center = (height / 2 - 0.5, width / 2 - 0.5)

        if radius is None:
            radius = height / 2  # Assuming square images, radius is half the height

        if num_radius is None:
            num_radius = int(np.ceil(radius))

        return cls(
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=height,
            width=width,
            device=device,
        )

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the cartesian image."""
        return (self.height, self.width)

    @property
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the offset polar image."""
        return (self.num_angle, self.num_radius)

    def _compute_offset_polar_to_cartesian_coords(
        self,
    ) -> np.ndarray | torch.Tensor:
        """Inverse coordinate mapping from offset polar to cartesian coordinates.

        Returns coordinates as NumPy array or PyTorch CUDA tensor depending on device.
        """
        if self._offset_polar_to_cartesian_coords is None:
            # Compute in NumPy first
            r = np.arange(self.height)
            c = np.arange(self.width)
            rr, cc = np.meshgrid(r, c, indexing="ij")
            output_coords = np.stack([cc.flatten(), rr.flatten()], axis=-1)

            # Compute the inverse mapping for the warp function
            source_coords = inverse_cartesian_to_offset_polar_mapping(
                output_coords,
                num_angle=self.num_angle,
                num_radius=self.num_radius,
                max_radius=self.radius,
                center=self.center,
            )

            # Reshape to (2, cols, rows) for warp function
            source_coords = source_coords[:, [1, 0]]  # Swap columns to (col, row) order
            source_coords = source_coords.T.reshape(2, self.width, self.height)

            # Convert to appropriate device
            source_coords = ensure_device(source_coords, self.device)
            self._offset_polar_to_cartesian_coords = source_coords

        return self._offset_polar_to_cartesian_coords

    def _compute_cartesian_to_offset_polar_coords(self) -> np.ndarray | torch.Tensor:
        """Inverse coordinate mapping from cartesian to offset polar coordinates.

        Returns coordinates as NumPy array or PyTorch CUDA tensor depending on device.
        """
        if self._cartesian_to_offset_polar_coords is None:
            # Compute in NumPy first
            t = np.arange(self.num_angle)
            r = np.arange(self.num_radius)
            tt, rr = np.meshgrid(t, r, indexing="ij")
            output_coords = np.stack([rr.flatten(), tt.flatten()], axis=-1)

            # Compute the inverse mapping for the warp function
            source_coords = inverse_offset_polar_to_cartesian_mapping(
                output_coords,
                num_angle=self.num_angle,
                num_radius=self.num_radius,
                max_radius=self.radius,
                center=self.center,
            )

            # Reshape to (2, rows, cols) for warp function
            source_coords = source_coords[:, [1, 0]]  # Swap columns to (col, row) order
            source_coords = source_coords.T.reshape(2, self.num_angle, self.num_radius)

            # Convert to appropriate device
            source_coords = ensure_device(source_coords, self.device)
            self._cartesian_to_offset_polar_coords = source_coords

        return self._cartesian_to_offset_polar_coords

    def to_offset_polar(
        self,
        image: np.ndarray | torch.Tensor,
        order: int = 5,
        mode: str = "symmetric",
        **kwargs: dict[Any, Any],
    ) -> np.ndarray | torch.Tensor:
        """Warp a cartesian image to offset polar coordinates.

        For CUDA device, image must be a PyTorch CUDA tensor on the same device
        as the cached coordinates.
        """
        # Validate device matches
        image_device = detect_device(image)
        if image_device != self.device:
            raise ValueError(
                f"Image device ({image_device}) does not match transformer "
                f"device ({self.device}). Convert image or create a new "
                f"transformer with the correct device."
            )

        # Handle a batch dimension
        if image.ndim == 3:
            if self.device == "numpy":
                return np.stack(
                    [self.to_offset_polar(img, order, mode, **kwargs) for img in image]
                )
            else:  # cuda
                return torch.stack(
                    [self.to_offset_polar(img, order, mode, **kwargs) for img in image]
                )

        if image.ndim != 2:
            raise ValueError(
                f"Input image must be 2D or 3D (batched), got {image.ndim}D."
            )

        # Handle case where input is complex - treat real and imag parts separately
        if self.device == "numpy":
            is_complex = np.iscomplexobj(image)
        else:
            is_complex = isinstance(image, torch.Tensor) and image.is_complex()

        if is_complex:
            real_part = self.to_offset_polar(image.real, order, mode, **kwargs)
            imag_part = self.to_offset_polar(image.imag, order, mode, **kwargs)
            if self.device == "numpy":
                return real_part + 1j * imag_part
            else:
                return torch.complex(real_part, imag_part)

        # Get the source coordinates and call warp function
        source_coords = self._compute_cartesian_to_offset_polar_coords()
        warped = self._warp_fn(
            image,
            source_coords,
            output_shape=self.polar_shape,
            order=order,
            mode=mode,
            **kwargs,
        )

        return warped

    def to_cartesian(
        self,
        image: np.ndarray | torch.Tensor,
        order: int = 5,
        mode: str = "symmetric",
        **kwargs: dict[Any, Any],
    ) -> np.ndarray | torch.Tensor:
        """Warp an offset polar image to cartesian coordinates.

        For CUDA device, image must be a PyTorch CUDA tensor on the same device
        as the cached coordinates.
        """
        # Validate device matches
        image_device = detect_device(image)
        if image_device != self.device:
            raise ValueError(
                f"Image device ({image_device}) does not match transformer "
                f"device ({self.device}). Convert image or create a new "
                f"transformer with the correct device."
            )

        # Handle a batch dimension
        if image.ndim == 3:
            if self.device == "numpy":
                return np.stack(
                    [self.to_cartesian(img, order, mode, **kwargs) for img in image]
                )
            else:  # cuda
                return torch.stack(
                    [self.to_cartesian(img, order, mode, **kwargs) for img in image]
                )

        if image.ndim != 2:
            raise ValueError(
                f"Input image must be 2D or 3D (batched), got {image.ndim}D."
            )

        # Handle case where input is complex - treat real and imag parts separately
        if self.device == "numpy":
            is_complex = np.iscomplexobj(image)
        else:
            is_complex = isinstance(image, torch.Tensor) and image.is_complex()

        if is_complex:
            real_part = self.to_cartesian(image.real, order, mode, **kwargs)
            imag_part = self.to_cartesian(image.imag, order, mode, **kwargs)
            if self.device == "numpy":
                return real_part + 1j * imag_part
            else:
                return torch.complex(real_part, imag_part)

        # Get the source coordinates and call warp function
        source_coords = self._compute_offset_polar_to_cartesian_coords()
        warped = self._warp_fn(
            image,
            source_coords,
            output_shape=self.cartesian_shape,
            order=order,
            mode=mode,
            **kwargs,
        )

        return warped

    def clear_cache(self) -> None:
        """Clear the cached coordinate mappings."""
        self._cartesian_to_offset_polar_coords = None
        self._offset_polar_to_cartesian_coords = None
