"""Coordinate transforms between Cartesian and offset-polar spaces.

This module provides the :class:`OffsetPolarTransform` class, which implements the
:class:`~panther_em.utils.transform_base.CoordinateTransform` interface for the
offset-polar coordinate system.  Features:

- NumPy (CPU) and PyTorch CUDA backends with the same API.
- Single 2D images and batched 3D inputs.
- Per-instance coordinate-grid caching for repeated warps on the same geometry.
- Optional energy-preserving transforms via Jacobian correction.
- Circular padding at the 0°/360° boundary for the inverse transform.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch

from .coordinates import (
    inverse_cartesian_to_offset_polar_mapping,
    inverse_offset_polar_to_cartesian_mapping,
    jacobian_correction_offset_polar,
)
from .transform_base import CoordinateTransform, register_transform
from .warp_backends import (
    detect_device,
    ensure_device,
    get_warp_function,
)


def _normalize_transform_device(
    device: Literal["numpy"] | str | torch.device,
) -> Literal["numpy"] | torch.device:
    """Normalize transform device to either 'numpy' or a concrete CUDA device."""
    if device == "numpy":
        return "numpy"

    device_obj = device if isinstance(device, torch.device) else torch.device(device)

    if device_obj.type == "cuda":
        return device_obj
    if device_obj.type == "cpu":
        return "numpy"

    raise ValueError(
        f"Unsupported device type: {device_obj}. Supported: 'numpy' or CUDA device."
    )


def _get_bhw_of_image(image: np.ndarray | torch.Tensor) -> tuple[int | None, int, int]:
    """Utility function to get the batch, height, and width of an image."""
    if image.ndim == 3:
        b, w, h = image.shape
        return b, w, h
    elif image.ndim == 2:
        w, h = image.shape
        return None, w, h

    raise ValueError(f"Input image must be 2D or 3D (batched), got {image.ndim}D.")


@register_transform
class OffsetPolarTransform(CoordinateTransform):
    """Manages offset polar coordinate transformations with coordinate caching.

    The *offset polar* coordinate system is identical to standard polar
    coordinates except that every other radial ring is angularly offset by
    half of the angular step size (``delta_theta / 2``). Improves spatial coverage
    (compared to standard polar coords) without introducing much more complexity.

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
    device : {"numpy"} | str | torch.device, optional
        Computational device. CUDA devices may be provided as strings (e.g.
        "cuda:1") or torch.device objects. If "numpy"/CPU, inputs must be NumPy
        arrays or CPU PyTorch tensors. By default "numpy".

    Methods
    -------
    from_image(image_shape, num_angle=360, num_radius=None, center=None, radius=None,
               device="numpy") -> OffsetPolarTransform
        Convenience constructor from image shape and polar geometry parameters.
    to_transform_space(image, preserve_energy=True, **kwargs)
        Warp a Cartesian image to offset polar coordinates (2D or batched 3D).
    to_cartesian(image, preserve_energy=False, wrap_angular_axis=True, **kwargs)
        Warp an offset polar image back to Cartesian coordinates.
    clear_cache()
        Release cached coordinate grids and Jacobian arrays.
    """

    # --- CoordinateTransform class attributes ---
    transform_name: str = "offset_polar"
    supports_energy_preservation: bool = True
    has_periodic_axis: bool = True
    periodic_axis: int = 0  # angle axis is periodic

    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        num_angle: int,
        num_radius: int,
        height: int,
        width: int,
        device: Literal["numpy"] | str | torch.device = "numpy",
    ) -> None:
        """Initialize OffsetPolarTransform."""
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width
        self.device = _normalize_transform_device(device)

        # Get the appropriate warp function for this device
        self._warp_fn = get_warp_function(device)

        # Cache the coordinate mappings for efficiency
        self._cartesian_to_offset_polar_coords: np.ndarray | torch.Tensor | None = None
        self._offset_polar_to_cartesian_coords: np.ndarray | torch.Tensor | None = None
        self._jacobian_correction: np.ndarray | torch.Tensor | None = None

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_angle: int = 360,
        num_radius: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
        device: Literal["numpy"] | str | torch.device = "numpy",
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

    def clear_cache(self) -> None:
        """Clear the cached coordinate mappings."""
        self._cartesian_to_offset_polar_coords = None
        self._offset_polar_to_cartesian_coords = None
        self._jacobian_correction = None

    # ============================================================================
    # Internal (private, cached) coordinate mapping computations
    # ============================================================================

    def _compute_offset_polar_to_cartesian_coords(self) -> np.ndarray | torch.Tensor:
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

    def _jacobian_correction_offset_polar(self) -> np.ndarray | torch.Tensor:
        """Compute and cache the exact area representation for each polar cell."""
        if self._jacobian_correction is None:
            jac = jacobian_correction_offset_polar(
                self.num_angle, self.num_radius, self.radius
            )
            jac = np.sqrt(jac)
            jac = jac.astype(np.float32)

            # Ensure proper shape (1, num_radius) to broadcast across angle dimension
            jac = jac.reshape(1, self.num_radius)
            self._jacobian_correction = ensure_device(jac, self.device)

        return self._jacobian_correction

    # ============================================================================
    # Helper functions for transformations
    # ============================================================================

    def _validate_device(self, image: np.ndarray | torch.Tensor) -> None:
        """Ensure input image matches transformer device."""
        image_device = detect_device(image)
        if image_device != self.device:
            raise ValueError(
                f"Image device ({image_device}) does not match transformer "
                f"device ({self.device}). Convert image or create a new "
                f"transformer with the correct device."
            )

    def _apply_recursive_routing(
        self,
        func: Callable,
        image: np.ndarray | torch.Tensor,
        **kwargs: dict[Any, Any],
    ) -> np.ndarray | torch.Tensor | None:
        """Batched and complex routing for transformation function."""
        if image.ndim == 3:
            result = [func(im, **kwargs) for im in image]
            if self.device == "numpy":
                return np.stack(result)
            else:
                return torch.stack(result)

        if image.ndim != 2:
            raise ValueError(
                f"Input image must be 2D or 3D (batched), got {image.ndim}D."
            )

        # Handle case where input is complex
        is_complex = (
            np.iscomplexobj(image)
            if self.device == "numpy"
            else (isinstance(image, torch.Tensor) and image.is_complex())
        )
        if is_complex:
            real_part = func(image.real, **kwargs)
            imag_part = func(image.imag, **kwargs)
            if self.device == "numpy":
                return real_part + 1j * imag_part
            else:
                return torch.complex(real_part, imag_part)

        # This is the base case with a single, real-valued 2D image.
        # Will signal public caller method to actually perform the warp transformation.
        return None

    def _apply_jacobian(
        self, image: np.ndarray | torch.Tensor, inverse: bool
    ) -> np.ndarray | torch.Tensor:
        """Applies or removes the calculated Jacobian correction."""
        jac = self._jacobian_correction_offset_polar()
        if image.ndim == 3:
            jac = jac[None, ...] if self.device == "numpy" else jac.unsqueeze(0)

        return image / jac if inverse else image * jac

    # ============================================================================
    # Public functions (and wrappers) for transformations
    # ============================================================================

    def to_cartesian(  # type: ignore[override]
        self,
        image: np.ndarray | torch.Tensor,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = False,
        wrap_angular_axis: bool = True,
        **kwargs: dict[Any, Any],
    ) -> np.ndarray | torch.Tensor:
        """Warp an offset polar image to cartesian coordinates.

        For CUDA device, image must be a PyTorch CUDA tensor on the same device
        as the cached coordinates.
        """
        self._validate_device(image)

        routed_result = self._apply_recursive_routing(
            self.to_cartesian,
            image,
            order=order,  # type: ignore
            mode=mode,  # type: ignore
            cval=cval,  # type: ignore
            preserve_energy=preserve_energy,  # type: ignore
            wrap_angular_axis=wrap_angular_axis,  # type: ignore
            **kwargs,
        )
        if routed_result is not None:
            return routed_result

        # Case where the routed result is None --> have a single 2D real-valued image
        image_to_warp = (
            self._apply_jacobian(image, inverse=True) if preserve_energy else image
        )
        source_coords = self._compute_offset_polar_to_cartesian_coords()

        # If requested, apply wrap padding along angular axis
        if wrap_angular_axis:
            pad_size = order
            if self.device == "numpy":
                image_to_warp = np.pad(
                    image_to_warp,
                    pad_width=((pad_size, pad_size), (0, 0)),
                    mode="wrap",
                )
            else:
                image_to_warp.unsqueeze_(0)  # dummy batch dim for padding
                image_to_warp = torch.nn.functional.pad(
                    image_to_warp, pad=(0, 0, pad_size, pad_size), mode="circular"
                )
                image_to_warp.squeeze_(0)  # remove dummy batch dim

            # Adjust source coordinates to account for new padding
            source_coords = (
                source_coords.copy()
                if self.device == "numpy"
                else source_coords.clone()
            )
            source_coords[0, ...] += pad_size

        # Call the warp function
        warped = self._warp_fn(
            image_to_warp,
            source_coords,
            output_shape=self.cartesian_shape,
            order=order,
            mode=mode,
            cval=cval,
            **kwargs,
        )

        return warped

    # ============================================================================
    # CoordinateTransform interface implementation
    # ============================================================================

    def to_transform_space(
        self,
        image: np.ndarray | torch.Tensor,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Warp a Cartesian image to offset polar coordinates.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2-D or batched 3-D Cartesian image.
        order : int, optional
            Spline interpolation order. By default 5.
        mode : str, optional
            Boundary mode passed to the warp function. By default ``"constant"``.
        cval : float, optional
            Constant fill value when ``mode="constant"``. By default 0.0.
        preserve_energy : bool, optional
            Apply Jacobian correction to preserve total energy. By default True.
        **kwargs
            Additional arguments forwarded to the warp function.

        Returns
        -------
        np.ndarray or torch.Tensor
            Warped image in offset polar coordinates.
        """
        self._validate_device(image)

        routed_result = self._apply_recursive_routing(
            self.to_transform_space,
            image,
            order=order,  # type: ignore
            mode=mode,  # type: ignore
            cval=cval,  # type: ignore
            preserve_energy=preserve_energy,  # type: ignore
            **kwargs,
        )
        if routed_result is not None:
            return routed_result

        source_coords = self._compute_cartesian_to_offset_polar_coords()
        warped = self._warp_fn(
            image,
            source_coords,
            output_shape=self.polar_shape,
            order=order,
            mode=mode,
            cval=cval,
            **kwargs,
        )

        if preserve_energy:
            warped = self._apply_jacobian(warped, inverse=False)

        return warped

    def jacobian_correction(self) -> np.ndarray:
        """Per-radial-bin Jacobian (area-correction) factor.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(num_radius,)`` with the square-root of the
            Cartesian area covered by each polar pixel column.
        """
        jac = jacobian_correction_offset_polar(
            self.num_angle, self.num_radius, self.radius
        )
        return np.sqrt(jac).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialise geometric parameters to a JSON-serialisable dict.

        Returns
        -------
        dict[str, Any]
            Keys: ``transform_name``, ``center``, ``radius``,
            ``num_angle``, ``num_radius``, ``height``, ``width``.
        """
        return {
            "transform_name": self.transform_name,
            "center": list(self.center),
            "radius": float(self.radius),
            "num_angle": int(self.num_angle),
            "num_radius": int(self.num_radius),
            "height": int(self.height),
            "width": int(self.width),
        }

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: Literal["numpy"] | str | torch.device = "numpy",
    ) -> "OffsetPolarTransform":
        """Reconstruct an :class:`OffsetPolarTransform` from serialised parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary produced by :meth:`to_dict`.
        device : str | torch.device, optional
            Computational device.  By default ``"numpy"`` (CPU).

        Returns
        -------
        OffsetPolarTransform
        """
        return cls(
            center=tuple(params["center"]),
            radius=float(params["radius"]),
            num_angle=int(params["num_angle"]),
            num_radius=int(params["num_radius"]),
            height=int(params["height"]),
            width=int(params["width"]),
            device=device,
        )
