"""Custom warp transformation functions.

The `scikit-image` package includes a function for a forward cartesian-to-polar
transformation on 2D images, but the inverse polar-to-cartesian transformation is
absent. This module provides the inverse transformation as well as namespace imports
for the `scikit-image` polar transformation functions.
"""

from typing import Any

import numpy as np
from skimage.transform import warp

from .coordinates import (
    inverse_cartesian_to_offset_polar_mapping,
    inverse_offset_polar_to_cartesian_mapping,
)

_TRANSFORMER_CACHE = {}


def _get_bhw_of_image(image: np.ndarray) -> tuple[int | None, int, int]:
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
    image: np.ndarray,
    num_angle: int,
    num_radius: int,
    center: tuple[float, float],
    radius: float,
    **kwargs: dict[Any, Any],
) -> np.ndarray:
    """Wrapper around the OffsetPolarTransform class for the forward transformation."""
    _, h, w = _get_bhw_of_image(image)

    # First, check if we already have a cached transformer for these parameters
    key = (center, radius, num_angle, num_radius, (h, w))
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = OffsetPolarTransform(
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=h,  # Gets inferred from the image shape
            width=w,  # Gets inferred from the image shape
        )

    # Obtain the transformer from the cache and call the to_offset_polar method
    transformer = _TRANSFORMER_CACHE[key]
    return transformer.to_offset_polar(image, **kwargs)


def warp_offset_polar_inverse(
    image: np.ndarray,
    height: int,
    width: int,
    center: tuple[float, float],
    radius: float,
    **kwargs: dict[Any, Any],
) -> np.ndarray:
    """Wrapper around the OffsetPolarTransform class for the inverse transformation."""
    _, na, nr = _get_bhw_of_image(image)

    # First, check if we already have a cached transformer for these parameters
    key = (center, radius, na, nr, (height, width))
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = OffsetPolarTransform(
            center=center,
            radius=radius,
            num_angle=na,  # Gets inferred from the image shape
            num_radius=nr,  # Gets inferred from the image shape
            height=height,
            width=width,
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

    Methods
    -------
    from_image(image_shape, num_angle=360, num_radius=None, center=None, radius=None)
        Class method to create an OffsetPolarTransform instance from image shape and
        other associated parameters.
    to_offset_polar(image, order=5, mode='symmetric', **kwargs)
        Warp a cartesian image to offset polar coordinates. Accepts 2D or 3D (batched)
        images. By default, uses the maximum order-5 interpolation and symmetric
        padding. Additional keyword arguments are passed to `skimage.transform.warp`
        function.
    to_cartesian(image, order=5, mode='symmetric', **kwargs)
        Warp an offset polar image to cartesian coordinates. Accepts 2D or 3D (batched)
        images. By default, uses the maximum order-5 interpolation and symmetric
        padding. Additional keyword arguments are passed to `skimage.transform.warp`
        function.
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
    ) -> None:
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width

        # Cache the coordinate mappings for efficiency
        self._cartesian_to_offset_polar_coords = None
        self._offset_polar_to_cartesian_coords = None

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_angle: int = 360,
        num_radius: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
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
        )

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the cartesian image."""
        return (self.height, self.width)

    @property
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the offset polar image."""
        return (self.num_angle, self.num_radius)

    def _compute_offset_polar_to_cartesian_coords(self) -> np.ndarray:
        """Inverse coordinate mapping from offset polar to cartesian coordinates."""
        if self._offset_polar_to_cartesian_coords is None:
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
            self._offset_polar_to_cartesian_coords = source_coords

        return self._offset_polar_to_cartesian_coords

    def _compute_cartesian_to_offset_polar_coords(self) -> np.ndarray:
        """Inverse coordinate mapping from cartesian to offset polar coordinates."""
        if self._cartesian_to_offset_polar_coords is None:
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
            self._cartesian_to_offset_polar_coords = source_coords

        return self._cartesian_to_offset_polar_coords

    def to_offset_polar(
        self,
        image: np.ndarray,
        order: int = 5,
        mode: str = "symmetric",
        **kwargs: dict[Any, Any],
    ) -> np.ndarray:
        """Warp a cartesian image to offset polar coordinates."""
        # Handle a batch dimension
        if image.ndim == 3:
            return np.stack(
                [self.to_offset_polar(img, order, mode, **kwargs) for img in image]
            )

        if image.ndim != 2:
            raise ValueError(
                f"Input image must be 2D or 3D (batched), got {image.ndim}D."
            )

        # Handle case where input is complex - treat real and imag parts separately
        if np.iscomplexobj(image):
            real_part = self.to_offset_polar(image.real, order, mode, **kwargs)
            imag_part = self.to_offset_polar(image.imag, order, mode, **kwargs)
            return real_part + 1j * imag_part

        # Get the source coordinates, if already computed, and call warp
        source_coords = self._compute_cartesian_to_offset_polar_coords()
        warped = warp(
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
        image: np.ndarray,
        order: int = 5,
        mode: str = "symmetric",
        **kwargs: dict[Any, Any],
    ) -> np.ndarray:
        """Warp an offset polar image to cartesian coordinates."""
        # Handle a batch dimension
        if image.ndim == 3:
            return np.stack(
                [self.to_cartesian(img, order, mode, **kwargs) for img in image]
            )

        if image.ndim != 2:
            raise ValueError(
                f"Input image must be 2D or 3D (batched), got {image.ndim}D."
            )

        # Handle case where input is complex - treat real and imag parts separately
        if np.iscomplexobj(image):
            real_part = self.to_cartesian(image.real, order, mode, **kwargs)
            imag_part = self.to_cartesian(image.imag, order, mode, **kwargs)
            return real_part + 1j * imag_part

        # Get the source coordinates, if already computed, and call warp
        source_coords = self._compute_offset_polar_to_cartesian_coords()
        warped = warp(
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
