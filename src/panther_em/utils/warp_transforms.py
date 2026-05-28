"""Offset-polar coordinate transform.

Provides :class:`OffsetPolarTransform`, a concrete
:class:`~panther_em.utils.transform_base.CoordinateTransform` that maps
between Cartesian and offset-polar coordinates. Features:

- Per-instance lazy caching of coordinate grids (inherited from the base
  class); grids are computed once and reused across all calls.
- Per-device caching of GPU tensor copies (populated on first use).
- Optional energy-preserving transforms via Jacobian correction.
- Circular padding at the 0°/360° boundary in the inverse transform.
"""

from typing import Any

import numpy as np

from .coordinates import (
    inverse_cartesian_to_offset_polar_mapping,
    inverse_offset_polar_to_cartesian_mapping,
    jacobian_correction_offset_polar,
)
from .transform_base import CoordinateTransform, get_transform, register_transform


@register_transform
class OffsetPolarTransform(CoordinateTransform):
    """Offset-polar coordinate transform with lazy grid caching.

    The *offset polar* coordinate system is identical to standard polar
    coordinates except that every other radial ring is angularly offset by
    half of the angular step size (``delta_theta / 2``), improving spatial
    coverage without adding too much complexity to coordinate system.

    Implements :class:`~panther_em.utils.transform_base.CoordinateTransform`
    and is registered under the key ``"offset_polar"``.

    Parameters
    ----------
    center : tuple[float, float]
        Center of the transformation (row, col) in Cartesian coordinates.
    radius : float
        Maximum radius, in pixels, for the transformation.
    num_angle : int
        Number of angular samples in the offset polar image.
    num_radius : int
        Number of radial samples in the offset polar image.
    height : int
        Height of the Cartesian image.
    width : int
        Width of the Cartesian image.
    """

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
    ) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_angle: int = 360,
        num_radius: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
    ) -> "OffsetPolarTransform":
        """Convenience constructor from image shape and polar geometry.

        Parameters
        ----------
        image_shape : tuple[int, int]
            ``(height, width)`` of the Cartesian image.
        num_angle : int, optional
            Number of angular samples. Default 360.
        num_radius : int | None, optional
            Number of radial samples. Defaults to ``ceil(radius)``.
        center : tuple[float, float] | None, optional
            ``(row, col)`` centre. Defaults to image centre.
        radius : float | None, optional
            Maximum radius in pixels. Defaults to ``height / 2``.

        Returns
        -------
        OffsetPolarTransform
        """
        height, width = image_shape

        if center is None:
            center = (height / 2 - 0.5, width / 2 - 0.5)

        if radius is None:
            radius = height / 2  # Assuming square images, radius is half the height

        if num_radius is None:
            num_radius = int(np.ceil(radius))

        return get_transform(  # type: ignore[return-value]
            cls,
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=height,
            width=width,
        )

    # ------------------------------------------------------------------ #
    # Shape properties
    # ------------------------------------------------------------------ #

    @property
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the transformed image: ``(num_angle, num_radius)``."""
        return (self.num_angle, self.num_radius)

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the Cartesian image: ``(height, width)``."""
        return (self.height, self.width)

    # ------------------------------------------------------------------ #
    # Coordinate grid computation (called lazily by base-class properties)
    # ------------------------------------------------------------------ #

    def compute_transform_coords(self) -> np.ndarray:
        """Inverse mapping for the Cartesian→polar warp.

        For each output pixel in polar space (radius_idx, angle_idx) returns
        the Cartesian source coordinates to sample.

        Returns
        -------
        np.ndarray
            Shape ``(2, num_angle, num_radius)``, dtype float64.
        """
        t = np.arange(self.num_angle)
        r = np.arange(self.num_radius)
        tt, rr = np.meshgrid(t, r, indexing="ij")
        output_coords = np.stack([rr.flatten(), tt.flatten()], axis=-1)

        source_coords = inverse_offset_polar_to_cartesian_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
        )

        source_coords = source_coords[:, [1, 0]]  # swap to (row, col) ordering
        return source_coords.T.reshape(2, self.num_angle, self.num_radius)

    def compute_cartesian_coords(self) -> np.ndarray:
        """Inverse mapping for the polar→Cartesian warp.

        For each output pixel in Cartesian space (row, col) returns the polar
        source coordinates to sample.

        Returns
        -------
        np.ndarray
            Shape ``(2, width, height)``.
        """
        r = np.arange(self.height)
        c = np.arange(self.width)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        output_coords = np.stack([cc.flatten(), rr.flatten()], axis=-1)

        source_coords = inverse_cartesian_to_offset_polar_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
        )

        source_coords = source_coords[:, [1, 0]]  # swap columns
        return source_coords.T.reshape(2, self.width, self.height)

    def compute_jacobian(self) -> np.ndarray:
        """Compute the Jacobian (area-correction) array.

        Each row is identical (the Jacobian depends only on radius for
        uniform angular spacing), so the 1-D radial correction is broadcast
        to the full ``(num_angle, num_radius)`` shape.

        Returns
        -------
        np.ndarray
            Shape ``(num_angle, num_radius)``, dtype float32.
        """
        jac_1d = jacobian_correction_offset_polar(
            self.num_angle, self.num_radius, self.radius
        )
        jac_sqrt = np.sqrt(jac_1d).astype(np.float32)
        return np.broadcast_to(
            jac_sqrt[np.newaxis, :], (self.num_angle, self.num_radius)
        ).copy()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize geometric parameters to a JSON-compatible dict."""
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
        device: Any = None,  # accepted but ignored; transforms are device-agnostic
    ) -> "OffsetPolarTransform":
        """Reconstruct from serialized parameters, returning a cached instance."""
        return get_transform(  # type: ignore[return-value]
            cls,
            center=tuple(params["center"]),
            radius=float(params["radius"]),
            num_angle=int(params["num_angle"]),
            num_radius=int(params["num_radius"]),
            height=int(params["height"]),
            width=int(params["width"]),
        )
