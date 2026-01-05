"""
The `scikit-image` package includes a function for a forward cartesian-to-polar
transformation on 2D images, but the inverse polar-to-cartesian transformation is
absent. This module provides the inverse transformation as well as namespace imports
for the `scikit-image` polar transformation functions.
"""

from typing import Literal

import numpy as np
from skimage.transform import warp, warp_polar
from skimage._shared.utils import safe_as_int


def _linear_cartesian_mapping(
    output_coords: np.ndarray,
    k_angle: float,
    k_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping: convert from polar to cartesian coordinates.

    Parameters
    ----------
    output_coords : ndarray
        (M, 2) array of (col, row) coordinates in the output (cartesian) image.
    k_angle : float
        Scaling factor between rows and angles: k_angle = num_rows / (2 * pi).
    k_radius : float
        Scaling factor between columns and radius: k_radius = num_cols / max_radius.
    center : tuple[float, float]
        Center of the polar transformation in the image (x_center, y_center).

    Returns
    -------
    coords : ndarray
        (M, 2) array of (col, row) coordinates in the input (polar) image.
    """
    rr = output_coords[:, 1] - center[1]
    cc = output_coords[:, 0] - center[0]
    radius = np.sqrt(rr**2 + cc**2)
    angle = np.arctan2(rr, cc)

    # Map back into polar-space pixel coordinates
    r_polar = radius * k_radius
    theta_polar = (angle % (2 * np.pi)) * k_angle
    coords = np.column_stack((r_polar, theta_polar))

    return coords


def _log_cartesian_mapping(
    output_coords: np.ndarray,
    k_angle: float,
    k_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping: convert from log-polar to cartesian coordinates.

    Parameters
    ----------
    output_coords : ndarray
        (M, 2) array of (col, row) coordinates in the output (cartesian) image.
    k_angle : float
        Scaling factor between rows and angles: k_angle = num_rows / (2 * pi).
    k_radius : float
        Scaling factor between columns and log-radius: k_radius = num_cols / log(max_radius).
    center : tuple[float, float]
        Center of the log-polar transformation in the image (x_center, y_center).

    Returns
    -------
    coords : ndarray
        (M, 2) array of (col, row) coordinates in the input (log-polar) image.
    """
    rr = output_coords[:, 1] - center[1]
    cc = output_coords[:, 0] - center[0]
    radius = np.sqrt(rr**2 + cc**2)
    angle = np.arctan2(rr, cc)

    # Map back into log-polar-space pixel coordinates
    r_log_polar = np.log(radius + 1e-10) * k_radius  # Small const to avoid log(0)
    theta_log_polar = (angle % (2 * np.pi)) * k_angle
    coords = np.column_stack((r_log_polar, theta_log_polar))

    return coords


def warp_polar_inverse(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    *args,
    radius: float | None = None,
    output_shape: tuple[int, int] | None = None,
    scaling: Literal["linear", "log"] = "linear",
    multichannel: bool = False,
    **kwargs,
) -> np.ndarray:
    """Perform an inverse polar or log-polar transformation on a 2D image.

    Parameters
    ----------
    image : ndarray
        The input image in polar (or log-polar) coordinates.
    center : tuple[float, float] | None, optional
        The center of the polar transformation in the the cartesian image
        (x_center, y_center). If None, assumed to be image center (h / 2, w / 2).
        Default is None.
    *args
        Additional positional arguments to pass to `skimage.transform.warp`.
    output_shape : tuple[int, int] | None, optional
        The shape (rows, cols) of the output cartesian image. If None, it is inferred
        from the number of radial pixels (columns) in the input polar image.
        Default is None.
    scaling : {'linear', 'log'}, optional
        The type of scaling used in the polar transformation. Default is 'linear'.
    multichannel : bool, optional
        Whether the input image is multichannel (e.g., RGB). Default is False.
    **kwargs
        Additional keyword arguments to pass to `skimage.transform.warp`.

    Returns
    -------
    warped : ndarray
        The inverse transformed cartesian image.
    """
    if image.ndim != 2 and not multichannel:
        raise ValueError("Input image must be 2D for single-channel images.")
    if image.ndim != 3 and multichannel:
        raise ValueError("Input image must be 3D for multichannel images.")

    # Infer the image center, if not provided
    if center is None:
        if output_shape is not None:
            center = (np.array(output_shape)[:2] / 2) - 0.5
        else:
            raise ValueError("One of 'center' or 'output_shape' must be provided.")

    # Infer radius, if not provided
    if radius is None:
        if output_shape is None:
            raise ValueError("One of 'radius' or 'output_shape' must be provided.")
        w, h = np.array(output_shape)[:2]
        radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

    # Infer output shape, if not provided
    if output_shape is None:
        width = int(np.ceil(2 * radius))
        height = width
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)

    height_polar, width_polar = image.shape[:2]

    # Setup mapping parameters and functions
    k_angle = height_polar / (2 * np.pi)
    if scaling == "linear":
        k_radius = width_polar / radius
        mapping_function = _linear_cartesian_mapping
    elif scaling == "log":
        k_radius = width_polar / np.log(radius)
        mapping_function = _log_cartesian_mapping
    else:
        raise ValueError("Scaling must be either 'linear' or 'log'.")

    warp_args = {"k_angle": k_angle, "k_radius": k_radius, "center": center}
    warped = warp(
        image,
        mapping_function,
        map_args=warp_args,
        output_shape=output_shape,
        **kwargs,
    )

    return warped
