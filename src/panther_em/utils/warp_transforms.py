"""Custom warp transformation functions.

The `scikit-image` package includes a function for a forward cartesian-to-polar
transformation on 2D images, but the inverse polar-to-cartesian transformation is
absent. This module provides the inverse transformation as well as namespace imports
for the `scikit-image` polar transformation functions.
"""

import numpy as np
from skimage._shared.utils import safe_as_int
from skimage.transform import warp

from .coordinates import (
    inverse_cartesian_to_offset_polar_mapping,
    inverse_offset_polar_to_cartesian_mapping,
)


def warp_offset_polar(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    output_shape: tuple[int, int] | None = None,
    num_angle: int | None = None,
    num_radius: int | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """Transform a 2D cartesian image to offset polar coordinates.

    The offset polar coordinate system shifts alternating radial rings by half
    an angular step (Δθ/2) to provide better spatial coverage than standard
    polar coordinates.

    Parameters
    ----------
    image : ndarray
        Input image in cartesian coordinates. Can be 2D (H, W) or 3D (B, H, W)
        where B is the batch dimension.
    center : tuple[float, float] | None, optional
        Center of the polar transformation (row, col). If None, defaults to
        the image center. Default is None.
    radius : float | None, optional
        Maximum radius for the transformation. If None, computed as the
        distance from center to the farthest corner. Default is None.
    output_shape : tuple[int, int] | None, optional
        Shape of output polar image (num_angle, num_radius). If None, inferred
        from num_angle and num_radius parameters. Default is None.
    num_angle : int | None, optional
        Number of angular samples. If None and output_shape is None, defaults
        to 360. Default is None.
    num_radius : int | None, optional
        Number of radial samples. If None and output_shape is None, computed
        from radius. Default is None.
    **kwargs
        Additional keyword arguments passed to `skimage.transform.warp`.

    Returns
    -------
    warped : ndarray
        Transformed image in offset polar coordinates. If input is 2D, output is
        (num_angle, num_radius). If input is 3D, output is (B, num_angle, num_radius).

    Examples
    --------
    >>> from skimage import data
    >>> image = data.camera()
    >>> polar = warp_offset_polar(image, num_angle=360, num_radius=256)
    >>> # Batch processing
    >>> batch = np.stack([data.camera(), data.camera()])  # (2, 512, 512)
    >>> polar_batch = warp_offset_polar(
    ...     batch, num_angle=360, num_radius=256
    ... )  # (2, 360, 256)
    """
    # Handle batch dimension
    if image.ndim == 3:
        # Batched input: (B, H, W)
        return np.stack(
            [
                warp_offset_polar(
                    img, center, radius, output_shape, num_angle, num_radius, **kwargs
                )
                for img in image
            ]
        )

    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D or 3D (batched), got {image.ndim}D.")

    # Infer center if not provided
    if center is None:
        center = (image.shape[0] / 2 - 0.5, image.shape[1] / 2 - 0.5)

    # Infer radius if not provided
    if radius is None:
        radius = np.sqrt((image.shape[0] / 2) ** 2 + (image.shape[1] / 2) ** 2)

    # Determine output shape
    if output_shape is not None:
        num_angle_out, num_radius_out = output_shape
    else:
        if num_angle is None:
            num_angle_out = 360
        else:
            num_angle_out = num_angle

        if num_radius is None:
            num_radius_out = int(np.ceil(radius))
        else:
            num_radius_out = num_radius

        output_shape = (num_angle_out, num_radius_out)

    # Create inverse mapping function
    def inverse_mapping(output_coords: np.ndarray) -> np.ndarray:
        """Inverse mapping from offset polar to cartesian coordinates."""
        return inverse_offset_polar_to_cartesian_mapping(
            output_coords,
            num_angle=output_shape[0],
            num_radius=output_shape[1],
            max_radius=radius,
            center=center,
        )

    # Perform warp with high-quality interpolation
    warped = warp(
        image,
        inverse_mapping,
        output_shape=output_shape,
        order=5,
        mode="symmetric",
        preserve_range=True,
        **kwargs,
    )

    return warped


def warp_offset_polar_inverse(
    image: np.ndarray,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    output_shape: tuple[int, int] | None = None,
    **kwargs: dict,
) -> np.ndarray:
    """Transform a 2D offset polar image back to cartesian coordinates.

    This is the inverse of `warp_offset_polar`, transforming from offset polar
    coordinates back to cartesian coordinates.

    Parameters
    ----------
    image : ndarray
        Input image in offset polar coordinates. Can be 2D (num_angle, num_radius)
        or 3D (B, num_angle, num_radius) where B is the batch dimension.
    center : tuple[float, float] | None, optional
        Center of the polar transformation in cartesian space (row, col). If None
        and output_shape is provided, defaults to the output image center.
        Default is None.
    radius : float | None, optional
        Maximum radius used in the forward transformation. If None and output_shape
        is provided, computed from output shape. Default is None.
    output_shape : tuple[int, int] | None, optional
        Shape of output cartesian image (rows, cols). If None, inferred from
        the radial dimension of the input polar image. Default is None.
    **kwargs
        Additional keyword arguments passed to `skimage.transform.warp`.

    Returns
    -------
    warped : ndarray
        Transformed image in cartesian coordinates. If input is 2D, output is
        (H, W). If input is 3D, output is (B, H, W).

    Examples
    --------
    >>> polar_image = warp_offset_polar(cartesian_image)
    >>> reconstructed = warp_offset_polar_inverse(polar_image)
    >>> # Batch processing
    >>> polar_batch = warp_offset_polar(batch)
    >>> reconstructed_batch = warp_offset_polar_inverse(polar_batch)
    """
    # Handle batch dimension
    if image.ndim == 3:
        # Batched input: (B, num_angle, num_radius)
        return np.stack(
            [
                warp_offset_polar_inverse(img, center, radius, output_shape, **kwargs)
                for img in image
            ]
        )

    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D or 3D (batched), got {image.ndim}D.")

    num_angle, num_radius = image.shape

    # Infer output shape if not provided
    if output_shape is None:
        # Use radial dimension to estimate cartesian size
        side_length = 2 * num_radius
        output_shape = (side_length, side_length)

    output_shape = safe_as_int(output_shape)

    # Infer center if not provided
    if center is None:
        center = (output_shape[0] / 2 - 0.5, output_shape[1] / 2 - 0.5)

    # Infer radius if not provided
    if radius is None:
        radius = np.sqrt((output_shape[0] / 2) ** 2 + (output_shape[1] / 2) ** 2)

    # Create inverse mapping function
    def inverse_mapping(output_coords: np.ndarray) -> np.ndarray:
        """Inverse mapping from offset polar to cartesian coordinates."""
        return inverse_cartesian_to_offset_polar_mapping(
            output_coords,
            num_angle=num_angle,
            num_radius=num_radius,
            max_radius=radius,
            center=center,
        )

    # Perform warp with high-quality interpolation
    warped = warp(
        image,
        inverse_mapping,
        output_shape=output_shape,
        order=5,
        mode="symmetric",
        preserve_range=True,
        **kwargs,
    )

    return warped


# def warp_polar_inverse(
#     image: np.ndarray,
#     center: tuple[float, float] | None = None,
#     *args,
#     radius: float | None = None,
#     output_shape: tuple[int, int] | None = None,
#     scaling: Literal["linear", "log"] = "linear",
#     multichannel: bool = False,
#     **kwargs,
# ) -> np.ndarray:
#     """Perform an inverse polar or log-polar transformation on a 2D image.

#     Parameters
#     ----------
#     image : ndarray
#         The input image in polar (or log-polar) coordinates.
#     center : tuple[float, float] | None, optional
#         The center of the polar transformation in the the cartesian image
#         (x_center, y_center). If None, assumed to be image center (h / 2, w / 2).
#         Default is None.
#     *args
#         Additional positional arguments to pass to `skimage.transform.warp`.
#     output_shape : tuple[int, int] | None, optional
#         The shape (rows, cols) of the output cartesian image. If None, it is inferred
#         from the number of radial pixels (columns) in the input polar image.
#         Default is None.
#     scaling : {'linear', 'log'}, optional
#         The type of scaling used in the polar transformation. Default is 'linear'.
#     multichannel : bool, optional
#         Whether the input image is multichannel (e.g., RGB). Default is False.
#     **kwargs
#         Additional keyword arguments to pass to `skimage.transform.warp`.

#     Returns
#     -------
#     warped : ndarray
#         The inverse transformed cartesian image.
#     """
#     raise NotImplementedError("Inverse polar transformation is not yet implemented.")

#     # if image.ndim != 2 and not multichannel:
#     #     raise ValueError("Input image must be 2D for single-channel images.")
#     # if image.ndim != 3 and multichannel:
#     #     raise ValueError("Input image must be 3D for multichannel images.")

#     # # Infer the image center, if not provided
#     # if center is None:
#     #     if output_shape is not None:
#     #         center = (np.array(output_shape)[:2] / 2) - 0.5
#     #     else:
#     #         raise ValueError("One of 'center' or 'output_shape' must be provided.")

#     # # Infer radius, if not provided
#     # if radius is None:
#     #     if output_shape is None:
#     #         raise ValueError("One of 'radius' or 'output_shape' must be provided.")
#     #     w, h = np.array(output_shape)[:2]
#     #     radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

#     # # Infer output shape, if not provided
#     # if output_shape is None:
#     #     width = int(np.ceil(2 * radius))
#     #     height = width
#     #     output_shape = (height, width)
#     # else:
#     #     output_shape = safe_as_int(output_shape)

#     # height_polar, width_polar = image.shape[:2]

#     # # Setup mapping parameters and functions
#     # k_angle = height_polar / (2 * np.pi)
#     # if scaling == "linear":
#     #     k_radius = width_polar / radius
#     #     mapping_function = _linear_cartesian_mapping
#     # elif scaling == "log":
#     #     k_radius = width_polar / np.log(radius)
#     #     mapping_function = _log_cartesian_mapping
#     # else:
#     #     raise ValueError("Scaling must be either 'linear' or 'log'.")

#     # warp_args = {"k_angle": k_angle, "k_radius": k_radius, "center": center}
#     # warped = warp(
#     #     image,
#     #     mapping_function,
#     #     map_args=warp_args,
#     #     output_shape=output_shape,
#     #     **kwargs,
#     # )

#     # return warped
