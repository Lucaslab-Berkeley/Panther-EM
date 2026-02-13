"""Coordinate transformation functions for the `scikit-image` transformations."""

import numpy as np


def forward_cartesian_to_offset_polar_mapping(
    input_coords: np.ndarray,  # (M, 2) - (row, col)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """The forward mapping function to take cartesian coords to 'offset polar' coords.

    The 'offset polar' coordinate system is similar to the normal polar coordinate
    system (r, theta) except every other radial ring is offset by half of
    the angular step size.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    polar_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in offset polar space.
    """
    row = input_coords[:, 0] - center[0]
    col = input_coords[:, 1] - center[1]

    # Convert to polar coordinates
    radius = np.sqrt(row**2 + col**2)
    angle = np.arctan2(row, col)
    angle = angle % (2 * np.pi)  # Map to range [0, 2*pi]

    # Convert to indices
    radius_idx = (radius / max_radius) * num_radius
    angle_idx = (angle / (2 * np.pi)) * num_angle

    # Calculate angular increment, and offset odd rings
    delta_theta = (2 * np.pi) / num_angle
    ring_idx = np.floor(radius_idx).astype(int)

    # Apply offset: subtract half step for odd rings (to reverse the offset)
    angle_offset = (ring_idx % 2) * (delta_theta / 2)
    angle_adjusted = angle - angle_offset

    # Wrap around if necessary
    angle_adjusted = np.where(
        angle_adjusted < 0, angle_adjusted + 2 * np.pi, angle_adjusted
    )
    angle_adjusted = np.where(
        angle_adjusted >= 2 * np.pi, angle_adjusted - 2 * np.pi, angle_adjusted
    )

    # Convert adjusted angle to index
    angle_idx = (angle_adjusted / (2 * np.pi)) * num_angle

    return np.column_stack((radius_idx, angle_idx))


def forward_offset_polar_to_cartesian_mapping(
    input_coords: np.ndarray,  # (M, 2) - (radius_idx, angle_idx)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Forward mapping: Offset Polar coordinates → Cartesian coordinates.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in offset polar space.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    cartesian_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    """
    radius_idx = input_coords[:, 0]
    angle_idx = input_coords[:, 1]

    # Convert indices to actual values
    angle = (angle_idx / num_angle) * (2 * np.pi)
    radius = (radius_idx / num_radius) * max_radius

    # Apply offset for odd rings
    ring_idx = np.floor(radius_idx).astype(int)
    delta_theta = (2 * np.pi) / num_angle
    angle_offset = (ring_idx % 2) * (delta_theta / 2)
    angle_with_offset = angle + angle_offset

    # Convert to cartesian
    row = radius * np.sin(angle_with_offset) + center[0]
    col = radius * np.cos(angle_with_offset) + center[1]

    return np.column_stack((row, col))


def inverse_offset_polar_to_cartesian_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping for warping cartesian image to offset polar space.

    Given output coordinates in offset polar space, returns the corresponding
    input coordinates in cartesian space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (offset polar) image.
        col corresponds to radius, row corresponds to angle.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system in the input cartesian
        image.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (cartesian) image.
    """
    # output_coords[:, 0] is the radius index (col in polar image)
    # output_coords[:, 1] is the angle index (row in polar image)

    radius_idx = output_coords[:, 0]
    angle_idx = output_coords[:, 1]

    # Convert indices to actual angle and radius
    angle = (angle_idx / num_angle) * (2 * np.pi)
    radius = (radius_idx / num_radius) * max_radius

    # Determine which ring we're in (for offset calculation)
    ring_idx = np.floor(radius_idx).astype(int)

    # Apply offset for odd rings: shift by half angular step
    delta_theta = (2 * np.pi) / num_angle
    angle_offset = (ring_idx % 2) * (delta_theta / 2)
    angle_with_offset = angle + angle_offset

    # # NOTE: Testing to remove the offset
    # angle_with_offset = angle

    # Convert from polar to cartesian
    row = radius * np.sin(angle_with_offset) + center[0]
    col = radius * np.cos(angle_with_offset) + center[1]

    # Return as (col, row) for warp function
    coords = np.column_stack((col, row))
    return coords


def inverse_cartesian_to_offset_polar_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping for warping offset polar image back to cartesian space.

    Given output coordinates in cartesian space, returns the corresponding
    input coordinates in offset polar space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (cartesian) image.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (offset polar) image.
        col corresponds to radius, row corresponds to angle.
    """
    col = output_coords[:, 0]
    row = output_coords[:, 1]

    # Center the coordinates
    dc = col - center[1]
    dr = row - center[0]

    # Convert to polar
    radius = np.sqrt(dr**2 + dc**2)
    angle = np.arctan2(dr, dc)

    # Normalize angle to [0, 2*pi]
    angle = angle % (2 * np.pi)

    # Convert to indices
    radius_idx = (radius / max_radius) * num_radius

    # Determine which ring we're in for offset calculation
    ring_idx = np.floor(radius_idx).astype(int)

    # Remove the offset that was applied to odd rings
    delta_theta = (2 * np.pi) / num_angle
    angle_offset = (ring_idx % 2) * (delta_theta / 2)
    angle_adjusted = angle - angle_offset

    # Wrap around if necessary
    angle_adjusted = np.where(
        angle_adjusted < 0, angle_adjusted + 2 * np.pi, angle_adjusted
    )
    angle_adjusted = np.where(
        angle_adjusted >= 2 * np.pi, angle_adjusted - 2 * np.pi, angle_adjusted
    )

    # # NOTE: Testing to remove the offset
    # angle_adjusted = angle

    # Convert to angle index
    angle_idx = (angle_adjusted / (2 * np.pi)) * num_angle

    # Return as (col, row) in polar space = (radius_idx, angle_idx)
    coords = np.column_stack((radius_idx, angle_idx))
    return coords
