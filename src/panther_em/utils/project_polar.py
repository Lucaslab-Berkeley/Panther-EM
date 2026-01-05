"""Utility functions to generate polar projections from 3D volumes."""

import torch
import numpy as np
from torch_fourier_slice.project import project_3d_to_2d
from scipy.spatial.transform import Rotation as R

from .polar_transform import warp_polar, warp_polar_inverse


def get_polar_projections_from_volume(
    volume: np.ndarray,
    phi: float | np.ndarray,
    theta: float | np.ndarray,
    psi: float | np.ndarray = 0.0,
    warp_polar_kwargs: dict | None = None,
) -> np.ndarray:
    """Generate 2D projections from a 3D volume in polar coordinates.

    Parameters
    ----------
    volume : ndarray
        Cubic 3D numpy array to generate 2D projections from.
    phi : float | ndarray
        Rotation angle(s) for projections, in degrees. In ZYZ Euler angle format.
    theta : float | ndarray
        Rotation angle(s) for projections, in degrees. In ZYZ Euler angle format.
    psi : float | ndarray, optional
        Rotation angle(s) for projections, in degrees. In ZYZ Euler angle format.
        Default is 0.0.
    warp_polar_kwargs : dict | None, optional
        Additional keyword arguments for the polar warping function.
        Default is None.

    Returns
    -------
    projections_polar : ndarray
        2D projections in polar coordinates. Shape is
        (num_projections, num_radial_pixels, num_angular_pixels).
    """
    # Broadcast angles to arrays of same shape ensuring same number of angles for each
    phi = np.asarray(phi)
    theta = np.asarray(theta)
    psi = np.asarray(psi)

    angles_shape = np.broadcast(phi, theta, psi).shape

    print("DEBUG - angles_shape:", angles_shape)

    phi = np.broadcast_to(phi, angles_shape).ravel()
    theta = np.broadcast_to(theta, angles_shape).ravel()
    psi = np.broadcast_to(psi, angles_shape).ravel()

    # Default warp polar kwargs
    if warp_polar_kwargs is None:
        warp_polar_kwargs = {
            "scaling": "linear",
            "mode": "wrap",  # Angular dimension is periodic
        }

    # Convert ZYZ euler angles into rotation matrices
    rot = R.from_euler("ZYZ", np.column_stack((phi, theta, psi)), degrees=True)
    rot_matrices = torch.from_numpy(rot.as_matrix().astype(np.float32))

    projections = project_3d_to_2d(
        volume=torch.from_numpy(volume),
        rotation_matrices=rot_matrices,
        pad_factor=2.0,
        fftfreq_max=0.5,
        zyx_matrices=False,
    )
    if projections.ndim == 2:
        projections = projections.unsqueeze(0)  # Add batch dimension

    projections = projections.numpy()

    # Warp each projection to polar coordinates
    projections_polar = []
    for i in range(projections.shape[0]):
        projections_polar.append(warp_polar(projections[i], **warp_polar_kwargs))
    projections_polar = np.array(projections_polar)

    return projections_polar
