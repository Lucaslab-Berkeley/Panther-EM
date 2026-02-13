"""Utility functions to generate polar projections from 3D volumes."""

import numpy as np
import roma
import torch
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice.project import project_3d_to_2d

from .warp_transforms import warp_offset_polar


def _numpy_polar_projections(
    volume: np.ndarray,
    phi: float | np.ndarray,
    theta: float | np.ndarray,
    psi: float | np.ndarray,
    num_angle: int,
    num_radius: int,
    warp_polar_kwargs: dict,
) -> np.ndarray:
    """Generate 2D projections in polar coordinates from a 3D volume using numpy."""
    phi = np.asarray(phi)
    theta = np.asarray(theta)
    psi = np.asarray(psi)

    angles_shape = np.broadcast(phi, theta, psi).shape

    phi = np.broadcast_to(phi, angles_shape).ravel()
    theta = np.broadcast_to(theta, angles_shape).ravel()
    psi = np.broadcast_to(psi, angles_shape).ravel()

    # Convert ZYZ euler angles into rotation matrices
    rot = R.from_euler("ZYZ", np.column_stack((phi, theta, psi)), degrees=True)
    rot_matrices = rot.as_matrix().astype(np.float32)

    projections = project_3d_to_2d(
        volume=torch.from_numpy(volume),
        rotation_matrices=torch.from_numpy(rot_matrices),
        pad_factor=2.0,
        fftfreq_max=0.5,
        zyx_matrices=False,
    )

    # Constants for the warp function
    center = (projections.shape[-2] / 2, projections.shape[-2] / 2)
    radius = projections.shape[-1] / 2  # Assuming square projections

    projections_polar = warp_offset_polar(
        projections,
        num_angle=num_angle,
        num_radius=num_radius,
        center=center,
        radius=radius,
        **warp_polar_kwargs,
    )

    return projections_polar


def _pytorch_polar_projections(
    volume: torch.Tensor,
    phi: float | torch.Tensor,
    theta: float | torch.Tensor,
    psi: float | torch.Tensor,
    num_angle: int,
    num_radius: int,
    warp_polar_kwargs: dict,
) -> torch.Tensor:
    """Generate 2D projections in polar coordinates from a 3D volume using PyTorch."""
    device = volume.device

    if isinstance(phi, (int, float)):
        phi = torch.tensor(phi, device=device, dtype=torch.float32)
    if isinstance(theta, (int, float)):
        theta = torch.tensor(theta, device=device, dtype=torch.float32)
    if isinstance(psi, (int, float)):
        psi = torch.tensor(psi, device=device, dtype=torch.float32)

    angles_shape = torch.broadcast_shapes(phi.shape, theta.shape, psi.shape)

    phi = phi.expand(angles_shape).reshape(-1)
    theta = theta.expand(angles_shape).reshape(-1)
    psi = psi.expand(angles_shape).reshape(-1)

    # Send to same device as volume
    phi = phi.to(device)
    theta = theta.to(device)
    psi = psi.to(device)

    # Convert ZYZ euler angles into rotation matrices
    rot_matrices = roma.euler_to_rotmat(
        "ZYZ", angles=torch.column_stack((phi, theta, psi)), degrees=True
    )

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rot_matrices,
        pad_factor=2.0,
        fftfreq_max=0.5,
        zyx_matrices=False,
    )

    # Constants for the warp function
    center = (projections.shape[-2] / 2, projections.shape[-2] / 2)
    radius = projections.shape[-1] / 2  # Assuming square projections

    projections_polar = warp_offset_polar(
        projections,
        num_angle=num_angle,
        num_radius=num_radius,
        center=center,
        radius=radius,
        **warp_polar_kwargs,
    )

    return projections_polar


def get_polar_projections_from_volume(
    volume: np.ndarray | torch.Tensor,
    phi: float | np.ndarray | torch.Tensor,
    theta: float | np.ndarray | torch.Tensor,
    psi: float | np.ndarray | torch.Tensor = 0.0,
    num_angle: int = 360,
    num_radius: int = 256,
    warp_polar_kwargs: dict | None = None,
) -> np.ndarray | torch.Tensor:
    """Generate 2D projections from a 3D volume in offset polar coordinates.

    Uses the offset polar coordinate system where alternating radial rings are
    shifted by half an angular step (Δθ/2) to provide better spatial coverage.

    Notes
    -----
    - This function supports both numpy arrays and PyTorch tensors as inputs, but all
    arguments must be of the same type (except for float inputs). Otherwise, a
    TypeError will be raised.
    - If torch tensors are passed, then all tensors (besides float inputs) must be on
    the same device, and all computation will happen on that device.
    - The returned type will match the input type (numpy array or PyTorch tensor), and
    if PyTorch tensors are used, the output will be on the same device as the input
    tensors.

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
    num_angle : int, optional
        Number of angular samples in the polar projection. Default is 360.
    num_radius : int | None, optional
        Number of radial samples in the polar projection. If None, computed
        automatically from the projection size. Default is None.
    warp_polar_kwargs : dict | None, optional
        Additional keyword arguments for the polar warping function.
        Default is None.

    Returns
    -------
    projections_polar : ndarray
        2D projections in offset polar coordinates. Shape is
        (num_projections, num_angle, num_radius).
    """
    # Check that all inputs are of the same type (except for float inputs)
    input_types = {
        type(arg)
        for arg in [volume, phi, theta, psi]
        if not isinstance(arg, (int, float))
    }
    if len(input_types) > 1:
        raise TypeError(
            "All inputs (except for float inputs) must be of the same type. "
            f"Got types: {input_types}"
        )

    if warp_polar_kwargs is None:
        warp_polar_kwargs = {}

    # Call the appropriate implementation based on the input type
    if isinstance(volume, torch.Tensor):
        return _pytorch_polar_projections(
            volume=volume,
            phi=phi,
            theta=theta,
            psi=psi,
            num_angle=num_angle,
            num_radius=num_radius,
            warp_polar_kwargs=warp_polar_kwargs,
        )
    elif isinstance(volume, np.ndarray):
        return _numpy_polar_projections(
            volume=volume,
            phi=phi,
            theta=theta,
            psi=psi,
            num_angle=num_angle,
            num_radius=num_radius,
            warp_polar_kwargs=warp_polar_kwargs,
        )
    else:
        raise TypeError(
            f"Unsupported input type: {type(volume)}. "
            "Expected numpy array or PyTorch tensor."
        )

    # Broadcast angles to arrays of same shape ensuring same number of angles for each
    phi = np.asarray(phi)
    theta = np.asarray(theta)
    psi = np.asarray(psi)

    angles_shape = np.broadcast(phi, theta, psi).shape

    phi = np.broadcast_to(phi, angles_shape).ravel()
    theta = np.broadcast_to(theta, angles_shape).ravel()
    psi = np.broadcast_to(psi, angles_shape).ravel()

    # Default warp polar kwargs
    if warp_polar_kwargs is None:
        warp_polar_kwargs = {}

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

    # Constants for the warp function
    center = (projections.shape[1] / 2, projections.shape[2] / 2)
    radius = projections.shape[1] / 2  # Assuming square projections

    # Warp each projection to offset polar coordinates
    projections_polar = []
    for i in range(projections.shape[0]):
        proj_polar = warp_offset_polar(
            projections[i],
            num_angle=num_angle,
            num_radius=num_radius,
            center=center,
            radius=radius,
            **warp_polar_kwargs,
        )
        projections_polar.append(proj_polar)

    projections_polar = np.array(projections_polar)

    return projections_polar
