"""Pipelined generation and transforms of projections in polar coordinates.

For the on-the-fly projection generation, groups of projections can be generated and
transformed together in batches on the GPU to not exceed GPU memory limits. Data for the
SVD step are generally too large to fit into GPU memory, so after each batch execution
data is moved back to CPU and stored. Retains GPU acceleration for many parallelizable
steps.

1. Transform batch of (B, 3) ZYZ Euler angles (phi, theta, psi) into rotation matrices.
2. Take Fourier slices from a RFFT'd 3D volume using the rotation matrices. Produces
   (B, h, w // 2 + 1) Fourier-space projections.
3. Apply defocus offsets in Fourier space to get (f, B, h, w // 2 + 1) defocus-offset
   Fourier-space projections.
4. Inverse FFT into set of (f, B, h, w) defocus-offset spatial-space projections.
5. Warp each projection in the batch into polar coordinates, producing
   (f, B, num_angle, num_radius) polar projections.
6. Complex FFT along the angular dimension (axis=1) to get (f, B, num_angle, num_radius)
   Fourier-space polar projections.
7. Move batch of Fourier-space polar projections back to CPU and store for SVD step.
"""

import roma
import torch
import torch.nn.functional as F
import tqdm
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_slice.volume_utils import compute_cube_face_averages

from panther_em.utils.warp_transforms import warp_offset_polar


def precompute_volume_dft(
    volume: torch.Tensor,
    pad_factor: float = 2.0,
) -> tuple[torch.Tensor, float, int]:
    """Precompute the 3D RFFT of a volume with padding.

    Parameters
    ----------
    volume : torch.Tensor
        `(d, d, d)` cubic volume.
    pad_factor : float
        Padding factor for the volume. Default is 2.0.

    Returns
    -------
    dft : torch.Tensor
        The fftshifted 3D RFFT of the padded volume, with DC zeroed out.
    volume_mean_scaled : float
        `volume.mean() * d`, the constant to add back to projections after IFFT.
    pad_width : int
        Number of pixels padded on each side (0 if pad_factor <= 1.0).
    """
    d = volume.shape[-1]

    pad_width = 0
    if pad_factor > 1.0:
        pad_width = int((d * (pad_factor - 1.0)) // 2)
        edge_value = compute_cube_face_averages(volume, n=4)
        volume = F.pad(volume, pad=[pad_width] * 6, mode="constant", value=edge_value)

    volume_mean_scaled = volume.mean() * d

    # Center-to-origin shift, then 3D RFFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft[..., 0, 0, 0] = 0.0  # zero DC to avoid low-res artifacts
    dft = torch.fft.fftshift(dft, dim=(-3, -2))  # shift so DC is at center

    return dft, volume_mean_scaled, pad_width


def project_from_precomputed_dft(
    dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    volume_mean_scaled: float,
    pad_width: int,
    fftfreq_max: float | None = 0.5,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Extract central slices from a precomputed DFT and transform to real space.

    Parameters
    ----------
    dft : torch.Tensor
        Precomputed fftshifted 3D RFFT from `precompute_volume_dft`.
    rotation_matrices : torch.Tensor
        `(..., 3, 3)` rotation matrices for slice extraction.
    volume_mean_scaled : float
        Constant to add back after IFFT (volume_mean * d).
    pad_width : int
        Padding width to remove from projections.
    fftfreq_max : float | None
        Maximum frequency in cycles per pixel. Default is 0.5.
    zyx_matrices : bool
        Whether matrices operate on zyx coordinates. Default is False.

    Returns
    -------
    projections : torch.Tensor
        `(..., d, d)` real-space projections.
    """
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )

    # Transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    # Unpad
    if pad_width > 0:
        projections = F.pad(projections, pad=[-pad_width] * 4)

    # Add back the mean
    projections += volume_mean_scaled

    return projections


def generate_projection_batch(
    dft: torch.Tensor,
    volume_mean_scaled: float,
    pad_width: int,
    phi: torch.Tensor,  # (B,)
    theta: torch.Tensor,  # (B,)
    psi: torch.Tensor,  # (B,)
    fftfreq_max: float = 0.5,
) -> torch.Tensor:  # (B, h, w)
    """Generate a batch of 2D projections from a precomputed volume DFT."""
    rot_matrix = roma.euler_to_rotmat("ZYZ", angles=(phi, theta, psi), degrees=True)

    return project_from_precomputed_dft(
        dft=dft,
        rotation_matrices=rot_matrix,
        volume_mean_scaled=volume_mean_scaled,
        pad_width=pad_width,
        fftfreq_max=fftfreq_max,
    )


def apply_fourier_filters(
    projections: torch.Tensor,  # (B, h, w)
    fourier_filters: torch.Tensor,  # (f, h, w // 2 + 1)
) -> torch.Tensor:  # (f, B, h, w)
    """Apply pre-calculated Fourier filters to a batch of real-space projections."""
    projections_fft = torch.fft.rfft2(projections)
    projections_fft = projections_fft[None, ...] * fourier_filters[:, None, :, :]

    return torch.fft.irfft2(projections_fft)


def process_batch(
    dft: torch.Tensor,
    volume_mean_scaled: float,
    pad_width: int,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor,
    num_angle: int,
    num_radius: int,
    warp_polar_kwargs: dict,
    fftfreq_max: float = 0.5,
) -> torch.Tensor:
    """Generate a batch of polar proj. from a DFT, FFT along angular dim."""
    projections = generate_projection_batch(
        dft=dft,
        volume_mean_scaled=volume_mean_scaled,
        pad_width=pad_width,
        phi=phi,
        theta=theta,
        psi=psi,
        fftfreq_max=fftfreq_max,
    )

    projections_filtered = apply_fourier_filters(projections, fourier_filters)

    # Constants for the warp function
    center = (projections_filtered.shape[-2] / 2, projections_filtered.shape[-2] / 2)
    radius = projections_filtered.shape[-1] / 2  # Assuming square projections

    # warp_offset_polar expects only a single batch dimension, but have two batch dims.
    # Create a temporary view to combine batch dimensions
    num_defocus = fourier_filters.shape[0]
    batch_size = phi.shape[0]
    projections_filtered_view = projections_filtered.view(
        -1,
        projections_filtered.shape[-2],
        projections_filtered.shape[-1],
    )

    projections_polar = warp_offset_polar(
        projections_filtered_view,
        num_angle=num_angle,
        num_radius=num_radius,
        center=center,
        radius=radius,
        **warp_polar_kwargs,
    )

    # FFT along angular dimension
    # NOTE: Orthonormal transformation to preserve forward-backward scaling of features
    projections_polar_fft = torch.fft.fft(projections_polar, dim=-2, norm="ortho")

    # Reshape back to (num_defocus, batch_size, num_angle, num_radius)
    projections_polar_fft = projections_polar_fft.view(
        num_defocus, batch_size, num_angle, num_radius
    )

    return projections_polar_fft


def do_pipelined_projection_and_transforms(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor | None,
    num_angle: int,
    num_radius: int,
    warp_polar_kwargs: dict,
    projection_batch_size: int = 128,
    pad_factor: float = 2.0,
    fftfreq_max: float = 0.5,
    show_progress: bool = True,
) -> torch.Tensor:
    """Pipelined generation and transforms of projections in polar coordinates.

    Parameters
    ----------
    volume : torch.Tensor
        `(d, d, d)` cubic volume on GPU.
    phi, theta, psi : torch.Tensor
        `(N,)` Euler angles in degrees.
    fourier_filters : torch.Tensor | None
        `(f, h, w // 2 + 1)` Fourier-space filters, or None for identity.
    num_angle : int
        Number of angular samples in polar projection.
    num_radius : int
        Number of radial samples in polar projection.
    warp_polar_kwargs : dict
        Additional kwargs for `warp_offset_polar`.
    projection_batch_size : int
        Orientations per GPU batch. Default is 128.
    pad_factor : float
        Volume padding factor for Fourier slicing. Default is 2.0.
    fftfreq_max : float
        Maximum frequency in cycles per pixel. Default is 0.5.
    show_progress : bool
        Whether to show a tqdm progress bar. Default is True.

    Returns
    -------
    torch.Tensor
        `(num_defocus, num_projections, num_angle, num_radius)` complex64 tensor
        on CPU (pinned memory).
    """
    if fourier_filters is None:
        fourier_filters = torch.ones(
            (1, volume.shape[1], volume.shape[2] // 2 + 1),
            device=volume.device,
            dtype=torch.complex64,
        )

    # Precompute the 3D RFFT once — stays on GPU for the entire pipeline
    dft, volume_mean_scaled, pad_width = precompute_volume_dft(
        volume, pad_factor=pad_factor
    )

    # Free the original volume from GPU since we only need the DFT now
    del volume

    # Allocate memory on CPU for full storage
    num_projections = phi.shape[0]
    num_defocus = fourier_filters.shape[0]
    polar_projections_transformed_cpu = torch.empty(
        (num_defocus, num_projections, num_angle, num_radius),
        dtype=torch.complex64,
        device="cpu",
        pin_memory=True,
    )

    pbar = None
    if show_progress:
        pbar = tqdm.tqdm(
            total=num_projections,
            desc="calc. projections",
            unit="proj",
        )

    for start_idx in range(0, num_projections, projection_batch_size):
        end_idx = min(start_idx + projection_batch_size, num_projections)
        s = slice(start_idx, end_idx)
        batch_size = end_idx - start_idx

        projections_polar_fft = process_batch(
            dft=dft,
            volume_mean_scaled=volume_mean_scaled,
            pad_width=pad_width,
            phi=phi[s],
            theta=theta[s],
            psi=psi[s],
            fourier_filters=fourier_filters,
            num_angle=num_angle,
            num_radius=num_radius,
            warp_polar_kwargs=warp_polar_kwargs,
            fftfreq_max=fftfreq_max,
        )

        # Copy to CPU pinned memory (async for overlap with next batch compute)
        polar_projections_transformed_cpu[:, s].copy_(
            projections_polar_fft, non_blocking=True
        )
        del projections_polar_fft

        if pbar is not None:
            pbar.update(batch_size)

    if pbar is not None:
        pbar.close()

    # Ensure all async copies are complete before returning
    if dft.is_cuda:
        torch.cuda.synchronize(dft.device)

    del dft

    return polar_projections_transformed_cpu
