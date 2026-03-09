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

import tqdm
import roma
import torch
from torch_fourier_slice import project_3d_to_2d

from panther_em.utils.warp_transforms import warp_offset_polar


def generate_projection_batch(
    volume: torch.Tensor,  # (d, h, w)
    phi: torch.Tensor,  # (B,)
    theta: torch.Tensor,  # (B,)
    psi: torch.Tensor,  # (B,)
    pad_factor: float = 2.0,
    fftfreq_max: float = 0.5,
) -> torch.Tensor:  # (B, h, w)
    """Generate a batch of 2D projections from a 3D volume."""
    rot_matrix = roma.euler_to_rotmat("ZYZ", angles=(phi, theta, psi), degrees=True)

    return project_3d_to_2d(
        volume=volume,
        rotation_matrices=rot_matrix,
        pad_factor=pad_factor,
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
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor,
    num_angle: int,
    num_radius: int,
    warp_polar_kwargs: dict,
) -> torch.Tensor:
    """Generate a batch of polar projections from a 3D volume, FFT along angular dim."""
    projections = generate_projection_batch(
        volume=volume,
        phi=phi,
        theta=theta,
        psi=psi,
    )

    projections_filtered = apply_fourier_filters(projections, fourier_filters)

    # Constants for the warp function
    center = (projections_filtered.shape[-2] / 2, projections_filtered.shape[-2] / 2)
    radius = projections_filtered.shape[-1] / 2  # Assuming square projections

    # warp_offset_polar expects only a single batch dimension, but have two batch dims.
    # Create a temporary view to combine batch dimensions
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

    return torch.fft.fft(projections_polar, dim=-2)


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
    show_progress: bool = True,
) -> torch.Tensor:
    """Pipelined generation and transforms of projections in polar coordinates."""
    if fourier_filters is None:
        fourier_filters = torch.ones(
            (1, volume.shape[1], volume.shape[2] // 2 + 1),
            device=volume.device,
            dtype=torch.complex64,
        )

    # Allocate memory on CPU for full storage
    num_projections = phi.shape[0]
    num_defocus = fourier_filters.shape[0]
    polar_projections_transformed_cpu = torch.empty(
        (num_defocus, num_projections, num_angle, num_radius),
        dtype=torch.complex64,
        device="cpu",
        pin_memory=True,
    )
    
    range_obj = range(0, num_projections, projection_batch_size)
    if show_progress:
        range_obj = tqdm.tqdm(
            range_obj,
            desc="calc. projections",
            unit="proj",
            total=num_projections,
        )

    for start_idx in range_obj:
        end_idx = min(start_idx + projection_batch_size, num_projections)
        s = slice(start_idx, end_idx)
        batch_size = end_idx - start_idx

        batch_phi = phi[s]
        batch_theta = theta[s]
        batch_psi = psi[s]

        projections_polar_fft = process_batch(
            volume=volume,
            phi=batch_phi,
            theta=batch_theta,
            psi=batch_psi,
            fourier_filters=fourier_filters,
            num_angle=num_angle,
            num_radius=num_radius,
            warp_polar_kwargs=warp_polar_kwargs,
        )

        polar_projections_transformed_cpu[:, s] = projections_polar_fft.cpu()

        # Update progress bar by actual batch size (tqdm auto-increments by 1)
        if show_progress and end_idx < num_projections:
            range_obj.update(batch_size - 1)

    # Ensure all async copies are complete before returning
    if volume.is_cuda:
        torch.cuda.synchronize(volume.device)

    return polar_projections_transformed_cpu
