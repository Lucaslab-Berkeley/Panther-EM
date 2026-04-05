"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from panther_em.decomposition.result import DecompositionResult

from .pipeline_projections import do_pipelined_projection_and_transforms

# TODO: Large numbers of projections (orientations) and high polar resolutions
# (number of angular and radial components) can lead to OOM errors on GPUs. Need to
# implements a batched way to handle portions of the data at a time handing off
# memory between CPU/GPU.


class PolarProjectionDecomposer:
    """Manages projection generation and block-circulant data decomposition.

    NOTE: The Euler angles (phi, theta) represent the first two angles in ZYZ format
    where the final in-plane rotation angle (psi) is assumed to be zero based on the
    polar projection model.

    Parameters
    ----------
    volume : np.ndarray | torch.Tensor
        The 3D volume to decompose. If torch.Tensor, should be on the target device.
    phi_values : np.ndarray | torch.Tensor
        Phi values, in degrees of ZYZ Euler angles, for projection orientations.
    theta_values : np.ndarray | torch.Tensor
        Theta values, in degrees of ZYZ Euler angles, for projection orientations.
    num_radius : int
        Number of radial components in polar projections.
    num_angle : int, optional
        Number of angular components in polar projections. Default is 360.
    device : str | torch.device, optional
        Device to use for computation ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.

    Attributes
    ----------
    result : DecompositionResult | None
        The decomposition result after calling `do_decomposition()`.
    polar_transform : OffsetPolarTransform | None
        Transform object for efficient polar<->cartesian conversion after decomposition.

    Examples
    --------
    >>> # CPU computation with numpy
    >>> decomposer = PolarProjectionDecomposer(
    ...     volume, phi, theta, num_radius=256, num_angle=720, device="cpu"
    ... )
    >>> result = decomposer.do_decomposition()

    >>> # GPU computation with torch tensors
    >>> volume_gpu = torch.from_numpy(volume).cuda()
    >>> phi_gpu = torch.from_numpy(phi).cuda()
    >>> theta_gpu = torch.from_numpy(theta).cuda()
    >>> decomposer = PolarProjectionDecomposer(
    ...     volume_gpu, phi_gpu, theta_gpu, num_radius=256, num_angle=720, device="cuda"
    ... )
    >>> result = decomposer.do_decomposition()
    >>> result.save("decomposition.npz")
    """

    def __init__(
        self,
        volume: np.ndarray | torch.Tensor,
        phi_values: np.ndarray | torch.Tensor,
        theta_values: np.ndarray | torch.Tensor,
        num_radius: int,
        fourier_filters: np.ndarray | torch.Tensor | None = None,
        num_angle: int = 360,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the polar projection decomposer."""
        self.device = torch.device(device)

        # Convert inputs to tensors on the correct device
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.copy())

        if isinstance(phi_values, np.ndarray):
            phi_values = torch.from_numpy(phi_values.copy())

        if isinstance(theta_values, np.ndarray):
            theta_values = torch.from_numpy(theta_values.copy())

        # Send to the target device
        self.volume = volume.to(self.device)
        self.phi_values = phi_values.to(self.device)
        self.theta_values = theta_values.to(self.device)
        self.fourier_filters = (
            fourier_filters.to(self.device) if fourier_filters is not None else None
        )

        self.num_radius = num_radius
        self.num_angle = num_angle

        self._result: DecompositionResult | None = None

    @property
    def result(self) -> DecompositionResult:
        """Access decomposition results.

        Returns
        -------
        DecompositionResult
            The stored decomposition result.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        if self._result is None:
            raise ValueError(
                "Decomposition not yet performed. Call do_decomposition() first."
            )
        return self._result

    @property
    def is_decomposed(self) -> bool:
        """Check if decomposition has been performed."""
        return self._result is not None

    def do_decomposition(
        self,
        k_max: int | None = None,
        projection_batch_size: int = 128,
        block_batch_size: int = 8,
    ) -> DecompositionResult:
        """Run the block-circulant decomposition using the held orientations.

        Generates polar projections on-the-fly and performs eigendecomposition
        on each angular frequency block.

        NOTE: Computation is split into two stages: 1) on-the-fly projection generation
        plus FFT transformation 2) block-wise SVD decomposition. A barrier exists
        between these two stages.

        Parameters
        ----------
        k_max : int | None, optional
            Maximum angular frequency index to compute. If None, uses all
            angular components. Default is None.
        projection_batch_size : int, optional
            Number of projections to process at a time for memory efficiency.
            Default is 128.
        block_batch_size : int, optional
            Number of frequency blocks to process at a time on GPU for SVD.
            Default is 8.

        Returns
        -------
        DecompositionResult
            The decomposition result containing singular values and vectors.
        """
        # Stage 1: GPU projection generation and transform. Results are generally too
        # large to fit in GPU memory, so stored on CPU memory.
        polar_projections_transformed_cpu = do_pipelined_projection_and_transforms(
            volume=self.volume,
            phi=self.phi_values,
            theta=self.theta_values,
            psi=torch.zeros_like(self.phi_values),
            fourier_filters=self.fourier_filters,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            warp_polar_kwargs={"preserve_energy": True},
            projection_batch_size=projection_batch_size,
        )

        # Stage 2: Decompose each frequency block with SVD
        # Collapse defocus dimension into orientations for SVD:
        # (num_defocus, num_orients, num_angle, num_radius) ->
        # (num_defocus * num_orients, num_angle, num_radius)
        num_fourier_filters, num_orients, num_angle, num_radius = (
            polar_projections_transformed_cpu.shape
        )
        polar_projections_transformed_cpu = polar_projections_transformed_cpu.reshape(
            num_fourier_filters * num_orients, num_angle, num_radius
        )
        num_rows = num_fourier_filters * num_orients  # combined orientation dimension

        # Determine k_max
        if k_max is None:
            k_max = num_angle

        # # Pre-compute radial scaling vector once
        # r = torch.arange(num_radius, device=self.device, dtype=torch.float32)
        # r = r / num_radius

        # Allocate tensors for the SVD results on CPU
        singular_values = torch.zeros(
            (k_max, num_radius), dtype=torch.float32, device="cpu"
        )
        left_singular_vectors = torch.zeros(
            (num_fourier_filters, num_orients, k_max, num_radius),
            dtype=torch.complex64,
            device="cpu",
        )
        right_singular_vectors = torch.zeros(
            (k_max, num_radius, num_radius),
            dtype=torch.complex64,
            device="cpu",
        )

        # Stage 2: Loop over frequency blocks in batches for SVD decomposition
        for k_start in tqdm.tqdm(
            range(0, k_max, block_batch_size), desc="decomp freq blocks"
        ):
            k_end = min(k_start + block_batch_size, k_max)

            for k in range(k_start, k_end):
                # Extract frequency block: (num_rows, num_radius)
                freq_block = polar_projections_transformed_cpu[:, k, :]
                freq_block = freq_block.to(device=self.device, non_blocking=True)

                # # Scale by the radial component (proper integration term r dr)
                # freq_block_scaled = freq_block * r[None, :]

                u, s, vh = torch.linalg.svd(freq_block, full_matrices=False)

                singular_values[k] = s.cpu()
                left_singular_vectors[:, :, k, :] = u.reshape(
                    num_fourier_filters, num_orients, num_radius
                ).cpu()
                right_singular_vectors[k] = vh.mH.cpu()

                # del freq_block, u, s, vh

        # Convert results back to numpy for storage
        self._result = DecompositionResult(
            singular_values=singular_values.cpu().numpy().astype(np.float32),
            left_singular_vectors=left_singular_vectors.cpu().numpy(),
            right_singular_vectors=right_singular_vectors.cpu().numpy(),
            num_fourier_filters=num_fourier_filters,
            num_orientations=num_orients,
            num_angular_components=num_angle,
            num_radial_components=num_radius,
            k_max=k_max,
        )

        return self._result

    @classmethod
    def from_result(
        cls,
        result: DecompositionResult | str | Path,
        volume: np.ndarray | torch.Tensor,
        phi_values: np.ndarray | torch.Tensor | None = None,
        theta_values: np.ndarray | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> "PolarProjectionDecomposer":
        """Create a decomposer with a pre-computed result.

        Useful for loading saved decompositions and constructing features
        without re-running the decomposition.

        Parameters
        ----------
        result : DecompositionResult | str | Path
            A DecompositionResult instance or path to a saved result.
        volume : np.ndarray | torch.Tensor
            The volume (needed for Cartesian feature construction).
        phi_values : np.ndarray | torch.Tensor | None, optional
            Azimuthal angles. Can be None if only constructing features.
        theta_values : np.ndarray | torch.Tensor | None, optional
            Polar angles. Can be None if only constructing features.
        device : str | torch.device, optional
            Device to use for computation. Default is 'cpu'.

        Returns
        -------
        PolarProjectionDecomposer
            A decomposer instance with the loaded result.
        """
        if isinstance(result, (str, Path)):
            result = DecompositionResult.load(result)

        device_obj = torch.device(device)

        # Create dummy arrays if not provided
        if phi_values is None:
            phi_values = torch.zeros(result.num_orientations, device=device_obj)
        if theta_values is None:
            theta_values = torch.zeros(result.num_orientations, device=device_obj)

        instance = cls(
            volume,
            phi_values,
            theta_values,
            num_radius=result.num_radial_components,
            num_angle=result.num_angular_components,
            device=device_obj,
        )
        instance._result = result

        return instance
