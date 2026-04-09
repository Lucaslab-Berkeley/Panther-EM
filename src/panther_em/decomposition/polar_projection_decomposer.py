"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from panther_em.decomposition.result import DecompositionResult

from .pipeline_projections import do_pipelined_projection_and_transforms


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
        if phi_values.shape != theta_values.shape:
            raise ValueError(
                "phi_values and theta_values must have the same shape, "
                f"got {phi_values.shape} and {theta_values.shape}"
            )

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
        eig_max: int | None = None,
        projection_batch_size: int = 128,
        block_batch_size: int = 8,
    ) -> DecompositionResult:
        """Run the block-circulant decomposition using the held orientations.

        NOTE: Computation is split into two stages: 1) on-the-fly projection generation
        plus FFT transformation 2) block-wise SVD decomposition. A barrier exists
        between these two stages, and intermediate results from (1) are stored on CPU
        memory to support problem size scaling.

        Parameters
        ----------
        k_max : int | None, optional
            Maximum angular frequency index to compute. If None, uses all
            angular components. Default is None.
        eig_max : int | None, optional
            Maximum radial eigenvalue index to compute. If None, uses all
            radial components. Default is None.
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
        # Store original shape for later reshaping of results
        batch_shape = polar_projections_transformed_cpu.shape[:-2]
        num_angle, num_radius = polar_projections_transformed_cpu.shape[-2:]

        # Flatten all outer dimensions:
        # (..., num_angle, num_radius) -> (batch_size, num_angle, num_radius)
        batch_size = int(np.prod(batch_shape))
        polar_projections_reshaped = polar_projections_transformed_cpu.reshape(
            batch_size, num_angle, num_radius
        )

        # Determine k_max
        if k_max is None:
            k_max = num_angle

        # Determine eig_max
        if eig_max is None:
            eig_max = num_radius

        # Allocate tensors for the SVD results on CPU
        U = torch.zeros(
            (*batch_shape, k_max, eig_max), dtype=torch.complex64, device="cpu"
        )
        S = torch.zeros((k_max, eig_max), dtype=torch.float32, device="cpu")
        Vh = torch.zeros(
            (k_max, eig_max, num_radius),
            dtype=torch.complex64,
            device="cpu",
        )

        # Stage 2: Loop over frequency blocks in batches for batched SVD decomposition
        for k_start in tqdm.tqdm(
            range(0, k_max, block_batch_size), desc="decomp freq blocks"
        ):
            k_end = min(k_start + block_batch_size, k_max)
            num_k_batch = k_end - k_start
            k_indices = torch.arange(k_start, k_end, device="cpu")

            freq_blocks = polar_projections_reshaped[:, k_indices, :]
            freq_blocks = freq_blocks.to(device=self.device, non_blocking=True)

            # Reshape from (rows, num_k_B, num_r) --> (num_k_B, rows, num_r)
            # since we want SVD to operate on (rows, num_r)
            freq_blocks = freq_blocks.permute(1, 0, 2)

            u, s, vh = torch.linalg.svd(freq_blocks, full_matrices=False)

            u = u[..., :eig_max]
            s = s[:, :eig_max]
            vh = vh[:, :eig_max, :]

            # Reshape outer indices for storage
            u_reshaped = u.permute(1, 0, 2)
            u_reshaped = u_reshaped.reshape(*batch_shape, num_k_batch, eig_max)

            U[..., k_indices, :] = u_reshaped.cpu()
            S[k_indices] = s.cpu()
            Vh[k_indices, :, :] = vh.cpu()

        # Convert results back to numpy for storage
        self._result = DecompositionResult(
            S=S.cpu().numpy().astype(np.float32),
            U=U.cpu().numpy(),
            Vh=Vh.cpu().numpy(),
            k_max=k_max,
            eig_max=eig_max,
            num_fourier_filters=batch_shape[0] if len(batch_shape) > 0 else 1,
            num_orientations=batch_shape[1] if len(batch_shape) > 1 else 1,
            num_angular_components=num_angle,
            num_radial_components=num_radius,
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
