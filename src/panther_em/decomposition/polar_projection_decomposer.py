"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from panther_em.decomposition.result import DecompositionResult
from panther_em.utils.warp_transforms import OffsetPolarTransform

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
        self._polar_transform: OffsetPolarTransform | None = None

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

    @property
    def polar_transform(self) -> OffsetPolarTransform:
        """Access the polar transform for coordinate conversions.

        Returns
        -------
        OffsetPolarTransform
            Transform object with cached coordinate mappings.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        if self._polar_transform is None:
            raise ValueError(
                "Polar transform not available. Call do_decomposition() first."
            )
        return self._polar_transform

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
        num_defocus, num_orients, num_angle, num_radius = (
            polar_projections_transformed_cpu.shape
        )
        polar_projections_transformed_cpu = polar_projections_transformed_cpu.reshape(
            num_defocus * num_orients, num_angle, num_radius
        )
        num_rows = num_defocus * num_orients  # combined orientation dimension

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
            (k_max, num_rows, num_radius),
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
                left_singular_vectors[k] = u.cpu()
                # Store V (columns of right singular vectors), not Vh
                right_singular_vectors[k] = vh.mH.cpu()

                # del freq_block, u, s, vh

        # Convert results back to numpy for storage
        self._result = DecompositionResult(
            singular_values=singular_values.cpu().numpy().astype(np.float32),
            left_singular_vectors=left_singular_vectors.cpu().numpy(),
            right_singular_vectors=right_singular_vectors.cpu().numpy(),
            num_orientations=num_rows,
            num_angular_components=num_angle,
            num_radial_components=num_radius,
            k_max=k_max,
        )

        # Create the polar transform for later use in reconstruction
        self._polar_transform = OffsetPolarTransform.from_image(
            image_shape=(self.volume.shape[-2], self.volume.shape[-1]),
            num_angle=num_angle,
            num_radius=num_radius,
            device="cuda" if self.device.type == "cuda" else "numpy",
        )

        return self._result

    def _get_angular_phase_component(self, k_idx: int) -> torch.Tensor:
        """Get the angular phase component for a given frequency.

        Parameters
        ----------
        k_idx : int
            Angular frequency index.

        Returns
        -------
        torch.Tensor
            Angular phase array with shape (num_angle,).
        """
        result = self.result
        angles = (
            2
            * np.pi
            * k_idx
            * torch.arange(
                result.num_angular_components, device=self.device, dtype=torch.float32
            )
            / result.num_angular_components
        )
        return torch.exp(1j * angles) / np.sqrt(result.num_angular_components)

    # def _undo_radial_scaling(self, radial_component: torch.Tensor) -> torch.Tensor:
    #     """Undo the r scaling applied during decomposition.

    #     Parameters
    #     ----------
    #     radial_component : torch.Tensor
    #         Radial component with shape (num_radius,).

    #     Returns
    #     -------
    #     torch.Tensor
    #         Unscaled radial component.
    #     """
    #     result = self.result
    #     r = (
    #         torch.arange(
    #             result.num_radial_components, device=self.device, dtype=torch.float32
    #         )
    #         / result.num_radial_components
    #     )
    #     # Avoid division by zero at r=0
    #     return radial_component / torch.where(r > 0, r, torch.ones_like(r))

    def construct_polar_feature(
        self, k_idx: int, eig_idx: int, return_torch: bool = False
    ) -> np.ndarray | torch.Tensor:
        """Construct a single polar feature for an angular frequency and eigenvalue.

        Parameters
        ----------
        k_idx : int
            The index of the angular frequency component.
        eig_idx : int
            The index of the eigenvector to use.
        return_torch : bool, optional
            If True, returns a torch.Tensor on the device.
            Otherwise returns numpy array. Default is False.

        Returns
        -------
        np.ndarray | torch.Tensor
            A complex array for the feature in polar space with shape
            (num_angle, num_radius).

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        result = self.result  # Will raise if not decomposed

        # Get angular and radial components
        angular_component = self._get_angular_phase_component(k_idx)
        radial_component = torch.from_numpy(
            result.get_radial_eigenvector(k_idx, eig_idx)
        ).to(device=self.device, dtype=torch.complex64)

        # Construct the feature by taking the outer product
        polar_feature = torch.outer(angular_component, radial_component)

        if return_torch:  # Ensure return_torch is handled correctly
            return polar_feature
        return polar_feature.cpu().numpy()

    def reconstruct_projection(
        self,
        orientation_idx: int,
        num_components: int | None = None,
        output_shape: tuple[int, int] | None = None,
        return_polar: bool = False,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
    ) -> np.ndarray | torch.Tensor:
        """Reconstruct a Cartesian projection at a specific orientation.

        Parameters
        ----------
        orientation_idx : int
            Index of the orientation to reconstruct (corresponds to phi[i], theta[i]).
        num_components : int | None, optional
            Number of singular value components to use in reconstruction.
            If None, uses all available components. Default is None.
        output_shape : tuple[int, int] | None, optional
            Shape of the output Cartesian image. If None, uses the volume's
            spatial dimensions. Default is None.
        return_polar : bool, optional
            If True, returns the polar representation instead of converting
            to Cartesian. Default is False.
        return_torch : bool, optional
            If True, returns a torch.Tensor on the device.
            Otherwise returns numpy array. Default is False.
        order : int, optional
            Interpolation order for polar to cartesian conversion (0-5).
            Default is 5.
        mode : str, optional
            How to handle values outside boundaries during warping.
            Default is "constant".
        cval : float, optional
            The constant value to use for padding when mode is "constant".
            Default is 0.0.

        Returns
        -------
        np.ndarray | torch.Tensor
            Reconstructed projection (complex-valued).
            Shape is (num_angle, num_radius) if return_polar is True,
            otherwise output_shape.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet or orientation_idx is out of
            bounds.
        """
        result = self.result  # Will raise if not decomposed

        if orientation_idx >= result.num_orientations:
            raise ValueError(
                f"orientation_idx {orientation_idx} out of bounds "
                f"(max: {result.num_orientations - 1})"
            )

        if num_components is None:
            num_components = result.num_radial_components

        # Initialize polar representation
        polar_projection = torch.zeros(
            (result.num_angular_components, result.num_radial_components),
            dtype=torch.complex64,
            device=self.device,
        )

        # Move result arrays to device as torch tensors
        left_singular_vectors = torch.from_numpy(result.left_singular_vectors).to(
            device=self.device, dtype=torch.complex64
        )
        singular_values = torch.from_numpy(result.singular_values).to(
            device=self.device, dtype=torch.float32
        )
        right_singular_vectors = torch.from_numpy(result.right_singular_vectors).to(
            device=self.device, dtype=torch.complex64
        )

        # Reconstruct in polar space by combining SVD components
        for k_idx in range(result.k_max):
            # Get angular phase component
            angular_component = self._get_angular_phase_component(k_idx)

            # Sum over the first num_components singular values
            for eig_idx in range(num_components):
                # Left singular vector for this orientation and frequency
                u_ki = left_singular_vectors[k_idx, orientation_idx, eig_idx]

                # Singular value
                s_k = singular_values[k_idx, eig_idx]

                # Right singular vector (radial component)
                v_k = right_singular_vectors[k_idx, :, eig_idx]

                # # Undo the radial scaling applied during decomposition
                # radial_component = self._undo_radial_scaling(v_k)
                radial_component = v_k

                # Add contribution: u * s * v for this (k, eigenvalue) pair
                contribution = u_ki * s_k * radial_component
                polar_projection += torch.outer(angular_component, contribution)

        if return_polar:
            if return_torch:
                return polar_projection
            return polar_projection.cpu().numpy()

        # Convert to Cartesian space using cached transform
        if output_shape is not None:
            # Need to create a new transform if output shape differs
            transform_device = "cuda" if self.device.type == "cuda" else "numpy"
            temp_transform = OffsetPolarTransform.from_image(
                image_shape=output_shape,
                num_angle=result.num_angular_components,
                num_radius=result.num_radial_components,
                device=transform_device,  # type: ignore
            )
            cartesian_projection = temp_transform.to_cartesian(
                polar_projection, order=order, mode=mode
            )
        else:
            # Use cached transform
            cartesian_projection = self.polar_transform.to_cartesian(
                polar_projection, order=order, mode=mode
            )

        if return_torch:
            return cartesian_projection
        # Convert to numpy if on GPU
        if isinstance(cartesian_projection, torch.Tensor):
            return cartesian_projection.cpu().numpy()
        return cartesian_projection

    def construct_cartesian_feature(
        self,
        k_idx: int,
        eig_idx: int,
        output_shape: tuple[int, int] | None = None,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Construct a single Cartesian feature for an angular frequency and eigenvalue.

        Uses the offset polar coordinate system for better spatial coverage when
        transforming from polar to cartesian space.

        Parameters
        ----------
        k_idx : int
            The index of the angular frequency component.
        eig_idx : int
            The index of the eigenvector to use.
        output_shape : tuple[int, int] | None, optional
            Shape of the output Cartesian image. If None, uses the volume's
            spatial dimensions. Default is None.
        return_torch : bool, optional
            If True, returns a torch.Tensor on the device.
            Otherwise returns numpy array. Default is False.
        order : int, optional
            Interpolation order for polar to cartesian conversion (0-5).
            Default is 5.
        mode : str, optional
            How to handle values outside boundaries during warping.
            Default is "constant".
        cval : float, optional
            Value to use for constant padding. Default is 0.0.
        preserve_energy : bool, optional
            Whether to apply a Jacobian correction to preserve energy between polar and
            cartesian spaces. By default True.
        wrap_angular_axis : bool, optional
            Whether to apply wrap padding along the angular axis *only* for proper
            interpolation across the 360-0 degree boundary. Default is True.

        Returns
        -------
        np.ndarray | torch.Tensor
            A complex array for the feature in Cartesian space.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        polar_feature = self.construct_polar_feature(k_idx, eig_idx, return_torch=True)

        # Convert to cartesian using cached transform
        if output_shape is not None:
            # Need to create a new transform if output shape differs
            result = self.result
            transform_device = "cuda" if self.device.type == "cuda" else "numpy"
            temp_transform = OffsetPolarTransform.from_image(
                image_shape=output_shape,
                num_angle=result.num_angular_components,
                num_radius=result.num_radial_components,
                device=transform_device,  # type: ignore
            )
            cartesian_feature = temp_transform.to_cartesian(
                polar_feature,
                order=order,
                mode=mode,
                cval=cval,
                preserve_energy=preserve_energy,
                wrap_angular_axis=wrap_angular_axis,
            )
        else:
            # Use cached transform
            cartesian_feature = self.polar_transform.to_cartesian(
                polar_feature,
                order=order,
                mode=mode,
                cval=cval,
                preserve_energy=preserve_energy,
                wrap_angular_axis=wrap_angular_axis,
            )

        if return_torch:
            return cartesian_feature
        # Convert to numpy if on GPU
        if isinstance(cartesian_feature, torch.Tensor):
            return cartesian_feature.cpu().numpy()
        return cartesian_feature

    def save_result(self, path: str | Path) -> None:
        """Save the decomposition result to disk.

        Parameters
        ----------
        path : str | Path
            Path to save the result.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        self.result.save(path)

    def clear_polar_transform_cache(self) -> None:
        """Clear the cached coordinate mappings in the polar transform.

        This can free up memory when the transform is no longer needed.
        """
        if self._polar_transform is not None:
            self._polar_transform.clear_cache()

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

        # Determine device string for OffsetPolarTransform
        transform_device = "cuda" if device_obj.type == "cuda" else "numpy"

        # Create the polar transform
        instance._polar_transform = OffsetPolarTransform.from_image(
            image_shape=(instance.volume.shape[-2], instance.volume.shape[-1]),
            num_angle=result.num_angular_components,
            num_radius=result.num_radial_components,
            device=transform_device,  # type: ignore
        )

        return instance
