"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path
import tqdm
import numpy as np
import torch

from panther_em.decomposition.result import DecompositionResult
from panther_em.utils import (
    get_polar_projections_from_volume,
    warp_offset_polar_inverse,
)


class PolarProjectionDecomposer:
    """Manages projection generation and block-circulant data decomposition.

    NOTE: The Euler angles (phi, theta) represent the first two angles in ZYZ format
    where the final in-plane rotation angle (psi) is assumed to be zero based on the
    polar projection model.

    Parameters
    ----------
    volume : np.ndarray
        The 3D volume to decompose.
    phi_values : np.ndarray
        Phi values, in degrees of ZYZ Euler angles, for projection orientations.
    theta_values : np.ndarray
        Theta values, in degrees of ZYZ Euler angles, for projection orientations.
    num_radial_components : int | None, optional
        Number of radial components in polar projections.
        If None, then the number of radial components are calculated automatically.
    num_angular_components : int | None, optional
        Number of angular components in polar projections.
        If None, then number of angular components is set to 360.
    device : str | torch.device, optional
        Device to use for computation ('cpu', 'cuda', etc.). Default is 'cpu'.

    Attributes
    ----------
    result : DecompositionResult | None
        The decomposition result after calling `do_decomposition()`.

    Examples
    --------
    >>> # assume volume, phi, theta are defined
    >>> decomposer = PolarProjectionDecomposer(volume, phi, theta, device='cuda')
    >>> result = decomposer.do_decomposition()
    >>> result.save("decomposition.npz")
    """

    def __init__(
        self,
        volume: np.ndarray,
        phi_values: np.ndarray,
        theta_values: np.ndarray,
        num_radial_components: int | None = None,
        num_angular_components: int = 360,
        device: str | torch.device = "cpu",
    ) -> None:
        self.volume = volume
        self.phi_values = phi_values
        self.theta_values = theta_values

        # Either both must be set or both must be None
        if (num_radial_components is None) != (num_angular_components is None):
            raise ValueError(
                "Either both num_radial_components and num_angular_components "
                "must be set, or both must be None."
            )

        # Set the values (None handled in projection generation step)
        self.num_radial_components = num_radial_components
        self.num_angular_components = num_angular_components

        self._result: DecompositionResult | None = None
        self.device = torch.device(device)

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

    def do_decomposition(self, k_max: int | None = None) -> DecompositionResult:
        """Run the block-circulant decomposition using the held orientations.

        Generates polar projections on-the-fly and performs eigendecomposition
        on each angular frequency block.

        Parameters
        ----------
        k_max : int | None, optional
            Maximum angular frequency index to compute. If None, uses all
            angular components. Default is None.

        Returns
        -------
        DecompositionResult
            The decomposition result containing singular values and vectors.
        """
        # Generate projections on-the-fly (memory efficient)
        # Note: get_polar_projections_from_volume returns numpy array
        if (
            self.num_radial_components is not None
            and self.num_angular_components is not None
        ):
            warp_polar_kwargs = {
                "output_shape": (
                    self.num_angular_components,
                    self.num_radial_components,
                ),
                "scaling": "linear",
                "mode": "wrap",  # Angular dimension is periodic
            }
        else:
            warp_polar_kwargs = None
        projections_polar = get_polar_projections_from_volume(
            volume=self.volume,
            phi=self.phi_values,
            theta=self.theta_values,
            psi=0.0,
            warp_polar_kwargs=warp_polar_kwargs,
        )
        num_orients, num_angular_comp, num_radial_comp = projections_polar.shape

        # Convert to torch tensor and move to device
        projections_polar = torch.from_numpy(projections_polar).to(
            device=self.device, dtype=torch.complex64
        )

        # FFT along angular dimension
        projections_fft = torch.fft.fft(projections_polar, dim=1)

        # Determine k_max
        if k_max is None:
            k_max = num_angular_comp

        # Iterate over all angular frequency components and store the SVD results
        singular_values = torch.zeros(
            (k_max, num_radial_comp), dtype=torch.float32, device=self.device
        )
        left_singular_vectors = torch.zeros(
            (k_max, num_orients, num_radial_comp),
            dtype=torch.complex64,
            device=self.device,
        )
        right_singular_vectors = torch.zeros(
            (k_max, num_radial_comp, num_radial_comp),
            dtype=torch.complex64,
            device=self.device,
        )

        for k in tqdm.tqdm(range(k_max), desc="decomp freq blocks"):
            freq_block = projections_fft[:, k, :]

            # Scale by the radial component (proper integration term $r dr$)
            r = torch.arange(num_radial_comp, device=self.device, dtype=torch.float32)
            r = r / num_radial_comp
            freq_block_scaled = freq_block * r[None, :]

            # Call SVD on the scaled frequency block
            u, s, vh = torch.linalg.svd(freq_block_scaled, full_matrices=False)

            singular_values[k] = s
            left_singular_vectors[k] = u
            right_singular_vectors[k] = vh.conj().T

        # Convert results back to numpy for storage
        self._result = DecompositionResult(
            singular_values=singular_values.cpu().numpy().astype(np.complex64),
            left_singular_vectors=left_singular_vectors.cpu().numpy(),
            right_singular_vectors=right_singular_vectors.cpu().numpy(),
            num_orientations=num_orients,
            num_angular_components=num_angular_comp,
            num_radial_components=num_radial_comp,
            k_max=k_max,
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
            Angular phase array with shape (num_angular_components,).
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

    def _undo_radial_scaling(self, radial_component: torch.Tensor) -> torch.Tensor:
        """Undo the r scaling applied during decomposition.

        Parameters
        ----------
        radial_component : torch.Tensor
            Radial component with shape (num_radial_components,).

        Returns
        -------
        torch.Tensor
            Unscaled radial component.
        """
        result = self.result
        r = (
            torch.arange(
                result.num_radial_components, device=self.device, dtype=torch.float32
            )
            / result.num_radial_components
        )
        # Avoid division by zero at r=0
        return radial_component / torch.where(r > 0, r, torch.ones_like(r))

    # @torch.compile
    def construct_polar_feature(self, k_idx: int, eig_idx: int) -> np.ndarray:
        """Construct a single polar feature for an angular frequency and eigenvalue.

        Parameters
        ----------
        k_idx : int
            The index of the angular frequency component.
        eig_idx : int
            The index of the eigenvector to use.

        Returns
        -------
        np.ndarray
            A complex np.ndarray for the feature in polar space with shape
            (num_angular_components, num_radial_components).

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

        return polar_feature.cpu().numpy()

    def reconstruct_projection(
        self,
        orientation_idx: int,
        num_components: int | None = None,
        output_shape: tuple[int, int] | None = None,
        scaling: str = "linear",
        return_polar: bool = False,
    ) -> np.ndarray:
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
        scaling : str, optional
            Radial scaling mode for polar to Cartesian conversion.
            Default is "linear".
        return_polar : bool, optional
            If True, returns the polar representation instead of converting
            to Cartesian. Default is False.

        Returns
        -------
        np.ndarray
            Reconstructed projection (complex-valued).
            Shape is (num_angular_components, num_radial_components) if return_polar=True,
            otherwise output_shape.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet or orientation_idx is out of bounds.
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
            device=self.device, dtype=torch.complex64
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

                # Undo the radial scaling applied during decomposition
                radial_component = self._undo_radial_scaling(v_k)

                # Add contribution: u * s * v for this (k, eigenvalue) pair
                contribution = u_ki * s_k * radial_component
                polar_projection += torch.outer(angular_component, contribution)

        # Convert back to numpy
        polar_projection_np = polar_projection.cpu().numpy()

        if return_polar:
            return polar_projection_np

        # Convert to Cartesian space
        if output_shape is None:
            output_shape = (self.volume.shape[-2], self.volume.shape[-1])

        cartesian_projection = np.zeros(output_shape, dtype=np.complex128)
        cartesian_projection.real = warp_polar_inverse(
            polar_projection_np.real,
            output_shape=output_shape,
            scaling=scaling,
        )
        cartesian_projection.imag = warp_polar_inverse(
            polar_projection_np.imag,
            output_shape=output_shape,
            scaling=scaling,
        )

        return cartesian_projection

    def construct_cartesian_feature(
        self,
        k_idx: int,
        eig_idx: int,
        output_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            A complex np.ndarray for the feature in Cartesian space.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        polar_feature = self.construct_polar_feature(k_idx, eig_idx)

        if output_shape is None:
            output_shape = (self.volume.shape[-2], self.volume.shape[-1])

        # Transform real and imaginary parts separately using offset polar inverse
        cartesian_feature = np.zeros(output_shape, dtype=np.complex64)

        cartesian_feature.real = warp_offset_polar_inverse(
            polar_feature.real,
            output_shape=output_shape,
        )
        cartesian_feature.imag = warp_offset_polar_inverse(
            polar_feature.imag,
            output_shape=output_shape,
        )

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

    @classmethod
    def from_result(
        cls,
        result: DecompositionResult | str | Path,
        volume: np.ndarray,
        phi_values: np.ndarray | None = None,
        theta_values: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ) -> "PolarProjectionDecomposer":
        """Create a decomposer with a pre-computed result.

        Useful for loading saved decompositions and constructing features
        without re-running the decomposition.

        Parameters
        ----------
        result : DecompositionResult | str | Path
            A DecompositionResult instance or path to a saved result.
        volume : np.ndarray
            The volume (needed for Cartesian feature construction).
        phi_values : np.ndarray | None, optional
            Azimuthal angles. Can be None if only constructing features.
        theta_values : np.ndarray | None, optional
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

        # Create dummy arrays if not provided
        if phi_values is None:
            phi_values = np.zeros(result.num_orientations)
        if theta_values is None:
            theta_values = np.zeros(result.num_orientations)

        instance = cls(volume, phi_values, theta_values, device=device)
        instance._result = result

        return instance
