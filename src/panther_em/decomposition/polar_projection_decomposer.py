"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path

import numpy as np

from panther_em.decomposition.result import DecompositionResult
from panther_em.utils import (
    get_polar_projections_from_volume,
    warp_polar_inverse,
)


class PolarProjectionDecomposer:
    """Manages projection generation and block-circulant data decomposition.

    NOTE: The Euler angles (phi, theta) represent the first two angles in ZYZ format
    where the final in-plane rotation angle (psi) is assumed to be zero based on the
    polar projection model.

    Parameters
    ----------
    phi_values : np.ndarray
        Phi values, in degrees of ZYZ Euler angles, for projection orientations.
    theta_values : np.ndarray
        Theta values, in degrees of ZYZ Euler angles, for projection orientations.
    volume : np.ndarray
        The 3D volume to decompose.

    Attributes
    ----------
    result : DecompositionResult | None
        The decomposition result after calling `do_decomposition()`.

    Examples
    --------
    >>> # assume volume, phi, theta are defined
    >>> decomposer = PolarProjectionDecomposer(volume, phi, theta)
    >>> result = decomposer.do_decomposition()
    >>> result.save("decomposition.npz")
    """

    def __init__(
        self,
        volume: np.ndarray,
        phi_values: np.ndarray,
        theta_values: np.ndarray,
    ) -> None:
        self.phi_values = phi_values
        self.theta_values = theta_values
        self.volume = volume
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
        projections_polar = get_polar_projections_from_volume(
            volume=self.volume,
            phi=self.phi_values,
            theta=self.theta_values,
            psi=0.0,
        )
        num_orients, num_angular_comp, num_radial_comp = projections_polar.shape
        
        ### DEBUGGING: Print the shape of the projections
        print(f"DEBUG - projections_polar shape: {projections_polar.shape}")
        print(f"DEBUG - num_orients: {num_orients}")
        print(f"DEBUG - num_angular_comp: {num_angular_comp}")
        print(f"DEBUG - num_radial_comp: {num_radial_comp}")
        ### END DEBUGGING

        projections_fft = np.fft.fft(projections_polar, axis=1)  # Along angular dim

        # Determine k_max
        if k_max is None:
            k_max = num_angular_comp

        # Iterate over all angular frequency components and store the SVD results
        singular_values = np.zeros((k_max, num_orients), dtype=np.complex64)
        left_singular_vectors = np.zeros(
            (k_max, num_orients, num_radial_comp), dtype=np.complex64
        )
        right_singular_vectors = np.zeros(
            (k_max, num_radial_comp, num_radial_comp), dtype=np.complex64
        )
        for k in range(k_max):
            freq_block = projections_fft[:, k, :]

            # Scale by the radial component (proper integration term $r dr$)
            r = np.arange(num_radial_comp)
            r = r / num_radial_comp
            freq_block_scaled = freq_block * r[None, :]

            # Call SVD on the scaled frequency block
            u, s, vh = np.linalg.svd(
                freq_block_scaled, full_matrices=False, compute_uv=True
            )
            
            ### DEBUGGING: Print the shapes of the SVD outputs
            print(f"DEBUG - k={k}: freq_block shape: {freq_block.shape}")
            print(f"DEBUG - k={k}: u shape: {u.shape}, s shape: {s.shape}, vh shape: {vh.shape}")
            ### END DEBUGGING
            
            singular_values[k] = s
            left_singular_vectors[k] = u
            right_singular_vectors[k] = vh.conj().T

        # # # Evaluate the SVD separately on each frequency block
        # # singular_values = np.zeros((k_max, num_radial_comp), dtype=np.complex64)
        # # singular_vectors = np.zeros(
        # #     (k_max, num_radial_comp, num_radial_comp), dtype=np.complex64
        # # )

        # for k in range(k_max):
        #     freq_block = projections_fft[:, k, :]
        #     freq_block_conj = np.conjugate(freq_block)
        #     radial_block = freq_block_conj.T @ freq_block

        #     # Reweight by the radius of the components
        #     r = np.arange(num_radial_comp)
        #     radial_block_scaled = (
        #         (r[:, None] * r[None, :]) * radial_block / (num_radial_comp**2)
        #     )

        #     # Do eigenvalue decomposition
        #     values, vectors = np.linalg.eig(radial_block_scaled)

        #     # Save singular values and radial component of singular vectors
        #     singular_values[k] = np.sqrt(values)
        #     singular_vectors[k] = vectors

        # Create and store the result
        self._result = DecompositionResult(
            singular_values=singular_values,
            left_singular_vectors=left_singular_vectors,
            right_singular_vectors=right_singular_vectors,
            num_orientations=num_orients,
            num_angular_components=num_angular_comp,
            num_radial_components=num_radial_comp,
            k_max=k_max,
        )

        return self._result

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

        # Compute the magnitude and phase of the feature
        cplx_magnitude_by_radius = result.get_radial_eigenvector(k_idx, eig_idx)
        cplx_phase_by_angle = np.exp(
            1j
            * 2
            * np.pi
            * k_idx
            * np.arange(result.num_angular_components)
            / result.num_angular_components
        ) / np.sqrt(result.num_angular_components)

        # Construct the feature by taking the outer product
        polar_feature = np.outer(cplx_phase_by_angle, cplx_magnitude_by_radius)

        return polar_feature

    def construct_cartesian_feature(
        self,
        k_idx: int,
        eig_idx: int,
        output_shape: tuple[int, int] | None = None,
        scaling: str = "linear",
    ) -> np.ndarray:
        """Construct a single Cartesian feature for an angular frequency and eigenvalue.

        Parameters
        ----------
        k_idx : int
            The index of the angular frequency component.
        eig_idx : int
            The index of the eigenvector to use.
        output_shape : tuple[int, int] | None, optional
            Shape of the output Cartesian image. If None, uses the volume's
            spatial dimensions. Default is None.
        scaling : str, optional
            Radial scaling mode for polar to Cartesian conversion.
            Default is "linear".

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

        cartesian_feature = np.zeros(output_shape, dtype=np.complex64)

        cartesian_feature.real = warp_polar_inverse(
            polar_feature.real,
            output_shape=output_shape,
            scaling=scaling,
        )
        cartesian_feature.imag = warp_polar_inverse(
            polar_feature.imag,
            output_shape=output_shape,
            scaling=scaling,
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

        instance = cls(volume, phi_values, theta_values)
        instance._result = result

        return instance
