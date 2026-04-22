"""Inference helpers for projection/feature reconstruction from decomp. results."""

import numpy as np
import torch

from panther_em.decomposition.result import DecompositionResult
from panther_em.utils.warp_transforms import OffsetPolarTransform


class ProjectionReconstructor:
    """Helper for reconstructing polar/cartesian features from a result."""

    def __init__(
        self,
        result: DecompositionResult,
        image_shape: tuple[int, int],
        device: str | torch.device = "cpu",
    ) -> None:
        self.result = result
        self.device = torch.device(device)
        self.image_shape = image_shape

        transform_device = "cuda" if self.device.type == "cuda" else "numpy"
        self._polar_transform = OffsetPolarTransform.from_image(
            image_shape=image_shape,
            num_angle=result.num_angular_components,
            num_radius=result.num_radial_components,
            device=transform_device,  # type: ignore
        )

        # self._U = torch.from_numpy(result.U).to(
        #     device=self.device, dtype=torch.complex64
        # )
        # self._S = torch.from_numpy(result.S).to(
        #     device=self.device, dtype=torch.complex64
        # )
        # self._Vh = torch.from_numpy(result.Vh).to(
        #     device=self.device, dtype=torch.complex64
        # )

    def clear_polar_transform_cache(self) -> None:
        """Remove any cached interpolation grids."""
        self._polar_transform.clear_cache()

    def _get_angular_phase_component(self, k_idx: int) -> torch.Tensor:
        """Private helper to compute phase component (circular modes)."""
        angles = (
            2
            * np.pi
            * k_idx
            * torch.arange(
                self.result.num_angular_components,
                device=self.device,
                dtype=torch.float32,
            )
            / self.result.num_angular_components
        )
        return torch.exp(1j * angles) / np.sqrt(self.result.num_angular_components)

    def construct_polar_feature(
        self, k_idx: int, eig_idx: int, return_torch: bool = False
    ) -> np.ndarray | torch.Tensor:
        """Construct a polar feature for a given k_idx and eig_idx.

        Parameters
        ----------
        k_idx : int
            Index of the angular component (circular mode).
        eig_idx : int
            Index of the radial eigenvector (component).
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.

        Returns
        -------
        np.ndarray or torch.Tensor
            The constructed polar feature as a 2D array (angular x radial) with shape
            (num_angular_components, num_radial_components).
        """
        angular_component = self._get_angular_phase_component(k_idx)
        _, _, vh = self.result.get_component(
            k_idx, eig_idx, return_u=False, return_s=False, return_vh=True
        )

        # Squeeze out out singleton dimensions
        vh = np.squeeze(vh)

        v_tensor = torch.from_numpy(vh).to(device=self.device, dtype=torch.complex64)
        polar_feature = torch.outer(angular_component, v_tensor)

        return polar_feature if return_torch else polar_feature.cpu().numpy()

    def construct_cartesian_feature(
        self,
        k_idx: int,
        eig_idx: int,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Construct the cartesian feature for a given k_idx and eig_idx.

        Parameters
        ----------
        k_idx : int
            Index of the angular component (circular mode).
        eig_idx : int
            Index of the radial eigenvector (component).
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.
        order : int, optional
            The order of the interpolation used in the polar to cartesian
            transformation, by default 5.
        mode : str, optional
            The mode parameter determines how the input array is extended when the
            transformation requires values outside of the input boundaries. By default
            "constant".
        cval : float, optional
            The value to fill past edges of input if mode is "constant", by default 0.0.
        preserve_energy : bool, optional
            Whether to preserve the energy of the feature during the polar to cartesian
            transformation, by default True.
        wrap_angular_axis : bool, optional
            Whether to wrap the angular axis during the polar to cartesian
            transformation, by default True.

        Returns
        -------
        np.ndarray or torch.Tensor
            The constructed cartesian feature as a 2D array with shape corresponding to
            the original image dimensions.
        """
        polar_feature = self.construct_polar_feature(k_idx, eig_idx, return_torch=True)

        cartesian_feature = self._polar_transform.to_cartesian(
            polar_feature,
            order=order,
            mode=mode,
            cval=cval,
            preserve_energy=preserve_energy,
            wrap_angular_axis=wrap_angular_axis,
        )

        return cartesian_feature if return_torch else cartesian_feature.cpu().numpy()

    def reconstruct_projection(
        self,
        orientation_idx: int,
        fourier_filter_idx: int = 0,
        num_components: int | None = None,
        return_polar: bool = False,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Reconstruct a projection for a given number of top components.

        Parameters
        ----------
        orientation_idx : int
            Index of the orientation (in-plane rotation) to reconstruct.
        fourier_filter_idx : int, optional
            Index of the Fourier filter to reconstruct, by default 0.
        num_components : int | None, optional
            Number of top components (by singular value magnitude) to use for
            reconstruction. If None, uses all available components, by default None.
        return_polar : bool, optional
            Whether to return the reconstructed projection in polar coordinates instead
            of cartesian, by default False.
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.
        order : int, optional
            The order of the interpolation used in the polar to cartesian
            transformation, by default 5.
        mode : str, optional
            The mode parameter determines how the input array is extended when the
            transformation requires values outside of the input boundaries. By default
            "constant".
        cval : float, optional
            The value to fill past edges of input if mode is "constant", by default 0.0.
        preserve_energy : bool, optional
            Whether to preserve the energy of the feature during the polar to cartesian
            transformation, by default True.
        wrap_angular_axis : bool, optional
            Whether to wrap the angular axis during the polar to cartesian
            transformation, by default True.

        Returns
        -------
        np.ndarray or torch.Tensor
            The reconstructed projection as a 2D array in either polar or cartesian
            coordinates, depending on the value of `return_polar`. The shape will be
            (num_angular_components, num_radial_components) for polar coordinates or the
            original image dimensions for cartesian coordinates.
        """
        if not (0 <= fourier_filter_idx < self.result.num_fourier_filters):
            raise ValueError(
                f"fourier_filter_idx {fourier_filter_idx} out of bounds "
                f"(max: {self.result.num_fourier_filters - 1})"
            )
        if not (0 <= orientation_idx < self.result.num_orientations):
            raise ValueError(
                f"orientation_idx {orientation_idx} out of bounds "
                f"(max: {self.result.num_orientations - 1})"
            )

        if num_components is None:
            num_components = self.result.k_max * self.result.eig_max
        else:
            num_components = min(
                num_components, self.result.k_max * self.result.eig_max
            )

        top_k_indices = self.result.get_top_k(num_components)

        # Accumulate weighted features in polar space, only one warp transform needed.
        polar_projection = torch.zeros(
            (self.result.num_angular_components, self.result.num_radial_components),
            dtype=torch.complex64,
            device=self.device,
        )

        # Group indices by k_idx for efficient accumulation
        k_idx_groups = {k_idx: [] for k_idx in range(self.result.k_max)}
        for k_idx, eig_idx in top_k_indices:
            k_idx_groups[k_idx].append(eig_idx)

        for k_idx in sorted(k_idx_groups.keys()):
            eig_indices = k_idx_groups[k_idx]

            # If no components to process
            if not eig_indices:
                continue

            eig_indices = np.array(eig_indices)
            angular_component = self._get_angular_phase_component(k_idx)

            u, s, vh = self.result.get_component(
                k_idx=int(k_idx),
                eig_idx=eig_indices,
                return_u=True,
                return_s=True,
                return_vh=True,
            )

            assert u is not None
            assert s is not None
            assert vh is not None

            # Extract the specific fourier_filter and orientation
            u_ki = u[fourier_filter_idx, orientation_idx]  # shape (num_components,)
            s_k = s  # shape (num_components,)
            vh_k = vh  # shape (num_components, num_radial_components)

            # Convert to torch tensors
            u_ki = np.array(u_ki, dtype=np.complex64)
            u_ki = torch.from_numpy(u_ki).to(device=self.device, dtype=torch.complex64)

            s_k = np.array(s_k, dtype=np.complex64)
            s_k = torch.from_numpy(s_k).to(device=self.device, dtype=torch.complex64)

            vh_k = torch.from_numpy(vh_k).to(device=self.device, dtype=torch.complex64)

            # Mathematical reconstruction: U_ki * S_k * Vh_k
            weighted_features = u_ki * s_k
            radial_contribution = weighted_features @ vh_k

            polar_projection += torch.outer(angular_component, radial_contribution)

        if return_polar:
            return polar_projection if return_torch else polar_projection.cpu().numpy()

        cartesian_projection = self._polar_transform.to_cartesian(
            polar_projection,
            order=order,
            mode=mode,
            cval=cval,
            preserve_energy=preserve_energy,
            wrap_angular_axis=wrap_angular_axis,
        )

        if return_torch:
            return cartesian_projection
        return (
            cartesian_projection.cpu().numpy()
            if isinstance(cartesian_projection, torch.Tensor)
            else cartesian_projection
        )
