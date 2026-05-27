"""Dataclass for storing decomposition results."""

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DecompositionResult:
    """Stores the full results of a polar projection decomposition.

    Note
    ----
    The structure of the block-circulant SVD imposes a structure on the singular values
    where each singular value is associated with a specific angular frequency index
    `k_idx` in the range of `[0, k_max)` and a radial eigenvalue index `eig_idx` in the
    range of `[0, eig_max)`.

    For complex-valued underlying projection data there will be
    `num_freq_blocks = k_max * 2 + 1` total frequency blocks corresponding to the
    positive and negative frequencies including the DC (k=0) component. When projection
    data is real-valued, only the non-negative frequencies are stored, so
    `num_freq_blocks = k_max + 1`, but negative frequencies can still be indexed.

    The left-singular-vectors (`U`) have shape (..., num_freq_blocks, eig_idx), the
    singular-values (`S`) have shape (num_freq_blocks, eig_idx), and the
    right-singular-vectors (`Vh`) have shape (num_freq_blocks, eig_idx,
    num_radial_components). Helper methods exist for querying the top L pairs based on
    singular value magnitude; similar methods exist for retrieving the corresponding
    singular vectors and values.

    Results are saved and loaded as HDF5 files (``.h5`` / ``.hdf5``) via
    :meth:`save` and :meth:`load`.

    Attributes
    ----------
    U : np.ndarray
        Left singular vectors with shape
        (num_fourier_filters, num_orientations, num_freq_blocks, num_radial_components).
    S : np.ndarray
        Singular values with shape (num_freq_blocks, num_radial_components).
    Vh : np.ndarray
        Right singular vectors (conjugate transpose) with shape
        (num_freq_blocks, num_radial_components, num_radial_components).
    num_fourier_filters : int
        Number of Fourier filters (defocus channels) used in decomposition.
    num_orientations : int
        Number of orientations used in the decomposition.
    num_angular_components : int
        Number of angular components in polar space.
    num_radial_components : int
        Number of radial components in polar space. Also the number of radial
        eigenvectors stored per angular frequency component
    k_max : int
        Maximum angular frequency index used.
    eig_max : int
        Maximum radial eigenvalue index used.
    is_complex_projection : bool
        Whether the underlying projection data is complex-valued.
    created_at : str
        ISO format timestamp of when the result was created.
    phi_values : np.ndarray | None
        Phi (azimuthal) ZYZ Euler angles in degrees used during decomposition, shape
        ``(num_orientations,)``. ``None`` if not recorded.
    theta_values : np.ndarray | None
        Theta (polar) ZYZ Euler angles in degrees used during decomposition, shape
        ``(num_orientations,)``. ``None`` if not recorded.
    fourier_filters : np.ndarray | None
        Fourier-space filters (e.g. CTF envelopes) applied during decomposition, shape
        ``(num_fourier_filters, ...)``. ``None`` if no filters were applied.
    """

    # Core SVD components
    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray

    # Shapes for singular values
    k_max: int
    eig_max: int
    is_complex_projection: bool

    # Decomposition metadata
    num_fourier_filters: int
    num_orientations: int
    num_angular_components: int
    num_radial_components: int

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Optional orientation recorded at decomposition time
    phi_values: np.ndarray | None = None
    theta_values: np.ndarray | None = None

    # Optional Fourier filters recorded at decomposition time
    fourier_filters: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate shapes after initialization."""
        if self.is_complex_projection:
            num_freq_blocks = self.k_max * 2  # NOTE: need plus 1?
        else:
            num_freq_blocks = self.k_max  # NOTE: need plus 1?

        expected_U_shape = (
            self.num_fourier_filters,
            self.num_orientations,
            num_freq_blocks,
            self.eig_max,
        )
        expected_S_shape = (num_freq_blocks, self.eig_max)
        expected_Vh_shape = (
            num_freq_blocks,
            self.eig_max,
            self.num_radial_components,
        )

        if self.S.shape != expected_S_shape:
            raise ValueError(
                f"S shape {self.S.shape} does not match expected {expected_S_shape}"
            )
        if self.U.shape != expected_U_shape:
            raise ValueError(
                f"U shape {self.U.shape} does not match expected {expected_U_shape}"
            )
        if self.Vh.shape != expected_Vh_shape:
            raise ValueError(
                f"Vh shape {self.Vh.shape} does not match expected {expected_Vh_shape}"
            )

        # Validate optional angle arrays
        if self.phi_values is not None:
            if self.phi_values.shape != (self.num_orientations,):
                raise ValueError(
                    f"phi_values shape {self.phi_values.shape} does not match "
                    f"expected ({self.num_orientations},)"
                )
        if self.theta_values is not None:
            if self.theta_values.shape != (self.num_orientations,):
                raise ValueError(
                    f"theta_values shape {self.theta_values.shape} does not match "
                    f"expected ({self.num_orientations},)"
                )

        # Validate optional Fourier filters
        if self.fourier_filters is not None:
            if self.fourier_filters.shape[0] != self.num_fourier_filters:
                raise ValueError(
                    f"fourier_filters leading dimension "
                    f"{self.fourier_filters.shape[0]} does not match "
                    f"num_fourier_filters={self.num_fourier_filters}"
                )

        # Per-instance cache for top-n queries
        self._top_n_cache: dict[tuple[int, bool], np.ndarray] = {}

    def __repr__(self) -> str:
        """String representation of the DecompositionResult."""
        s = textwrap.dedent(f"""
            DecompositionResult(
                k_max={self.k_max},
                eig_max={self.eig_max},
                is_complex_projection={self.is_complex_projection},
                num_fourier_filters={self.num_fourier_filters},
                num_orientations={self.num_orientations},
                num_angular_components={self.num_angular_components},
                num_radial_components={self.num_radial_components},
                has_euler_angles={self.phi_values is not None},
                has_fourier_filters={self.fourier_filters is not None},
                created_at='{self.created_at}'
            )
            """)

        return s.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_hdf5_metadata(self, f: h5py.File) -> None:
        """Write scalar metadata as root-level HDF5 attributes."""
        f.attrs["k_max"] = self.k_max
        f.attrs["eig_max"] = self.eig_max
        f.attrs["is_complex_projection"] = self.is_complex_projection
        f.attrs["num_fourier_filters"] = self.num_fourier_filters
        f.attrs["num_orientations"] = self.num_orientations
        f.attrs["num_angular_components"] = self.num_angular_components
        f.attrs["num_radial_components"] = self.num_radial_components
        f.attrs["created_at"] = self.created_at

    def _write_hdf5_optional_arrays(self, f: h5py.File) -> None:
        """Write optional orientation angles and Fourier filters if present."""
        if self.phi_values is not None:
            f.create_dataset("phi_values", data=self.phi_values)
        if self.theta_values is not None:
            f.create_dataset("theta_values", data=self.theta_values)
        if self.fourier_filters is not None:
            f.create_dataset("fourier_filters", data=self.fourier_filters)

    def save(self, path: str | Path) -> None:
        """Save the full decomposition result to an HDF5 file.

        Scalar metadata is stored as root-level HDF5 attributes.  To save only
        the most significant components and reduce file size, use
        :meth:`save_top_n` instead.

        Parameters
        ----------
        path : str | Path
            Destination path.  The extension is replaced with ``.h5`` if it
            is not already ``.h5`` or ``.hdf5``.
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")

        with h5py.File(path, "w") as f:
            self._write_hdf5_metadata(f)
            f.create_dataset("U", data=self.U)
            f.create_dataset("S", data=self.S)
            f.create_dataset("Vh", data=self.Vh)
            self._write_hdf5_optional_arrays(f)

    def save_top_n(self, path: str | Path, top_k: int) -> None:
        r"""Save only the ``top_k`` most significant components to HDF5.

        Selects the ``top_k`` unique ``(k_stored, eig_idx)`` pairs ranked by
        ``|S|`` and writes only the corresponding slices of ``U``, ``S``, and
        ``Vh``.  The file is tagged ``is_sparse=True`` so :meth:`load`
        transparently reconstructs full-shape zero-padded arrays, leaving all
        downstream code (``get_top_n``, ``get_component``, ``compute_weights``,
        etc.) unchanged.

        File size is :math:`O(top\\_k)` rather than :math:`O(k\\_max \\times
        eig\\_max)`.

        Parameters
        ----------
        path : str | Path
            Destination path.  Extension is normalized to ``.h5``.
        top_k : int
            Number of unique ``(k_stored, eig_idx)`` pairs to store.

        Raises
        ------
        ValueError
            If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")

        # include_negative=False gives unique stored pairs (no double-counting
        # for real decompositions where ±k share the same stored row)
        pairs = self.get_top_n(top_k, include_negative=False)
        pairs = pairs[:top_k]

        k_stored_arr = pairs[:, 0].astype(int)
        eig_idx_arr = pairs[:, 1].astype(int)

        # Compact slices — shapes (num_ff, num_or, L), (L,), (L, num_r)
        U_sparse = self.U[..., k_stored_arr, eig_idx_arr]
        S_sparse = self.S[k_stored_arr, eig_idx_arr]
        Vh_sparse = self.Vh[k_stored_arr, eig_idx_arr, :]
        selected_indices = np.stack([k_stored_arr, eig_idx_arr], axis=1)

        with h5py.File(path, "w") as f:
            f.attrs["is_sparse"] = True
            self._write_hdf5_metadata(f)
            f.create_dataset("selected_indices", data=selected_indices)
            f.create_dataset("U", data=U_sparse)
            f.create_dataset("S", data=S_sparse)
            f.create_dataset("Vh", data=Vh_sparse)
            self._write_hdf5_optional_arrays(f)

    @classmethod
    def load(cls, path: str | Path) -> "DecompositionResult":
        """Load a decomposition result from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to the ``.h5`` or ``.hdf5`` file.

        Returns
        -------
        DecompositionResult
            The loaded decomposition result.

        Raises
        ------
        ValueError
            If the file extension is not ``.h5`` or ``.hdf5``.
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unknown file extension '{path.suffix}'. Expected '.h5' or '.hdf5'."
            )
        return cls._load_hdf5(path)

    @classmethod
    def _load_hdf5(cls, path: Path) -> "DecompositionResult":
        """Load from an HDF5 file produced by :meth:`save` or :meth:`save_top_n`.

        When the file was written by :meth:`save_top_n` (``is_sparse=True``
        attribute present), the compact sparse arrays are scattered back into
        full zero-padded dense arrays so that all downstream indexing code works
        without modification.
        """
        with h5py.File(path, "r") as f:
            is_sparse = bool(f.attrs.get("is_sparse", False))

            k_max = int(f.attrs["k_max"])
            eig_max = int(f.attrs["eig_max"])
            is_complex = bool(f.attrs["is_complex_projection"])
            num_ff = int(f.attrs["num_fourier_filters"])
            num_or = int(f.attrs["num_orientations"])
            num_ang = int(f.attrs["num_angular_components"])
            num_r = int(f.attrs["num_radial_components"])

            # h5py >= 3 returns str; older versions may return bytes
            created_at = f.attrs["created_at"]
            if isinstance(created_at, bytes):
                created_at = created_at.decode("utf-8")

            phi_values = f["phi_values"][()] if "phi_values" in f else None
            theta_values = f["theta_values"][()] if "theta_values" in f else None
            fourier_filters = (
                f["fourier_filters"][()] if "fourier_filters" in f else None
            )

            if is_sparse:
                # Scatter compact arrays into zero-filled dense arrays so that
                # all existing indexing (get_top_n, get_component, compute_weights)
                # works without any downstream changes.
                num_freq_blocks = k_max * 2 if is_complex else k_max
                selected_indices = f["selected_indices"][()]  # (L, 2)
                U_sparse = f["U"][()]  # (num_ff, num_or, L)
                S_sparse = f["S"][()]  # (L,)
                Vh_sparse = f["Vh"][()]  # (L, num_r)

                k_stored_arr = selected_indices[:, 0]
                eig_idx_arr = selected_indices[:, 1]

                U = np.zeros(
                    (num_ff, num_or, num_freq_blocks, eig_max), dtype=np.complex64
                )
                S = np.zeros((num_freq_blocks, eig_max), dtype=np.float32)
                Vh = np.zeros((num_freq_blocks, eig_max, num_r), dtype=np.complex64)

                U[..., k_stored_arr, eig_idx_arr] = U_sparse
                S[k_stored_arr, eig_idx_arr] = S_sparse
                Vh[k_stored_arr, eig_idx_arr, :] = Vh_sparse
            else:
                U = f["U"][()]
                S = f["S"][()]
                Vh = f["Vh"][()]

            return cls(
                U=U,
                S=S,
                Vh=Vh,
                k_max=k_max,
                eig_max=eig_max,
                is_complex_projection=is_complex,
                num_fourier_filters=num_ff,
                num_orientations=num_or,
                num_angular_components=num_ang,
                num_radial_components=num_r,
                created_at=created_at,
                phi_values=phi_values,
                theta_values=theta_values,
                fourier_filters=fourier_filters,
            )

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def k_to_stored(self, k_idx: int | np.ndarray) -> int | np.ndarray:
        """Convert an angular-frequency index to its storage-array row index."""
        if self.is_complex_projection:
            return k_idx + self.k_max

        # Since singular values are real-valued, complex conjugate returns same
        return np.abs(k_idx) if isinstance(k_idx, np.ndarray) else abs(k_idx)

    def is_conjugate_mode(self, k_idx: int | np.ndarray) -> bool | np.ndarray:
        """Helper for determining when conj is needed across real/complex modes."""
        # Always False for complex-valued decompositions
        if self.is_complex_projection:
            return (
                np.zeros_like(k_idx, dtype=bool)
                if isinstance(k_idx, np.ndarray)
                else False
            )

        # True when k_idx is negative for real-valued decompositions
        return (
            (k_idx < 0)
            if isinstance(k_idx, (int, np.integer))
            else (np.asarray(k_idx) < 0)
        )

    def get_top_n(self, top_k: int, include_negative: bool = True) -> np.ndarray:
        """Get the top-n singular values across all (frequency, radial) pairs.

        Parameters
        ----------
        top_k : int
            The number of top singular values to retrieve.
        include_negative : bool, optional
            If this is a real-valued decomposition, this flag chooses whether to include
            negative frequency indices (default True) or to only include non-negative
            angular frequencies (False). No effect when is_complex_projection is True.

        Returns
        -------
        np.ndarray
            An array of shape (l, 2) where l <= top_k containing the (k_idx, eig_idx)
            pairs corresponding to the top singular values sorted by decreasing
            magnitude.
        """
        cache_key = (top_k, include_negative)
        if cache_key not in self._top_n_cache:
            all_svs = self.S.flatten()
            all_svs_sorted = np.argsort(np.abs(all_svs))[::-1]
            top_indices = all_svs_sorted[:top_k]
            k_indices, eig_indices = np.unravel_index(top_indices, self.S.shape)

            # Handle real-valued decomposition case where must also include negative k
            # indices for each positive k index
            if not self.is_complex_projection and include_negative:
                tmp_k_indices = []
                tmp_eig_indices = []

                for k_idx, eig_idx in zip(k_indices, eig_indices, strict=False):
                    if k_idx == 0:
                        tmp_k_indices.append(k_idx)
                        tmp_eig_indices.append(eig_idx)
                    else:
                        tmp_k_indices.extend([k_idx, -k_idx])
                        tmp_eig_indices.extend([eig_idx, eig_idx])

                    # Exit condition since could be 2x number of entries
                    if len(tmp_k_indices) >= top_k:
                        break

                k_indices = np.array(tmp_k_indices)[:top_k]
                eig_indices = np.array(tmp_eig_indices)[:top_k]

            self._top_n_cache[cache_key] = np.stack((k_indices, eig_indices), axis=-1)

        return self._top_n_cache[cache_key]

    def get_component(
        self,
        k_idx: int | np.ndarray,  # shape (l,)
        eig_idx: int | np.ndarray,  # shape (l,)
        return_u: bool = True,
        return_s: bool = True,
        return_vh: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Helper method for retrieving specific components of the SVD.

        Parameters
        ----------
        k_idx : int | np.ndarray
            Actual angular-frequency index or indices.  Valid range is
            ``[0, k_max]`` for real-valued and ``[-k_max, k_max]`` for complex
            decompositions.
        eig_idx : int | np.ndarray
            Eigenvalue index or indices.
        return_u : bool, optional
            Whether to return the left singular vectors. Default is True.
        return_s : bool, optional
            Whether to return the singular values. Default is True.
        return_vh : bool, optional
            Whether to return the right singular vectors. Default is True.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
            A tuple containing the requested components. Elements not selected by
            (return_u, return_s, return_vh) will be None.
        """
        # Must be returning at least one component
        if not (return_u or return_s or return_vh):
            raise ValueError(
                "At least one of return_u, return_s, return_vh must be True."
            )

        k_idx_arr = np.atleast_1d(k_idx)
        k_stored = np.atleast_1d(self.k_to_stored(k_idx_arr))
        eig_idx = np.atleast_1d(eig_idx)
        conj_mask = np.atleast_1d(self.is_conjugate_mode(k_idx_arr))

        u = None
        s = None
        vh = None

        if return_u:
            u = self.U[..., k_stored, eig_idx]
            if conj_mask.any():
                u = u.copy()
                u[..., conj_mask] = u[..., conj_mask].conj()
        if return_s:
            s = self.S[k_stored, eig_idx]
        if return_vh:
            vh = self.Vh[k_stored, eig_idx]
            if conj_mask.any():
                vh = vh.copy()
                vh[conj_mask] = vh[conj_mask].conj()

        return u, s, vh

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def scree_plot(
        self, k_idx: int | None = None, **kwargs: dict[str, Any]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Helper function for plotting sorted singular values vs index.

        Parameters
        ----------
        k_idx : int | None, optional
            Actual angular frequency to plot. Valid range is ``[0, k_max]``
            for real-valued and ``[-k_max, k_max]`` for complex decompositions.
            If None, plot all singular values across all frequencies, sorted by
            decreasing magnitude. Default is None.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to plt.subplots()

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib figure object and axes containing the scree plot.
        """
        fig, ax = plt.subplots(**kwargs)
        svs = (
            self.S[self.k_to_stored(k_idx), :]
            if k_idx is not None
            else self.S.flatten()
        )
        title = (
            f"Scree Plot for k={k_idx}"
            if k_idx is not None
            else "Scree Plot for All Singular Values"
        )

        sorted_svs = svs[np.argsort(np.abs(svs))[::-1]]
        ax.plot(sorted_svs)
        ax.set_title(title)
        ax.set_xlabel("Index (sorted by magnitude)")
        ax.set_ylabel("Singular Value (magnitude)")
        return fig, ax

    def variance_explained_plot(
        self,
        k_idx: int | None = None,
        inverted: bool = True,
        **kwargs: dict[str, Any],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Helper function for plotting cumulative variance explained.

        Parameters
        ----------
        k_idx : int | None, optional
            Actual angular frequency to plot.  Valid range is ``[0, k_max]``
            for real-valued and ``[-k_max, k_max]`` for complex decompositions.
            If None, plot for all frequencies. Default is None.
        inverted : bool, optional
            If True, plot (1 - var exp) rather than (var exp). Default is True.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to plt.subplots()

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib figure object and axes containing the variance explained
            plot.
        """
        fig, ax = plt.subplots(**kwargs)
        svs = (
            self.S[self.k_to_stored(k_idx), :]
            if k_idx is not None
            else self.S.flatten()
        )
        title = (
            f"Cumulative Variance Explained for k={k_idx}"
            if k_idx is not None
            else "Cumulative Variance Explained for All Singular Values"
        )

        sorted_svs = svs[np.argsort(np.abs(svs))[::-1]]
        denom = np.sum(np.abs(sorted_svs) ** 2)
        variance_explained = np.cumsum(np.abs(sorted_svs) ** 2) / denom
        if inverted:
            variance_explained = 1 - variance_explained

        ax.plot(variance_explained)
        ax.set_title(title)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Explained")
        return fig, ax
