"""Dataclass for storing decomposition results."""

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
    created_at : str
        ISO format timestamp of when the result was created.
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
                created_at='{self.created_at}'
            )
            """)

        return s.strip()

    def save(self, path: str | Path) -> None:
        """Save decomposition result to disk.

        Parameters
        ----------
        path : str | Path
            Path to save the result. Will be saved as .npz file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        np.savez_compressed(
            path,
            S=self.S,
            U=self.U,
            Vh=self.Vh,
            num_fourier_filters=self.num_fourier_filters,
            num_orientations=self.num_orientations,
            num_angular_components=self.num_angular_components,
            num_radial_components=self.num_radial_components,
            k_max=self.k_max,
            eig_max=self.eig_max,
            is_complex_projection=self.is_complex_projection,
            created_at=self.created_at,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DecompositionResult":
        """Load decomposition result from disk.

        Parameters
        ----------
        path : str | Path
            Path to the saved .npz file.

        Returns
        -------
        DecompositionResult
            The loaded decomposition result.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        return cls(
            S=data["S"],
            U=data["U"],
            Vh=data["Vh"],
            num_fourier_filters=int(data["num_fourier_filters"]),
            num_orientations=int(data["num_orientations"]),
            num_angular_components=int(data["num_angular_components"]),
            num_radial_components=int(data["num_radial_components"]),
            k_max=int(data["k_max"]),
            eig_max=int(data["eig_max"]),
            is_complex_projection=bool(data["is_complex_projection"]),
            created_at=str(data["created_at"]),
        )

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
