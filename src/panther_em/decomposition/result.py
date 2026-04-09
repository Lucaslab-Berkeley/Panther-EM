"""Dataclass for storing decomposition results."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import textwrap
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
    range of `[0, eig_max)`. Usually, `eig_max = num_radial_components`, but this can be
    less if only a subset of eigenvalues were retained during decomposition.

    The left-singular-vectors (`U`) have shape (..., k_idx, eig_idx), the
    singular-values (`S`) have shape (k_idx, eig_idx), and the right-singular-vectors
    (`Vh`) have shape (k_idx, eig_idx, num_radial_components). Helper methods exist for
    querying the top L pairs based on singular value magnitude; similar methods exist
    for retrieving the corresponding singular vectors and values.

    Attributes
    ----------
    U : np.ndarray
        Left singular vectors with shape
        (num_fourier_filters, num_orientations, k_max, num_radial_components).
    S : np.ndarray
        Singular values with shape (k_max, num_radial_components).
    Vh : np.ndarray
        Right singular vectors (conjugate transpose) with shape
        (k_max, num_radial_components, num_radial_components).
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

    # Decomposition metadata
    num_fourier_filters: int
    num_orientations: int
    num_angular_components: int
    num_radial_components: int

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate shapes after initialization."""
        expected_U_shape = (
            self.num_fourier_filters,
            self.num_orientations,
            self.k_max,
            self.eig_max,
        )
        expected_S_shape = (self.k_max, self.eig_max)
        expected_Vh_shape = (
            self.k_max,
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

    def __repr__(self) -> str:
        """String representation of the DecompositionResult."""
        s = textwrap.dedent(
            f"""
            DecompositionResult(
                k_max={self.k_max},
                eig_max={self.eig_max},
                num_fourier_filters={self.num_fourier_filters},
                num_orientations={self.num_orientations},
                num_angular_components={self.num_angular_components},
                num_radial_components={self.num_radial_components},
                created_at='{self.created_at}'
            )
            """
        )

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
            created_at=str(data["created_at"]),
        )

    def get_top_k(self, top_k: int) -> np.ndarray:
        """Get the top-k singular values across all (frequency, radial) pairs.

        Parameters
        ----------
        top_k : int
            Number of top singular values to retrieve.

        Returns
        -------
        np.ndarray
            Indices of (k_idx, eig_idx) pairs, sorted by magnitude. Shape of (top_k, 2).
        """
        all_svs = self.S.flatten()
        all_svs_sorted = np.argsort(np.abs(all_svs))[::-1]  # reverse for largest first
        top_indices = all_svs_sorted[:top_k]
        k_indices, eig_indices = np.unravel_index(top_indices, self.S.shape)

        return np.stack((k_indices, eig_indices), axis=-1)

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
            Angular frequency index or indices to retrieve.
        eig_idx : int | np.ndarray
            Eigenvalue index or indices to retrieve.
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

        k_idx = np.atleast_1d(k_idx)
        eig_idx = np.atleast_1d(eig_idx)

        u = None
        s = None
        vh = None

        if return_u:
            u = self.U[..., k_idx, eig_idx]
        if return_s:
            s = self.S[k_idx, eig_idx]
        if return_vh:
            vh = self.Vh[k_idx, eig_idx]

        return u, s, vh

    def scree_plot(
        self, k_idx: int | None = None, **kwargs: dict[str, Any]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Helper function for plotting sorted singular values vs index.

        Parameters
        ----------
        k_idx : int | None, optional
            If specified, plot singular values for this specific angular frequency
            index. If None, plot all singular values across all k indices, sorted by
            decreasing magnitude. Default is None.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to plt.subplots()

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib figure object and axes containing the scree plot.
        """
        fig, ax = plt.subplots(**kwargs)
        svs = self.S[k_idx, :] if k_idx is not None else self.S.flatten()
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
            If specified, plot variance explained for this specific angular frequency
            index. If None, plot for all singular values across all k indices.
            Default is None.
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
        svs = self.S[k_idx, :] if k_idx is not None else self.S.flatten()
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
