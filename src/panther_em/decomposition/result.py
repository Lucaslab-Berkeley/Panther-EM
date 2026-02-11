"""Dataclass for storing decomposition results."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class DecompositionResult:
    """Stores the full results of a polar projection decomposition.

    Attributes
    ----------
    singular_values : np.ndarray
        Complex singular values with shape (k_max, num_radial_components).
    left_singular_vectors : np.ndarray
        Left singular vectors with shape
        (k_max, num_orientations, num_radial_components).
    right_singular_vectors : np.ndarray
        Right singular vectors (radial eigenvectors) with shape
        (k_max, num_radial_components, num_radial_components).
    num_orientations : int
        Number of orientations used in the decomposition.
    num_angular_components : int
        Number of angular components in polar space.
    num_radial_components : int
        Number of radial components in polar space.
    k_max : int
        Maximum angular frequency index used.
    created_at : str
        ISO format timestamp of when the result was created.
    """

    # Core SVD components
    singular_values: np.ndarray
    left_singular_vectors: np.ndarray
    right_singular_vectors: np.ndarray

    # Decomposition metadata
    num_orientations: int
    num_angular_components: int
    num_radial_components: int
    k_max: int

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate shapes after initialization."""
        expected_sv_shape = (self.k_max, self.num_radial_components)
        expected_vec_shape = (
            self.k_max,
            self.num_radial_components,
            self.num_radial_components,
        )

        if self.singular_values.shape != expected_sv_shape:
            raise ValueError(
                f"singular_values shape {self.singular_values.shape} "
                f"does not match expected {expected_sv_shape}"
            )
        if self.right_singular_vectors.shape != expected_vec_shape:
            raise ValueError(
                f"right_singular_vectors shape {self.right_singular_vectors.shape} "
                f"does not match expected {expected_vec_shape}"
            )
            
    def __repr__(self) -> str:
        """String representation of the DecompositionResult."""
        return (
            f"DecompositionResult(num_orientations={self.num_orientations}, "
            f"num_angular_components={self.num_angular_components}, "
            f"num_radial_components={self.num_radial_components}, "
            f"k_max={self.k_max}, created_at='{self.created_at}')"
        )

        return

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
            singular_values=self.singular_values,
            right_singular_vectors=self.right_singular_vectors,
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
            singular_values=data["singular_values"],
            left_singular_vectors=data["left_singular_vectors"],
            right_singular_vectors=data["right_singular_vectors"],
            num_orientations=int(data["num_orientations"]),
            num_angular_components=int(data["num_angular_components"]),
            num_radial_components=int(data["num_radial_components"]),
            k_max=int(data["k_max"]),
            created_at=str(data["created_at"]),
        )

    def get_singular_value(self, k_idx: int, eig_idx: int) -> complex:
        """Get a specific singular value.

        Parameters
        ----------
        k_idx : int
            Angular frequency index.
        eig_idx : int
            Eigenvalue index.

        Returns
        -------
        complex
            The singular value.
        """
        return complex(self.singular_values[k_idx, eig_idx])

    def get_radial_eigenvector(self, k_idx: int, eig_idx: int) -> np.ndarray:
        """Get a specific radial eigenvector.

        Parameters
        ----------
        k_idx : int
            Angular frequency index.
        eig_idx : int
            Eigenvalue index.

        Returns
        -------
        np.ndarray
            The radial eigenvector with shape (num_radial_components,).
        """
        return self.right_singular_vectors[k_idx, :, eig_idx]
