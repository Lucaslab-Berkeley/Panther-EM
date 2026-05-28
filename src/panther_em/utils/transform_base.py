"""Abstract base class and registry for coordinate transforms.

Registering a custom transform
-------------------------------
Decorate your subclass with :func:`register_transform` and set the class
attribute ``transform_name`` to a unique string key::

    from panther_em.utils.transform_base import CoordinateTransform, register_transform


    @register_transform
    class MyPolarTransform(CoordinateTransform):
        transform_name = "my_polar"
        ...

The transform will then be available via :func:`reconstruct_transform` when
loading a :class:`~panther_em.decomposition.result.DecompositionResult` that
was saved with this transform embedded.

NOTE: Built-in transforms (``"offset_polar"``, ``"nonuniform_polar"``) are
registered automatically when the corresponding modules are imported.  Custom
transforms must be imported by the user *before* loading a result that
references them.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[CoordinateTransform]] = {}


def register_transform(
    cls: type[CoordinateTransform] | None = None,
    *,
    name: str | None = None,
) -> type[CoordinateTransform]:
    """Class decorator to register a :class:`CoordinateTransform` subclass.

    Parameters
    ----------
    cls : type[CoordinateTransform] | None
        The class being decorated.  When calling ``@register_transform``
        without arguments this is supplied automatically by Python's decorator
        machinery; when calling ``@register_transform(name="foo")`` it is None
        and the inner decorator is returned instead.
    name : str | None, optional
        Override the registry key.  Defaults to ``cls.transform_name``.

    Returns
    -------
    type[CoordinateTransform]
        The unmodified class (decorator is transparent).

    Examples
    --------
    Simple usage (uses ``transform_name`` class attribute as key)::

        @register_transform
        class OffsetPolarTransform(CoordinateTransform):
            transform_name = "offset_polar"
            ...

    With explicit name override::

        @register_transform(name="my_polar_v2")
        class MyPolarTransform(CoordinateTransform):
            transform_name = "my_polar_v2"
            ...
    """

    def decorator(klass: type[CoordinateTransform]) -> type[CoordinateTransform]:
        key = name if name is not None else klass.transform_name
        if key in _REGISTRY and _REGISTRY[key] is not klass:
            warnings.warn(
                f"A transform named '{key}' is already registered "
                f"({_REGISTRY[key].__qualname__}). Overwriting with "
                f"{klass.__qualname__}.",
                UserWarning,
                stacklevel=2,
            )
        _REGISTRY[key] = klass
        return klass

    # Called as @register_transform without parentheses
    if cls is not None:
        return decorator(cls)

    # Called as @register_transform(...) with keyword arguments
    return decorator  # type: ignore[return-value]


def get_transform_class(name: str) -> type[CoordinateTransform]:
    """Look up a registered transform class by name.

    Parameters
    ----------
    name : str
        Registry key (``transform_name`` of the desired class).

    Returns
    -------
    type[CoordinateTransform]
        The registered class.

    Raises
    ------
    KeyError
        If no transform with *name* has been registered. The error message
        lists all currently registered names.
    """
    if name not in _REGISTRY:
        known = ", ".join(f"'{k}'" for k in sorted(_REGISTRY))
        raise KeyError(
            f"No coordinate transform named '{name}' found in registry. "
            f"Known transforms: {known or '(none)'}. "
            "Make sure the module that registers your custom transform has "
            "been imported before loading the result."
        )
    return _REGISTRY[name]


def reconstruct_transform(
    params: dict[str, Any],
    device: str | Any = "numpy",
) -> CoordinateTransform:
    """Reconstruct a :class:`CoordinateTransform` from its serialized parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary produced by :meth:`CoordinateTransform.to_dict`.  Must
        contain a ``"transform_name"`` key.
    device : str | torch.device, optional
        Computational device to pass to
        :meth:`CoordinateTransform.from_dict`.  By default ``"numpy"`` (CPU).

    Returns
    -------
    CoordinateTransform
        A fully initialised transform instance.
    """
    if "transform_name" not in params:
        raise KeyError(
            "Serialized transform parameters are missing 'transform_name'. "
            "The dictionary must have been produced by CoordinateTransform.to_dict()."
        )
    cls = get_transform_class(params["transform_name"])
    return cls.from_dict(params, device=device)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CoordinateTransform(ABC):
    """Abstract base class for all coordinate transforms in Panther-EM.

    Required class attribute
    ------------------------
    transform_name : str
        A unique string identifier that is also used as the serialization key
        in :func:`register_transform` and :func:`reconstruct_transform`.

    Required abstract members
    -------------------------
    Concrete subclasses must implement:

    * :attr:`polar_shape`
    * :attr:`cartesian_shape`
    * :meth:`to_transform_space`
    * :meth:`to_cartesian`
    * :meth:`to_dict`
    * :meth:`from_dict`

    Optional capability flags
    -------------------------
    * ``supports_energy_preservation`` (default ``False``) — set to ``True``
      and implement :meth:`jacobian_correction` to enable energy-preserving
      transforms.
    * ``has_periodic_axis`` (default ``False``) — set to ``True`` when the
      first (angular) axis of the polar image wraps around periodically.
      Callers (e.g. :class:`~panther_em.utils.warp_transforms.OffsetPolarTransform`)
      use this flag to decide whether to apply circular padding at the
      0°/360° boundary.
    * ``periodic_axis`` (default ``None``) — index of the periodic axis (0 or
      1), relevant only when ``has_periodic_axis=True``.

    Device support
    --------------
    All concrete implementations **must** support both NumPy/CPU arrays and
    PyTorch CUDA tensors. The device is NOT serialised; it is supplied at
    reconstruction time via :meth:`from_dict`.
    """

    # --- Unique registry key (set as class attribute in subclasses) ---
    transform_name: str

    # --- Optional capability flags ---
    supports_energy_preservation: bool = False
    has_periodic_axis: bool = False
    periodic_axis: int | None = None

    # ------------------------------------------------------------------ #
    # Required shape properties
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the transformed image: ``(num_angle, num_radius)``."""

    @property
    @abstractmethod
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the Cartesian image: ``(height, width)``."""

    # ------------------------------------------------------------------ #
    # Required warp methods
    # ------------------------------------------------------------------ #

    @abstractmethod
    def to_transform_space(
        self,
        image: np.ndarray | torch.Tensor,
        preserve_energy: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Warp a Cartesian image to this transform's coordinate space.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2-D image (H, W) or batched 3-D array (B, H, W) in
            Cartesian coordinates.
        preserve_energy : bool, optional
            When ``True`` and :attr:`supports_energy_preservation` is
            ``True``, scale the output by the Jacobian of the coordinate
            change so that total energy is preserved.  If the transform
            does not support this, a ``NotImplementedError`` is raised.
        **kwargs
            Additional backend-specific parameters (e.g. interpolation
            ``order``, boundary ``mode``, ``cval``).

        Returns
        -------
        np.ndarray or torch.Tensor
            Warped image in the transformed space with shape
            :attr:`polar_shape` (or a batched equivalent), same type as
            input.
        """

    @abstractmethod
    def to_cartesian(
        self,
        image: np.ndarray | torch.Tensor,
        preserve_energy: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Warp an image from this transform's space back to Cartesian.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2-D image (num_angle, num_radius) or batched 3-D array in
            transformed coordinates.
        preserve_energy : bool, optional
            Inverse Jacobian correction — remove the area-scaling applied by
            the forward transform.
        **kwargs
            Additional backend-specific parameters.

        Returns
        -------
        np.ndarray or torch.Tensor
            Warped image in Cartesian space with shape
            :attr:`cartesian_shape` (or batched equivalent).
        """

    # ------------------------------------------------------------------ #
    # Required serialization
    # ------------------------------------------------------------------ #

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise the transform's geometric parameters to a plain dict.

        The returned dictionary must be JSON-serialisable (scalar values and
        plain Python lists for array-valued parameters). It must contain a
        ``"transform_name"`` key whose value equals
        ``self.__class__.transform_name``.

        Device information is **not** included; it is re-supplied at
        reconstruction time.

        Returns
        -------
        dict[str, Any]
            JSON-serializable parameter dictionary.
        """

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: str | torch.device = "numpy",
    ) -> CoordinateTransform:
        """Reconstruct an instance from its serialized parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary produced by :meth:`to_dict`.
        device : str | torch.device, optional
            Computational device for the reconstructed instance.
            By default ``"numpy"`` (CPU).

        Returns
        -------
        CoordinateTransform
            Fully initialized transform instance.
        """

    # ------------------------------------------------------------------ #
    # Optional energy preservation
    # ------------------------------------------------------------------ #

    def jacobian_correction(self) -> np.ndarray:
        """Per-radial-bin Jacobian (area-correction) factor.

        Override this method and set ``supports_energy_preservation = True``
        to enable energy-preserving forward/inverse transforms.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(num_radius,)`` containing the square-root
            of the Cartesian area represented by each polar pixel column.
            Dividing by this factor in the forward direction, and multiplying
            in the inverse direction, preserves total image energy.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented energy preservation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement energy-preserving "
            "transforms (supports_energy_preservation=False)."
        )

    def clear_cache(self) -> None:
        """Invalidate any cached coordinate grids or Jacobian arrays.

        Caching is an implementation detail; subclasses that cache computed
        coordinate mappings should override this method to release them. The
        default implementation is a no-op so that callers can always safely
        call ``transform.clear_cache()`` without checking the type.
        """
        return None
