"""Coordinate transform system for Panther-EM.

Provides the abstract base class, registry, and built-in polar coordinate
transform implementations.
"""

# Import built-in transforms to trigger @register_transform
from .nonuniform_polar import NonUniformPolarTransform
from .offset_polar import OffsetPolarTransform
from .transform_base import (
    CoordinateTransform,
    GridTransform,
    get_transform,
    get_transform_class,
    reconstruct_transform,
    register_transform,
)

__all__ = [
    "CoordinateTransform",
    "GridTransform",
    "NonUniformPolarTransform",
    "OffsetPolarTransform",
    "get_transform",
    "get_transform_class",
    "reconstruct_transform",
    "register_transform",
]
