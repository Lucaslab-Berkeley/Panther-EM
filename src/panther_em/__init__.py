"""Panther-EM: A Python package for eigendecomposition-accelerated 2DTM.

Pipelined AcceleratioN of Template matcHing via Eigendecomposition of Rotational
projections in cryo-EM (Panther-EM) is a Python package which:
 1. Implements a semi-analytical method for computing the SVD of particle projections
    for 2DTM using the inherent in-plane rotation symmetry of the projections in
    polar coordinates and the hypothesis we want to test against.
 2. Provides a framework for performing a 2DTM search using the SVD results of a large
    set of particle projections.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("panther-em")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Matthew Giammar"
__email__ = "mdgiammar@gmail.com"
