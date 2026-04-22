"""Module for reconstructing orientation-dependent cross-correlograms.

Note on tensor contraction
--------------------------
Reconstructing a cross-correlogram from SVD features is a large tensor contraction on a
hyper-spectral feature stack based on SVD components (U, S, Vh). The feature stack
F: (L, h, w) comes from the correlation of an image, I, against the right singular
vectors (features), Vh. That is, F = I * Vh. For the weights, W = (US), there may be
multiple dimensions of outer indices which are important for downstream tasks. Our
weights tensor has shape (... L), and the tensor contraction is W x F = C mapping shapes
(..., L) x (L, h, w) --> (..., h, w).

Running the semi-analytical SVD decomposition in polar coordinates means we get a
slightly more complicated indexing scheme thus warranting a separate module for tensor
contraction calculations. Each singular value is indexed by a angular frequency
component (analytical) and radial basis function (computed) to produce a feature; each
singular value is indexed by (k_idx, eig_idx) for the angular frequency and radial
eigenvector, respectively. We consider L = {(k_idx, eig_idx)_i} to be a set of indices.

The tensor contraction for reconstructing a cross-correlogram then becomes,
(..., (k_idx, eig_idx)) x ((k_idx, eig_idx), h, w) --> (..., h, w), where particular
singular value indices used are chosen among the much larger set. In practice, we would
flatten out the pairs into L and have (..., L) x (L, h, w) --> (..., h, w).

Computation can then
be broken into three stages:
1. Select which singular value indices L = {(k_idx, eig_idx)_i} to use in the
   computation.
2. Compute the hyper-spectral cross-correlation for the corresponding right-singular
   vectors for an input image.
3. Linearly recombine the stack by taking the inner product with the singular values
   and left-singular vectors across the spatial dimension.
Note for computational and memory efficiency reasons, steps 2 and 3 may be batched
across singular values (e.g. split selected indices into smaller chunks).
"""

from typing import Any, Iterator, Optional

import torch

from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.projection_reconstruction import ProjectionReconstructor


# ---------------------------------------------------------------------------
# Kernel construction and feature-stack computation
# ---------------------------------------------------------------------------


def build_cartesian_kernels(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
    **polar_to_cart_kwargs: Any,
) -> torch.Tensor:
    """Construct the Cartesian spatial kernels for each selected index pair.

    Each kernel ``V[k, eig]`` is built by
    :meth:`ProjectionReconstructor.construct_cartesian_feature`, which
    composes the analytic angular phase component with the radial eigenvector
    and inverse-maps the result to Cartesian coordinates via the stored
    :class:`OffsetPolarTransform`.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Provides the polar transform and stored SVD tensors.
    indices : torch.Tensor
        `(N, 2)` integer tensor of `[k_idx, eig_idx]` pairs as returned
        by :func:`select_indices`.
    **polar_to_cart_kwargs
        Forwarded to
        :meth:`ProjectionReconstructor.construct_cartesian_feature`.

    Returns
    -------
    torch.Tensor
        Complex64 tensor of shape `(N, kH, kW)` on
        `reconstructor.device`.
    """
    kernels = []
    for row in indices:
        k_idx = int(row[0].item())
        eig_idx = int(row[1].item())
        tmp = reconstructor.construct_cartesian_feature(
            k_idx=k_idx,
            eig_idx=eig_idx,
            return_torch=True,
            **polar_to_cart_kwargs,
        )
        kernels.append(tmp)

    return torch.stack(kernels, dim=0)  # (N, kH, kW) complex64


def compute_feature_stack(
    image: torch.Tensor,  # shape (B, H, W), real
    kernels: torch.Tensor,  # shape (L, kH, kW), complex
) -> torch.Tensor:  # shape (B, L, H - kH + 1, W - kW + 1), complex
    """Compute valid cross-correlation of image against kernels using FFT.

    For each image in the batch and each kernel, computes the cross-correlation.
    Outer-product operation: (B, H, W) x (L, kH, kW) --> (B, L, H - kh + 1, W - kw + 1)

    Parameters
    ----------
    image : torch.Tensor
        Input image batch of shape `(B, H, W)` (monochromatic).
    kernels : torch.Tensor
        Complex kernel bank of shape `(L, kH, kW)`.

    Returns
    -------
    torch.Tensor
        Complex feature stack of shape `(B, L, H - kH + 1, W - kW + 1)`.
    """
    # Ensure image has exactly 3 dims (B, H, W)
    if image.dim() != 3:
        raise ValueError(
            f"Image must have exactly 3 dimensions (B, H, W), got {image.dim()}"
        )
    if kernels.dim() != 3:
        raise ValueError(
            f"Kernels must have exactly 3 dimensions (L, kH, kW), got {kernels.dim()}"
        )

    B, H, W = image.shape
    L, h, w = kernels.shape

    image_fft = torch.fft.fft2(image)
    kernels_fft = torch.fft.fft2(kernels, s=(H, W))

    # Outer product: unsqueeze to (B, 1, H, W) and (1, L, H, W), then multiply
    corr_fft = image_fft.unsqueeze(1) * kernels_fft.conj().unsqueeze(0)
    corr = torch.fft.ifft2(corr_fft)
    corr = corr[..., : H - h + 1, : W - w + 1]

    return corr


# ---------------------------------------------------------------------------
# Weights construction and tensor contraction
# ---------------------------------------------------------------------------


def compute_weights(result: DecompositionResult, indices: torch.Tensor) -> torch.Tensor:
    """Extract weights `W[ff, orient, n] = U[ff, orient, k, eig] * S[k, eig]`.

    Parameters
    ----------
    result : DecompositionResult
        Source of singular values `S` (float32) and left singular vectors `U`
        (complex64).
    indices : torch.Tensor
        `(N, 2)` integer tensor of `[k_idx, eig_idx]` pairs. Note the device of
        this tensor controls the device of the returned weights.

    Returns
    -------
    torch.Tensor
        Complex64 tensor of shape `(num_fourier_filters, num_orientations, N)`.
    """
    device = indices.device

    U = torch.from_numpy(result.U)
    S = torch.from_numpy(result.S)
    S = torch.sqrt(S)

    # Send to devices and ensure correct dtype
    U = U.to(dtype=torch.complex64, device=device)  # (FF, O, k_max, eig_max)
    S = S.to(dtype=torch.complex64, device=device)  # (k_max, eig_max)

    k_idx = indices[:, 0]  # (L,)
    eig_idx = indices[:, 1]  # (L,)

    U_selected = U[:, :, k_idx, eig_idx]  # (FF, O, L)
    S_selected = S[k_idx, eig_idx]  # (L,)

    return U_selected * S_selected


def contract_features(weights: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Contract weights against the feature stack to form the correlogram.

    Parameters
    ----------
    weights : torch.Tensor
        Shape `(FF, O, L)` complex weights for the `L` selected
        components across all Fourier-filter / orientation pairs.
    features : torch.Tensor
        Shape `(L, H, W)` or `(B, L, H, W)` complex feature stack
        from :func:`compute_feature_stack`.

    Returns
    -------
    torch.Tensor
        Complex correlogram of shape `(FF, O, H, W)` or `(B, FF, O, H, W)`.
    """
    if features.dim() == 3:
        return torch.einsum("fol, lhw -> fohw", weights, features)
    return torch.einsum("fol, blhw -> bfohw", weights, features)


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------


def _chunk_indices(indices: torch.Tensor, chunk_size: int) -> Iterator[torch.Tensor]:
    """Yield successive row-slices of `indices` of at most `chunk_size` rows.

    Parameters
    ----------
    indices : torch.Tensor
        `(N, 2)` index tensor.
    chunk_size : int
        Maximum rows per chunk.

    Yields
    ------
    torch.Tensor
        Sub-tensor of `indices` with up to `chunk_size` rows.
    """
    for start in range(0, len(indices), chunk_size):
        yield indices[start : start + chunk_size]


# ---------------------------------------------------------------------------
# Primary public entry point
# ---------------------------------------------------------------------------


def compute_correlogram(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    *,
    k_indices: Optional[torch.Tensor] = None,
    eig_indices: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **polar_to_cart_kwargs: Any,
) -> torch.Tensor:
    """Compute the orientation-dependent cross-correlogram end-to-end.

    Orchestrates all three stages — index selection, feature-stack
    computation, and weighted recombination — with optional chunking over the
    component axis to bound peak memory.

    Parameters
    ----------
    image : torch.Tensor
        Pre-processed input image of shape `(B, C, H, W)` or `(C, H, W)`.
        In a typical cryo-EM workflow this is a (patch of a) micrograph.
    reconstructor : ProjectionReconstructor
        Holds the :class:`DecompositionResult`, polar transform, and stored
        SVD tensors needed to build kernels and weights.
    k_indices : torch.Tensor, optional
        1-D integer tensor of angular-frequency indices to include.
    eig_indices : torch.Tensor, optional
        1-D integer tensor of radial-eigenvalue indices to include.
    top_k : int, optional
        If given, select only the `top_k` components ranked by `|S|`,
        ignoring `k_indices` / `eig_indices`.
    chunk_size : int, optional
        Maximum number of components to process at once. If `None`, processes
        all components in a single pass. Use this to limit peak memory when
        working with many selected components (e.g. 10,000+).
    **polar_to_cart_kwargs
        Forwarded to
        :meth:`ProjectionReconstructor.construct_cartesian_feature` for each
        kernel (e.g. `order`, `mode`, `preserve_energy`).

    Returns
    -------
    torch.Tensor
        Complex correlogram on `reconstructor.device` of shape
        `(B, FF, O, H, W)` or `(FF, O, H, W)`, where
        `FF = num_fourier_filters`, `O = num_orientations`.

    Raises
    ------
    ValueError
        If no index pairs are selected.

    # Examples
    # --------
    # >>> reconstructor = ProjectionReconstructor(result, image_shape=(256, 256))
    # >>> C = compute_correlogram(micrograph_patch, reconstructor, top_k=32, chunk_size=8)
    # >>> C.shape  # e.g. (1, num_fourier_filters, num_orientations, 256, 256)
    """
    result = reconstructor.result
    device = reconstructor.device

    kH, kW = reconstructor.image_shape
    H = image.shape[-2]
    W = image.shape[-1]
    out_H = H - kH + 1
    out_W = W - kW + 1

    _has_batch = image.dim() == 4
    B = image.shape[0] if _has_batch else None
    FF, O = result.num_fourier_filters, result.num_orientations
    out_shape = (B, FF, O, out_H, out_W) if _has_batch else (FF, O, out_H, out_W)

    accumulator = torch.zeros(out_shape, dtype=torch.complex64, device=device)
    image = image.to(device)

    # --- Stage 1: select (k_idx, eig_idx) pairs ----------------------------
    if k_indices is None and eig_indices is None:
        if top_k is None:
            raise ValueError(
                "Must provide either 'top_k' or both 'k_indices' and 'eig_indices'."
            )
        top_k_indices = result.get_top_k(top_k=top_k)
        top_k_indices = torch.from_numpy(top_k_indices).to(device)
    else:
        if k_indices is None or eig_indices is None:
            raise ValueError("Must provide both 'k_indices' and 'eig_indices'.")
        k_indices = k_indices.to(device)
        eig_indices = eig_indices.to(device)
        ki, ei = torch.meshgrid(k_indices, eig_indices, indexing="ij")
        top_k_indices = torch.stack([ki.flatten(), ei.flatten()], dim=1)

    N = len(top_k_indices)
    if N == 0:
        raise ValueError(
            "No (k_idx, eig_idx) pairs were selected; cannot compute correlogram."
        )

    # --- Stages 2 & 3: build kernels, correlate, contract (optionally chunked)
    effective_chunk = chunk_size if chunk_size is not None else N

    for chunk_indices in _chunk_indices(top_k_indices, effective_chunk):
        kernels = build_cartesian_kernels(
            reconstructor, chunk_indices, **polar_to_cart_kwargs
        )
        F_chunk = compute_feature_stack(image, kernels)
        W_chunk = compute_weights(result, chunk_indices)

        accumulator = accumulator + contract_features(W_chunk, F_chunk)

    return accumulator


# ---------------------------------------------------------------------------
# Memory planning utility
# ---------------------------------------------------------------------------


def estimate_memory_bytes(
    result: DecompositionResult,
    image_shape: tuple[int, ...],
    *,
    n_selected: Optional[int] = None,
    chunk_size: Optional[int] = None,
    dtype: torch.dtype = torch.complex64,
) -> dict[str, int]:
    """Estimate peak memory for a :func:`compute_correlogram` call.

    Use this before setting `chunk_size` to avoid GPU OOM errors.  All
    figures are approximate; actual usage will be higher due to PyTorch
    allocator overhead and intermediate autograd buffers.

    Parameters
    ----------
    result : DecompositionResult
        Provides shape metadata only; no tensors are allocated.
    image_shape : tuple[int, ...]
        Shape of the input image, e.g. `(B, 1, H, W)` or `(1, H, W)`.
    n_selected : int, optional
        Number of `(k, eig)` pairs to be selected.  Defaults to the full
        `k_max x num_radial_components` product.
    chunk_size : int, optional
        Chunk size to evaluate; defaults to `n_selected` (single pass).
    dtype : torch.dtype
        Element type for computation.  Defaults to `complex64` (8 bytes).

    Returns
    -------
    dict[str, int]
        Byte estimates keyed by:
        `"kernel_chunk"`, `"feature_chunk"`, `"weight_chunk"`,
        `"accumulator"`, `"total_peak"`.
    """
    # Bytes per element, accounting for complex types.
    if dtype.is_complex:
        bpe = torch.finfo(torch.float32).bits // 8 * 2  # 8 for complex64
    else:
        bpe = torch.finfo(dtype).bits // 8

    N = n_selected or (result.k_max * result.num_radial_components)
    chunk = chunk_size or N

    # Kernel spatial footprint matches the particle box used by the transform.
    kH = kW = result.num_radial_components
    H, W = image_shape[-2], image_shape[-1]
    B = image_shape[0] if len(image_shape) == 4 else 1
    FF = result.num_fourier_filters
    O = result.num_orientations

    kernel_bytes = chunk * kH * kW * bpe  # (chunk, kH, kW)
    feature_bytes = B * chunk * H * W * bpe  # (B, chunk, H, W)
    weight_bytes = FF * O * chunk * bpe  # (FF, O, chunk)
    acc_bytes = B * FF * O * H * W * bpe  # (B, FF, O, H, W)
    total = kernel_bytes + feature_bytes + weight_bytes + acc_bytes

    return {
        "kernel_chunk": kernel_bytes,
        "feature_chunk": feature_bytes,
        "weight_chunk": weight_bytes,
        "accumulator": acc_bytes,
        "total_peak": total,
    }
