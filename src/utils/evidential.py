"""Utilities for evidential (Dirichlet) predictions."""

from __future__ import annotations

import torch


def aggregate_dirichlet_evidence(
    alpha: torch.Tensor,
    dim: int = -2,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate independent Dirichlet evidence along ``dim``.

    Each prediction is parameterised as ``alpha = evidence + 1``.  Therefore
    only evidence is additive; the unit prior must be added exactly once:

        alpha_agg = 1 + sum(alpha_t - 1)

    Summing ``alpha_t`` directly would count the unit prior once per item and
    make confidence and uncertainty depend spuriously on sequence length.
    """
    if not isinstance(alpha, torch.Tensor):
        raise TypeError("alpha must be a torch.Tensor")
    if not alpha.is_floating_point():
        raise TypeError("alpha must have a floating-point dtype")
    if alpha.ndim == 0:
        raise ValueError("alpha must have at least one dimension")
    if alpha.shape[dim] == 0:
        raise ValueError("cannot aggregate an empty evidence dimension")
    evidence = alpha - 1.0
    if mask is not None:
        weight = mask.to(device=alpha.device, dtype=alpha.dtype)
        # The class axis is conventionally last; a (B,T) mask therefore
        # becomes (B,T,1). More general broadcastable masks are also accepted.
        while weight.ndim < evidence.ndim:
            weight = weight.unsqueeze(-1)
        try:
            evidence = evidence * weight
        except RuntimeError as exc:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is not broadcastable to "
                f"alpha shape {tuple(alpha.shape)}"
            ) from exc
    return 1.0 + evidence.sum(dim=dim)
