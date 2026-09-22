"""SEAL: bounded refinement-step conditioning for the public ICASSP model.

The RADR router forms ``normalize(acoustic_evidence + step_embedding[s])``.
M1-StepBound tests the hypothesis that refinement identity should act as a
tie-breaker without overriding a confident acoustic route. It applies an
absolute L2 cap to the existing router step-embedding rows; the cap is selected
The public release uses the paper operating point ``max_norm=0.15``. This is a
fixed model setting, not a claim that 0.15 is globally optimal.

This module guarantees a direction-preserving, zero-parameter intervention:
rows below the cap are unchanged, rows above it are norm-clipped, and parameter
keys remain compatible with the unbounded graph. These implementation facts do
not establish separation gain or acoustic specialization. The matched Full /
Unbounded / No-step experiments, clipping history, same-query counterfactual
route flips, and step-conditioned routing MI provide those falsifiable tests.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .gtcrn_ss_noncausal_M1_core import GTCRN_SS_NonCausal_M1_Core
except ImportError:  # pragma: no cover - flat script execution
    from gtcrn_ss_noncausal_M1_core import GTCRN_SS_NonCausal_M1_Core


_M1_STEPBOUND_ARCHITECTURE_VERSIONS = {"m1_stepbound_v1"}


class NormClippedStepEmbedding(nn.Module):
    """``nn.Embedding`` whose rows are clipped to a maximum L2 norm.

    The source parameter is **adopted by reference**, never copied or re-drawn,
    so a model built with this wrapper has exactly M1's initialization and
    exactly M1's parameter count.

    Clipping rather than squashing is deliberate. Below the threshold the map is
    the identity, so the mechanism is inert until the embedding actually tries
    to grow past its budget; above it, the direction is preserved and only the
    magnitude is capped. A ``tanh`` would distort every row including small ones
    and can saturate its gradient once driven hard.
    """

    def __init__(self, source: nn.Embedding, max_norm: float):
        super().__init__()
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive; use the unbounded arm instead")
        self.weight = source.weight
        self.max_norm = float(max_norm)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        embedded = F.embedding(index, self.weight)
        norm = embedded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # clamp(max=1) makes this the identity while the row is within budget,
        # which is what keeps a zero-initialized M1 bit-exact at construction.
        return embedded * (self.max_norm / norm).clamp(max=1.0)


class GTCRN_SS_NonCausal_M1_StepBound(GTCRN_SS_NonCausal_M1_Core):
    """SEAL implementation with a bounded router step-embedding cue."""

    def __init__(
        self,
        *args,
        step_embedding_max_norm: float = 0.15,
        **kwargs,
    ):
        requested_version = (
            kwargs.pop("architecture_version", None) or "m1_stepbound_v1"
        )
        if requested_version not in _M1_STEPBOUND_ARCHITECTURE_VERSIONS:
            raise ValueError(
                "M1-StepBound architecture_version must be one of "
                f"{sorted(_M1_STEPBOUND_ARCHITECTURE_VERSIONS)}"
            )
        kwargs["architecture_version"] = "m1_core_radr_latent_additive_v1"
        super().__init__(*args, **kwargs)
        self.architecture_version = requested_version

        self.step_embedding_max_norm = float(step_embedding_max_norm)
        self.step_router_names: List[str] = [
            name
            for name, module in self.named_modules()
            if name.endswith("temporal_readout.router")
            and isinstance(getattr(module, "step_embedding", None), nn.Embedding)
        ]
        self.bounded_routers: List[str] = []
        if self.step_embedding_max_norm > 0.0:
            self._bound_step_embeddings()

    def _bound_step_embeddings(self) -> None:
        # Only the MoE router's step embedding is bounded. `separator.fusion`
        # owns a separate one that feeds SAFR, not the expert choice; bounding
        # it would put two mechanisms in one arm.
        for name, module in self.named_modules():
            if not name.endswith("temporal_readout.router"):
                continue
            source = getattr(module, "step_embedding", None)
            if not isinstance(source, nn.Embedding):
                raise RuntimeError(
                    f"{name}.step_embedding is {type(source).__name__}, not "
                    "nn.Embedding; the router layout is not what this ablation "
                    "assumes"
                )
            module.step_embedding = NormClippedStepEmbedding(
                source, self.step_embedding_max_norm
            )
            self.bounded_routers.append(name)

        if not self.bounded_routers:
            raise RuntimeError(
                "no temporal_readout.router found; this ablation requires the "
                "MoE router to be enabled"
            )

    def step_bound_report(self) -> Dict[str, object]:
        norms = {}
        all_raw = []
        all_effective = []
        all_clipped = []
        for name in self.step_router_names:
            module = self.get_submodule(name)
            with torch.no_grad():
                rows = module.step_embedding.weight
                effective_rows = module.step_embedding(
                    torch.arange(rows.shape[0], device=rows.device)
                )
                raw_norm = rows.norm(dim=-1)
                effective_norm = effective_rows.norm(dim=-1)
                clipped_mask = (
                    raw_norm > self.step_embedding_max_norm + 1e-7
                    if self.step_embedding_max_norm > 0.0
                    else torch.zeros_like(raw_norm, dtype=torch.bool)
                )
                norms[name] = {
                    "raw": [float(v) for v in raw_norm],
                    "effective": [float(v) for v in effective_norm],
                    "clipped": [bool(v) for v in clipped_mask],
                    "clip_active_fraction": float(clipped_mask.float().mean()),
                }
                all_raw.append(raw_norm)
                all_effective.append(effective_norm)
                all_clipped.append(clipped_mask)

        if all_raw:
            raw = torch.cat(all_raw)
            effective = torch.cat(all_effective)
            clipped = torch.cat(all_clipped)
            rows_total = int(raw.numel())
            rows_clipped = int(clipped.sum())
            raw_mean = float(raw.mean())
            raw_max = float(raw.max())
            effective_mean = float(effective.mean())
            effective_max = float(effective.max())
        else:
            rows_total = rows_clipped = 0
            raw_mean = raw_max = effective_mean = effective_max = 0.0
        return {
            "max_norm": self.step_embedding_max_norm,
            "bounded_routers": list(self.bounded_routers),
            "step_router_names": list(self.step_router_names),
            "rows_total": rows_total,
            "rows_clipped": rows_clipped,
            "clip_active_fraction": (
                float(rows_clipped) / rows_total if rows_total else 0.0
            ),
            "bound_is_active": bool(rows_clipped),
            "raw_norm_mean": raw_mean,
            "raw_norm_max": raw_max,
            "effective_norm_mean": effective_mean,
            "effective_norm_max": effective_max,
            "step_embedding_norms": norms,
        }

    def _step_bound_tensor_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Return logging-safe tensors without Python conversion or GPU sync."""

        raw_rows = []
        effective_rows = []
        clipped_rows = []
        for name in self.step_router_names:
            module = self.get_submodule(name)
            rows = module.step_embedding.weight.detach()
            effective = module.step_embedding(
                torch.arange(rows.shape[0], device=rows.device)
            ).detach()
            raw_norm = rows.norm(dim=-1).float()
            effective_norm = effective.norm(dim=-1).float()
            clipped = (
                raw_norm > self.step_embedding_max_norm + 1e-7
                if self.step_embedding_max_norm > 0.0
                else torch.zeros_like(raw_norm, dtype=torch.bool)
            )
            raw_rows.append(raw_norm)
            effective_rows.append(effective_norm)
            clipped_rows.append(clipped)

        if raw_rows:
            raw = torch.cat(raw_rows)
            effective = torch.cat(effective_rows)
            clipped = torch.cat(clipped_rows).float()
            return {
                "raw_norm_per_step": raw,
                "effective_norm_per_step": effective,
                "clipped_per_step": clipped,
                "clip_active_fraction": clipped.mean(),
                "bound_is_active": clipped.max(),
                "raw_norm_mean": raw.mean(),
                "raw_norm_max": raw.max(),
                "effective_norm_mean": effective.mean(),
                "effective_norm_max": effective.max(),
            }
        zero = next(self.parameters()).detach().float().new_zeros(())
        empty = zero.new_zeros(0)
        return {
            "raw_norm_per_step": empty,
            "effective_norm_per_step": empty,
            "clipped_per_step": empty,
            "clip_active_fraction": zero,
            "bound_is_active": zero,
            "raw_norm_mean": zero,
            "raw_norm_max": zero,
            "effective_norm_mean": zero,
            "effective_norm_max": zero,
        }

    def get_m1_diagnostics(self) -> Dict[str, object]:
        diagnostics = super().get_m1_diagnostics()
        # Nested numeric fields are flattened by the trainer and written every
        # epoch.  This proves whether StepBound actually bound the learned
        # embedding instead of relying on the configured threshold alone.
        diagnostics["step_bound"] = self._step_bound_tensor_diagnostics()
        return diagnostics
