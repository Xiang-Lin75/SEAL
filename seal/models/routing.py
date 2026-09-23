"""Temporal readout experts and routers.

Every refinement step routes each time-frequency token to one stateless
residual expert (Top-1). The refinement-aware router adds a step cue whose
norm is capped by :class:`NormClippedStepEmbedding`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(value: float) -> float:
    """Stable inverse sigmoid for bounded scalar initialization."""

    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class DirectionGroupReadoutExpert(nn.Module):
    """Stateless gated readout over the two grouped bidirectional GRU streams.

    Each grouped GRU contributes one ``C``-wide tensor containing its forward
    and reverse directions.  The expert never owns recurrent state or recurrent
    matrices; it only maps the shared Temporal GRNN trajectory to a residual
    readout.  This keeps memory semantics in the shared GRNN while giving the
    sparse experts a precise conditional-computation role.
    """

    def __init__(self, channels: int, expert_width: int, dropout: float = 0.0):
        super().__init__()
        if channels < 1 or expert_width < 1:
            raise ValueError("channels and expert_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("moe_dropout must be in [0, 1)")
        self.channels = int(channels)
        self.expert_width = int(expert_width)
        self.value = nn.ModuleList(
            [nn.Linear(channels, expert_width) for _ in range(2)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(channels, expert_width) for _ in range(2)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * expert_width, channels)

        # Near-baseline warm start: experts are distinct, but their initial
        # correction is small enough not to overwrite the shared inter_fc path.
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        group_one: torch.Tensor,
        group_two: torch.Tensor,
    ) -> torch.Tensor:
        if group_one.shape != group_two.shape or group_one.shape[-1] != self.channels:
            raise ValueError("TRR expert inputs must be two matching (..., C) tensors")
        values = []
        for group_index, group in enumerate((group_one, group_two)):
            value = F.silu(self.value[group_index](group))
            gate = torch.sigmoid(self.gate[group_index](group))
            values.append(value * gate)
        fused = self.dropout(torch.cat(values, dim=-1))
        return self.output(fused)


class SharedHierarchicalAcousticRouter(nn.Module):
    """SHAR: bounded multi-evidence cosine routing for Temporal GRNN tokens."""

    BRANCH_NAMES = ("local", "anchor", "progress", "trajectory", "global")

    def __init__(
        self,
        channels: int,
        num_experts: int,
        router_dim: int = 24,
        temperature_init: float = 0.7,
        temperature_min: float = 0.1,
        temperature_max: float = 2.0,
    ):
        super().__init__()
        if channels < 1 or num_experts < 1 or router_dim < 1:
            raise ValueError("channels, num_experts and router_dim must be positive")
        if router_dim < num_experts:
            raise ValueError("router_dim must be >= num_experts for orthogonal prototypes")
        if not 0.0 < temperature_min < temperature_init < temperature_max:
            raise ValueError(
                "router temperatures must satisfy 0 < min < init < max"
            )
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.router_dim = int(router_dim)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)

        self.local_norm = nn.LayerNorm(2 * channels, eps=1e-6)
        self.local_projection = nn.Linear(2 * channels, router_dim, bias=False)
        self.anchor_norm = nn.LayerNorm(channels, eps=1e-6)
        self.anchor_projection = nn.Linear(channels, router_dim, bias=False)
        self.progress_norm = nn.LayerNorm(channels, eps=1e-6)
        self.progress_projection = nn.Linear(channels, router_dim, bias=False)
        self.trajectory_norm = nn.LayerNorm(3 * channels, eps=1e-6)
        self.trajectory_projection = nn.Linear(3 * channels, router_dim, bias=False)
        self.global_norm = nn.LayerNorm(2 * channels, eps=1e-6)
        self.global_projection = nn.Linear(2 * channels, router_dim, bias=False)

        initial_weights = torch.tensor([0.55, 0.07, 0.20, 0.15, 0.03])
        self.evidence_logits = nn.Parameter(initial_weights.log())
        self.prototypes = nn.Parameter(torch.empty(num_experts, router_dim))
        nn.init.orthogonal_(self.prototypes)

        ratio = (temperature_init - temperature_min) / (
            temperature_max - temperature_min
        )
        self.raw_temperature = nn.Parameter(torch.tensor(_logit(ratio)))

    @property
    def temperature(self) -> torch.Tensor:
        span = self.temperature_max - self.temperature_min
        return self.temperature_min + span * torch.sigmoid(self.raw_temperature)

    @staticmethod
    def _mask(
        batch: int,
        frames: int,
        device: torch.device,
        valid_time_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if valid_time_mask is None:
            return torch.ones(batch, frames, device=device, dtype=torch.bool)
        if valid_time_mask.shape != (batch, frames):
            raise ValueError(
                f"valid_time_mask must be {(batch, frames)}, got {tuple(valid_time_mask.shape)}"
            )
        return valid_time_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _normalize(evidence: torch.Tensor) -> torch.Tensor:
        return F.normalize(evidence, p=2.0, dim=-1, eps=1e-6)

    @staticmethod
    def _project_fp32(
        values: torch.Tensor,
        norm: nn.LayerNorm,
        projection: nn.Linear,
    ) -> torch.Tensor:
        device_type = values.device.type
        if device_type in {"cpu", "cuda", "xpu", "mps"}:
            with torch.autocast(device_type=device_type, enabled=False):
                return projection(norm(values.float()))
        return projection(norm(values.float()))

    @staticmethod
    def _masked_mean_std(
        values: torch.Tensor,
        mask_bft1: torch.Tensor,
        dimensions: Tuple[int, ...],
        denominator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values_fp32 = values.float()
        mask = mask_bft1.float()
        mean = (values_fp32 * mask).sum(dim=dimensions) / denominator
        mean_for_broadcast = mean
        for dimension in sorted(dimensions):
            mean_for_broadcast = mean_for_broadcast.unsqueeze(dimension)
        variance = (
            (values_fp32 - mean_for_broadcast).square() * mask
        ).sum(dim=dimensions) / denominator
        return mean, variance.clamp_min(0.0).add(1e-8).sqrt()

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if anchor_bftc.ndim != 4 or anchor_bftc.shape[-1] != self.channels:
            raise ValueError("anchor_bftc must have shape (B,F,T,C)")
        batch, bands, frames, _ = anchor_bftc.shape
        valid = self._mask(batch, frames, anchor_bftc.device, valid_time_mask)
        mask_bft1 = valid[:, None, :, None]
        masked_anchor = anchor_bftc * mask_bft1.to(anchor_bftc.dtype)

        anchor_evidence = self._normalize(
            self._project_fp32(masked_anchor, self.anchor_norm, self.anchor_projection)
        )
        denominator = (
            valid.sum(dim=1).float().mul(float(bands)).clamp_min(1.0)[:, None]
        )
        global_mean, global_std = self._masked_mean_std(
            masked_anchor,
            mask_bft1,
            dimensions=(1, 2),
            denominator=denominator,
        )
        global_statistics = torch.cat([global_mean, global_std], dim=-1)
        global_evidence = self._normalize(
            self._project_fp32(
                global_statistics,
                self.global_norm,
                self.global_projection,
            )
        )[:, None, None, :]
        return {
            "anchor_bftc": masked_anchor,
            "anchor_evidence": anchor_evidence,
            "global_evidence": global_evidence,
            "valid_time_mask": valid,
        }

    def _trajectory_statistics(
        self,
        shared_readout: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, bands, frames, _ = shared_readout.shape
        mask_bft1 = valid[:, None, :, None]
        denominator = valid.sum(dim=1).float().clamp_min(1.0)[:, None, None]
        mean, std = self._masked_mean_std(
            shared_readout,
            mask_bft1,
            dimensions=(2,),
            denominator=denominator,
        )
        if frames == 1:
            mean_absolute_delta = torch.zeros_like(mean)
        else:
            pair_mask = valid[:, 1:] & valid[:, :-1]
            pair_mask_bft1 = pair_mask[:, None, :, None].float()
            pair_denominator = (
                pair_mask.sum(dim=1).float().clamp_min(1.0)[:, None, None]
            )
            delta = (
                shared_readout[:, :, 1:].float()
                - shared_readout[:, :, :-1].float()
            ).abs()
            mean_absolute_delta = (delta * pair_mask_bft1).sum(dim=2) / pair_denominator
        return torch.cat([mean, std, mean_absolute_delta], dim=-1)

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        prepared_context: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if raw_states.ndim != 4 or raw_states.shape[-1] != 2 * self.channels:
            raise ValueError("raw_states must have shape (B,F,T,2C)")
        if shared_readout.shape != raw_states.shape[:-1] + (self.channels,):
            raise ValueError("shared_readout must align with raw_states as (B,F,T,C)")
        anchor = prepared_context["anchor_bftc"]
        valid = prepared_context["valid_time_mask"]
        if anchor.shape != shared_readout.shape:
            raise ValueError("prepared anchor and shared_readout shapes differ")

        local = self._normalize(
            self._project_fp32(raw_states, self.local_norm, self.local_projection)
        )
        progress = self._normalize(
            self._project_fp32(
                shared_readout - anchor,
                self.progress_norm,
                self.progress_projection,
            )
        )
        trajectory_statistics = self._trajectory_statistics(shared_readout, valid)
        trajectory = self._normalize(
            self._project_fp32(
                trajectory_statistics,
                self.trajectory_norm,
                self.trajectory_projection,
            )
        )[:, :, None, :].expand_as(local)
        anchor_evidence = prepared_context["anchor_evidence"]
        global_evidence = prepared_context["global_evidence"].expand_as(local)

        branches = torch.stack(
            [local, anchor_evidence, progress, trajectory, global_evidence],
            dim=-2,
        )
        evidence_weights = torch.softmax(self.evidence_logits.float(), dim=0)
        query = self._normalize(
            (branches * evidence_weights.view(1, 1, 1, -1, 1)).sum(dim=-2)
        )

        batch, bands, frames, _ = query.shape
        token_valid = valid[:, None, :].expand(batch, bands, frames).reshape(-1)
        valid_indices = token_valid.nonzero(as_tuple=False).squeeze(1)
        valid_query = query.reshape(-1, self.router_dim).index_select(0, valid_indices)
        prototypes = self._normalize(self.prototypes.float())
        cosine_logits = valid_query @ prototypes.transpose(0, 1)
        scaled_logits = cosine_logits / self.temperature.float()
        probabilities = torch.softmax(scaled_logits, dim=-1)

        prototype_cosine = prototypes @ prototypes.transpose(0, 1)
        if self.num_experts > 1:
            off_diagonal = ~torch.eye(
                self.num_experts,
                device=prototype_cosine.device,
                dtype=torch.bool,
            )
            prototype_max_cosine = prototype_cosine.masked_select(off_diagonal).abs().max()
            prototype_orthogonality_loss = (
                prototype_cosine.masked_select(off_diagonal).square().mean()
            )
        else:
            prototype_max_cosine = prototype_cosine.new_zeros(())
            prototype_orthogonality_loss = prototype_cosine.new_zeros(())
        mean_query_norm = (
            valid_query.norm(dim=-1).mean()
            if valid_query.numel() > 0
            else query.new_zeros(())
        )
        router_aux = {
            "evidence_weights": evidence_weights,
            "router_temperature": self.temperature.float(),
            "prototype_max_cosine": prototype_max_cosine,
            "prototype_orthogonality_loss": prototype_orthogonality_loss,
            "mean_query_norm": mean_query_norm,
        }
        return valid_indices, scaled_logits, probabilities, router_aux


class SparseTop1TemporalReadoutMoE(nn.Module):
    """Sparse top-1 direction/group-aware readout on raw Temporal GRNN states."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        expert_width: int = 96,
        dropout: float = 0.0,
        top_k: int = 1,
        router_dim: int = 24,
        router_temperature_init: float = 0.7,
        router_temperature_min: float = 0.1,
        router_temperature_max: float = 2.0,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k != 1:
            raise ValueError("this readout implements sparse Top-1 routing; moe_top_k must be 1")
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError("moe_residual_scale_init must be in (0, 1)")
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.group_norms = nn.ModuleList(
            [nn.LayerNorm(channels, eps=1e-6) for _ in range(2)]
        )
        self.router = SharedHierarchicalAcousticRouter(
            channels=channels,
            num_experts=num_experts,
            router_dim=router_dim,
            temperature_init=router_temperature_init,
            temperature_min=router_temperature_min,
            temperature_max=router_temperature_max,
        )
        self.experts = nn.ModuleList(
            [
                DirectionGroupReadoutExpert(channels, expert_width, dropout)
                for _ in range(num_experts)
            ]
        )
        self.raw_residual_scale = nn.Parameter(torch.tensor(_logit(residual_scale_init)))

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_residual_scale)

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.router.prepare_anchor_context(anchor_bftc, valid_time_mask)

    def _ddp_graph_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        anchor = reference.new_zeros(())
        for parameter in self.parameters():
            if parameter.numel() > 0:
                anchor = anchor + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
        return anchor

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        prepared_context: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        valid_indices, router_logits, probabilities, router_aux = self.router(
            raw_states,
            shared_readout,
            prepared_context,
        )
        batch, bands, frames, _ = raw_states.shape
        output = shared_readout.new_zeros(batch * bands * frames, self.channels)
        zero = raw_states.float().new_zeros(())
        empty_vector = raw_states.float().new_zeros(self.num_experts)
        if valid_indices.numel() == 0:
            output = output + self._ddp_graph_anchor(output)
            aux = {
                "balance_loss": zero,
                "router_z_loss": zero,
                "router_entropy": zero,
                "mean_top1_probability": zero,
                "prob_fraction": empty_vector,
                "hard_fraction": empty_vector,
                "expert_counts": empty_vector,
                "probability_sums": empty_vector,
                "num_valid_tokens": zero,
                "adjacent_route_flip": zero,
                "routed_residual_rms": zero,
                **router_aux,
            }
            return output.reshape(batch, bands, frames, self.channels), aux

        flat_raw = raw_states.reshape(-1, 2 * self.channels).index_select(0, valid_indices)
        device_type = flat_raw.device.type
        if device_type in {"cpu", "cuda", "xpu", "mps"}:
            with torch.autocast(device_type=device_type, enabled=False):
                group_one = self.group_norms[0](flat_raw[:, : self.channels].float())
                group_two = self.group_norms[1](flat_raw[:, self.channels :].float())
        else:
            group_one = self.group_norms[0](flat_raw[:, : self.channels].float())
            group_two = self.group_norms[1](flat_raw[:, self.channels :].float())

        top1_probability, top1_expert = probabilities.max(dim=-1)
        for expert_index, expert in enumerate(self.experts):
            local_indices = (top1_expert == expert_index).nonzero(as_tuple=False).squeeze(1)
            if local_indices.numel() == 0:
                continue
            expert_output = expert(
                group_one.index_select(0, local_indices),
                group_two.index_select(0, local_indices),
            )
            gate = top1_probability.index_select(0, local_indices).to(expert_output.dtype)
            expert_output = (expert_output * gate.unsqueeze(-1)).to(output.dtype)
            global_indices = valid_indices.index_select(0, local_indices)
            output = output.index_copy(0, global_indices, expert_output)

        output = output * self.residual_scale.to(output.dtype)
        output = output + self._ddp_graph_anchor(output)

        hard_assignment = F.one_hot(top1_expert, num_classes=self.num_experts).float()
        expert_counts = hard_assignment.sum(dim=0)
        probability_sums = probabilities.sum(dim=0)
        hard_fraction = expert_counts / float(valid_indices.numel())
        prob_fraction = probability_sums / float(valid_indices.numel())
        balance_loss = self.num_experts * torch.sum(hard_fraction * prob_fraction)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        entropy = -(probabilities.clamp_min(1e-9) * probabilities.clamp_min(1e-9).log())
        entropy = entropy.sum(dim=-1).mean()

        full_routes = torch.full(
            (batch * bands * frames,),
            -1,
            device=top1_expert.device,
            dtype=top1_expert.dtype,
        ).index_copy(0, valid_indices, top1_expert)
        full_routes = full_routes.reshape(batch, bands, frames)
        valid = prepared_context["valid_time_mask"]
        if frames > 1:
            pair_mask = (valid[:, 1:] & valid[:, :-1])[:, None, :].expand(
                batch, bands, frames - 1
            )
            route_changed = full_routes[:, :, 1:] != full_routes[:, :, :-1]
            adjacent_route_flip = (
                route_changed.masked_select(pair_mask).float().mean()
                if bool(pair_mask.any())
                else zero
            )
        else:
            adjacent_route_flip = zero
        routed_values = output.index_select(0, valid_indices).float()
        routed_residual_rms = routed_values.square().mean().sqrt()

        aux = {
            "balance_loss": balance_loss,
            "router_z_loss": router_z_loss,
            "router_entropy": entropy,
            "mean_top1_probability": top1_probability.mean(),
            "prob_fraction": prob_fraction,
            "hard_fraction": hard_fraction,
            "expert_counts": expert_counts,
            "probability_sums": probability_sums,
            "num_valid_tokens": router_logits.new_tensor(float(valid_indices.numel())),
            "adjacent_route_flip": adjacent_route_flip,
            "routed_residual_rms": routed_residual_rms,
            **router_aux,
        }
        return output.reshape(batch, bands, frames, self.channels), aux


class DenseTemporalReadout(nn.Module):
    """Active-MAC-matched dense control: one always-on TRR expert, no router."""

    def __init__(
        self,
        channels: int,
        expert_width: int = 96,
        dropout: float = 0.0,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError("moe_residual_scale_init must be in (0, 1)")
        self.channels = int(channels)
        self.group_norms = nn.ModuleList(
            [nn.LayerNorm(channels, eps=1e-6) for _ in range(2)]
        )
        self.expert = DirectionGroupReadoutExpert(channels, expert_width, dropout)
        self.raw_residual_scale = nn.Parameter(torch.tensor(_logit(residual_scale_init)))

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_residual_scale)

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch, _, frames, _ = anchor_bftc.shape
        valid = SharedHierarchicalAcousticRouter._mask(
            batch, frames, anchor_bftc.device, valid_time_mask
        )
        return {"valid_time_mask": valid}

    def _ddp_graph_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        """Give every dense-control parameter a zero gradient for empty masks."""

        anchor = reference.new_zeros(())
        for parameter in self.parameters():
            if parameter.numel() > 0:
                anchor = anchor + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
        return anchor

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        prepared_context: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, bands, frames, _ = raw_states.shape
        valid = prepared_context["valid_time_mask"]
        token_valid = valid[:, None, :].expand(batch, bands, frames).reshape(-1)
        valid_indices = token_valid.nonzero(as_tuple=False).squeeze(1)
        flat_raw = raw_states.reshape(-1, 2 * self.channels)
        output = shared_readout.new_zeros(batch * bands * frames, self.channels)
        if valid_indices.numel() > 0:
            selected = flat_raw.index_select(0, valid_indices)
            group_one = self.group_norms[0](selected[:, : self.channels].float())
            group_two = self.group_norms[1](selected[:, self.channels :].float())
            dense_output = self.expert(group_one, group_two).to(output.dtype)
            dense_output = dense_output * self.residual_scale.to(output.dtype)
            output = output.index_copy(0, valid_indices, dense_output)
        output = output + self._ddp_graph_anchor(output)

        zero = raw_states.float().new_zeros(())
        one = raw_states.float().new_ones(1)
        count = raw_states.float().new_tensor([float(valid_indices.numel())])
        residual = output.index_select(0, valid_indices).float()
        residual_rms = (
            residual.square().mean().sqrt() if residual.numel() > 0 else zero
        )
        return output.reshape(batch, bands, frames, self.channels), {
            "balance_loss": zero,
            "router_z_loss": zero,
            "router_entropy": zero,
            "mean_top1_probability": zero,
            "prob_fraction": one,
            "hard_fraction": one,
            "expert_counts": count,
            "probability_sums": count,
            "num_valid_tokens": count.squeeze(0),
            "adjacent_route_flip": zero,
            "routed_residual_rms": residual_rms,
            "evidence_weights": raw_states.float().new_zeros(5),
            "router_temperature": zero,
            "prototype_max_cosine": zero,
            "prototype_orthogonality_loss": zero,
            "mean_query_norm": zero,
        }


class RefinementAwareDynamicRouter(nn.Module):
    """RADR identity router with token/step-dependent evidence fusion.

    ``delta`` is deliberately defined only between successive tensors emitted
    by the same shared temporal readout.  The first step marks delta evidence
    unavailable and uses only its explicit step embedding; an encoder/readout
    subtraction is never labelled as refinement progress.
    """

    BRANCH_NAMES = ("local", "anchor", "delta", "trajectory", "global")
    DELTA_BRANCH_INDEX = BRANCH_NAMES.index("delta")

    def __init__(
        self,
        channels: int,
        num_experts: int,
        router_dim: int = 24,
        max_refinement_steps: int = 4,
        temperature_init: float = 0.7,
        temperature_min: float = 0.1,
        temperature_max: float = 2.0,
    ):
        super().__init__()
        if channels < 1 or num_experts < 1 or router_dim < 1:
            raise ValueError("channels, num_experts and router_dim must be positive")
        if router_dim < num_experts:
            raise ValueError("router_dim must be >= num_experts")
        if max_refinement_steps < 1:
            raise ValueError("max_refinement_steps must be positive")
        if not 0.0 < temperature_min < temperature_init < temperature_max:
            raise ValueError("router temperatures must satisfy 0 < min < init < max")

        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.router_dim = int(router_dim)
        self.max_refinement_steps = int(max_refinement_steps)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)

        self.local_norm = nn.LayerNorm(2 * channels, eps=1e-6)
        self.local_projection = nn.Linear(2 * channels, router_dim, bias=False)
        self.anchor_norm = nn.LayerNorm(channels, eps=1e-6)
        self.anchor_projection = nn.Linear(channels, router_dim, bias=False)
        self.delta_norm = nn.LayerNorm(channels, eps=1e-6)
        self.delta_projection = nn.Linear(channels, router_dim, bias=False)
        self.trajectory_norm = nn.LayerNorm(3 * channels, eps=1e-6)
        self.trajectory_projection = nn.Linear(3 * channels, router_dim, bias=False)
        self.global_norm = nn.LayerNorm(2 * channels, eps=1e-6)
        self.global_projection = nn.Linear(2 * channels, router_dim, bias=False)

        self.branch_scorers = nn.ModuleList(
            [nn.Linear(router_dim, 1, bias=True) for _ in self.BRANCH_NAMES]
        )
        initial_weights = torch.tensor([0.55, 0.07, 0.20, 0.15, 0.03])
        for scorer, initial_weight in zip(self.branch_scorers, initial_weights):
            nn.init.zeros_(scorer.weight)
            nn.init.constant_(scorer.bias, float(initial_weight.log()))
        self.step_branch_bias = nn.Parameter(
            torch.zeros(max_refinement_steps, len(self.BRANCH_NAMES))
        )
        self.step_embedding = nn.Embedding(max_refinement_steps, router_dim)
        nn.init.zeros_(self.step_embedding.weight)
        # Ablation-only semantic hook.  The production default treats the
        # unavailable first refinement delta as missing evidence.  A matched
        # control may set this to ``zero_observed`` so the delta branch sees a
        # true constant-zero measurement without creating a gradient path back
        # to ``shared_readout``.
        self.first_delta_semantics = "absent"
        # Ablation-only semantic hook.  Disabled evidence is removed from the
        # branch softmax rather than represented by a numeric zero, which would
        # still consume probability mass and change the scale of all survivors.
        self.disabled_evidence_branches: Tuple[str, ...] = ()

        self.prototypes = nn.Parameter(torch.empty(num_experts, router_dim))
        nn.init.orthogonal_(self.prototypes)
        ratio = (temperature_init - temperature_min) / (
            temperature_max - temperature_min
        )
        self.raw_temperature = nn.Parameter(torch.tensor(_logit(ratio)))

    @property
    def temperature(self) -> torch.Tensor:
        span = self.temperature_max - self.temperature_min
        return self.temperature_min + span * torch.sigmoid(self.raw_temperature)

    @staticmethod
    def _normalize(evidence: torch.Tensor) -> torch.Tensor:
        return F.normalize(evidence, p=2.0, dim=-1, eps=1e-6)

    @staticmethod
    def _project_fp32(
        values: torch.Tensor,
        norm: nn.LayerNorm,
        projection: nn.Linear,
    ) -> torch.Tensor:
        device_type = values.device.type
        if device_type in {"cpu", "cuda", "xpu", "mps"}:
            with torch.autocast(device_type=device_type, enabled=False):
                return projection(norm(values.float()))
        return projection(norm(values.float()))

    @staticmethod
    def _mask(
        batch: int,
        frames: int,
        device: torch.device,
        valid_time_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return SharedHierarchicalAcousticRouter._mask(
            batch,
            frames,
            device,
            valid_time_mask,
        )

    @staticmethod
    def _masked_mean_std(
        values: torch.Tensor,
        mask_bft1: torch.Tensor,
        dimensions: Tuple[int, ...],
        denominator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return SharedHierarchicalAcousticRouter._masked_mean_std(
            values,
            mask_bft1,
            dimensions,
            denominator,
        )

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if anchor_bftc.ndim != 4 or anchor_bftc.shape[-1] != self.channels:
            raise ValueError("anchor_bftc must have shape (B,F,T,C)")
        batch, bands, frames, _ = anchor_bftc.shape
        valid = self._mask(batch, frames, anchor_bftc.device, valid_time_mask)
        mask_bft1 = valid[:, None, :, None]
        masked_anchor = anchor_bftc * mask_bft1.to(anchor_bftc.dtype)
        anchor_evidence = self._normalize(
            self._project_fp32(
                masked_anchor,
                self.anchor_norm,
                self.anchor_projection,
            )
        )
        denominator = (
            valid.sum(dim=1).float().mul(float(bands)).clamp_min(1.0)[:, None]
        )
        global_mean, global_std = self._masked_mean_std(
            masked_anchor,
            mask_bft1,
            dimensions=(1, 2),
            denominator=denominator,
        )
        global_statistics = torch.cat([global_mean, global_std], dim=-1)
        global_evidence = self._normalize(
            self._project_fp32(
                global_statistics,
                self.global_norm,
                self.global_projection,
            )
        )[:, None, None, :]
        return {
            "anchor_bftc": masked_anchor,
            "anchor_evidence": anchor_evidence,
            "global_evidence": global_evidence,
            "valid_time_mask": valid,
        }

    def _trajectory_statistics(
        self,
        shared_readout: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, frames, _ = shared_readout.shape
        mask_bft1 = valid[:, None, :, None]
        denominator = valid.sum(dim=1).float().clamp_min(1.0)[:, None, None]
        mean, std = self._masked_mean_std(
            shared_readout,
            mask_bft1,
            dimensions=(2,),
            denominator=denominator,
        )
        if frames == 1:
            mean_absolute_delta = torch.zeros_like(mean)
        else:
            pair_mask = valid[:, 1:] & valid[:, :-1]
            pair_mask_bft1 = pair_mask[:, None, :, None].float()
            pair_denominator = (
                pair_mask.sum(dim=1).float().clamp_min(1.0)[:, None, None]
            )
            within_step_delta = (
                shared_readout[:, :, 1:].float()
                - shared_readout[:, :, :-1].float()
            ).abs()
            mean_absolute_delta = (
                within_step_delta * pair_mask_bft1
            ).sum(dim=2) / pair_denominator
        return torch.cat([mean, std, mean_absolute_delta], dim=-1)

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        previous_shared_readout: Optional[torch.Tensor],
        prepared_context: Dict[str, torch.Tensor],
        step_index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if not 0 <= int(step_index) < self.max_refinement_steps:
            raise ValueError("step_index is outside max_refinement_steps")
        if raw_states.ndim != 4 or raw_states.shape[-1] != 2 * self.channels:
            raise ValueError("raw_states must have shape (B,F,T,2C)")
        if shared_readout.shape != raw_states.shape[:-1] + (self.channels,):
            raise ValueError("shared_readout must align as (B,F,T,C)")

        anchor = prepared_context["anchor_bftc"]
        valid = prepared_context["valid_time_mask"]
        if anchor.shape != shared_readout.shape:
            raise ValueError("prepared anchor and shared_readout shapes differ")
        first_refinement = previous_shared_readout is None
        first_delta_semantics = getattr(self, "first_delta_semantics", "absent")
        if first_delta_semantics not in {"absent", "zero_observed"}:
            raise ValueError(
                "first_delta_semantics must be 'absent' or 'zero_observed'"
            )
        first_delta_zero_observed = (
            first_refinement and first_delta_semantics == "zero_observed"
        )
        if first_refinement:
            # ``detach`` is explicit even though zeros_like currently creates a
            # non-grad tensor.  The semantic control must never acquire the
            # hidden ``current - current.detach()`` backward path that an
            # apparently equivalent forward-zero construction would create.
            refinement_delta = torch.zeros_like(shared_readout).detach()
        else:
            if previous_shared_readout.shape != shared_readout.shape:
                raise ValueError("previous_shared_readout must match shared_readout")
            refinement_delta = shared_readout - previous_shared_readout.detach()
        refinement_delta = refinement_delta * valid[:, None, :, None].to(
            refinement_delta.dtype
        )

        local = self._normalize(
            self._project_fp32(raw_states, self.local_norm, self.local_projection)
        )
        if first_refinement and not first_delta_zero_observed:
            # Delta_1 is semantically absent, not a learnable constant.  Do not
            # send the exact zero tensor through affine LayerNorm followed by
            # L2 normalization: at the origin F.normalize has a 1 / eps
            # backward scale, and one optimizer step can turn LayerNorm's bias
            # into false first-step "progress" evidence.  The enclosing MoE
            # keeps a zero-valued graph anchor for R=1 DDP safety.
            delta_evidence = torch.zeros_like(local)
        else:
            delta_evidence = self._normalize(
                self._project_fp32(
                    refinement_delta,
                    self.delta_norm,
                    self.delta_projection,
                )
            )
        trajectory_statistics = self._trajectory_statistics(shared_readout, valid)
        trajectory = self._normalize(
            self._project_fp32(
                trajectory_statistics,
                self.trajectory_norm,
                self.trajectory_projection,
            )
        )[:, :, None, :].expand_as(local)
        branches = torch.stack(
            [
                local,
                prepared_context["anchor_evidence"],
                delta_evidence,
                trajectory,
                prepared_context["global_evidence"].expand_as(local),
            ],
            dim=-2,
        )
        branch_scores = torch.cat(
            [
                scorer(branches[..., branch_index, :])
                for branch_index, scorer in enumerate(self.branch_scorers)
            ],
            dim=-1,
        )
        branch_scores = branch_scores.float() + self.step_branch_bias[int(step_index)]
        if first_refinement and not first_delta_zero_observed:
            # An unavailable measurement must not consume softmax probability
            # mass or become an implicit gate on the step embedding.
            branch_scores = branch_scores.clone()
            branch_scores[..., self.DELTA_BRANCH_INDEX] = -torch.inf
        disabled_branches = tuple(
            getattr(self, "disabled_evidence_branches", ())
        )
        unknown_disabled = set(disabled_branches).difference(self.BRANCH_NAMES)
        if unknown_disabled:
            raise ValueError(
                "unknown disabled router evidence branches: "
                + ", ".join(sorted(unknown_disabled))
            )
        if len(set(disabled_branches)) >= len(self.BRANCH_NAMES):
            raise ValueError("at least one router evidence branch must remain active")
        if disabled_branches:
            branch_scores = branch_scores.clone()
            for branch_name in disabled_branches:
                branch_scores[..., self.BRANCH_NAMES.index(branch_name)] = -torch.inf
        evidence_weights = torch.softmax(branch_scores, dim=-1)
        step_tensor = torch.tensor(
            int(step_index),
            device=shared_readout.device,
            dtype=torch.long,
        )
        step_embedding = self.step_embedding(step_tensor).float()
        query = self._normalize(
            (branches * evidence_weights.unsqueeze(-1)).sum(dim=-2)
            + step_embedding
        )

        batch, bands, frames, _ = query.shape
        token_valid = valid[:, None, :].expand(batch, bands, frames).reshape(-1)
        valid_indices = token_valid.nonzero(as_tuple=False).squeeze(1)
        valid_query = query.reshape(-1, self.router_dim).index_select(
            0,
            valid_indices,
        )
        prototypes = self._normalize(self.prototypes.float())
        cosine_logits = valid_query @ prototypes.transpose(0, 1)
        scaled_logits = cosine_logits / self.temperature.float()
        probabilities = torch.softmax(scaled_logits, dim=-1)

        prototype_cosine = prototypes @ prototypes.transpose(0, 1)
        if self.num_experts > 1:
            off_diagonal = ~torch.eye(
                self.num_experts,
                device=prototype_cosine.device,
                dtype=torch.bool,
            )
            off_values = prototype_cosine.masked_select(off_diagonal)
            prototype_max_cosine = off_values.abs().max()
            prototype_orthogonality_loss = off_values.square().mean()
        else:
            prototype_max_cosine = prototype_cosine.new_zeros(())
            prototype_orthogonality_loss = prototype_cosine.new_zeros(())

        valid_delta = refinement_delta.reshape(-1, self.channels).index_select(
            0,
            valid_indices,
        ).float()
        valid_delta_evidence = delta_evidence.reshape(
            -1,
            self.router_dim,
        ).index_select(0, valid_indices).float()
        if valid_delta.numel() > 0:
            delta_rms = valid_delta.square().mean().sqrt()
            delta_max = valid_delta.abs().max()
            delta_evidence_rms = valid_delta_evidence.square().mean().sqrt()
            delta_evidence_max = valid_delta_evidence.abs().max()
            mean_evidence_weights = evidence_weights.reshape(
                -1,
                len(self.BRANCH_NAMES),
            ).index_select(0, valid_indices).mean(dim=0)
        else:
            delta_rms = shared_readout.float().new_zeros(())
            delta_max = shared_readout.float().new_zeros(())
            delta_evidence_rms = shared_readout.float().new_zeros(())
            delta_evidence_max = shared_readout.float().new_zeros(())
            mean_evidence_weights = shared_readout.float().new_zeros(
                len(self.BRANCH_NAMES)
            )
        router_aux = {
            "evidence_weights": mean_evidence_weights,
            "router_temperature": self.temperature.float(),
            "prototype_max_cosine": prototype_max_cosine,
            "prototype_orthogonality_loss": prototype_orthogonality_loss,
            "mean_query_norm": (
                valid_query.norm(dim=-1).mean()
                if valid_query.numel() > 0
                else shared_readout.float().new_zeros(())
            ),
            "refinement_delta_rms": delta_rms,
            "refinement_delta_max": delta_max,
            "delta_evidence_rms": delta_evidence_rms,
            "delta_evidence_max": delta_evidence_max,
            "step_embedding_norm": step_embedding.norm(),
            "disabled_evidence_branch_count": shared_readout.float().new_tensor(
                float(len(set(disabled_branches)))
            ),
            "first_delta_zero_observed": shared_readout.float().new_tensor(
                float(first_delta_zero_observed)
            ),
        }
        return (
            valid_indices,
            scaled_logits,
            probabilities,
            valid_query,
            refinement_delta,
            router_aux,
        )


class RefinementAwareSparseTemporalMoE(nn.Module):
    """Sparse acoustic expert identity with an independent strength head."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        expert_width: int = 96,
        dropout: float = 0.0,
        top_k: int = 1,
        router_dim: int = 24,
        max_refinement_steps: int = 4,
        router_temperature_init: float = 0.7,
        router_temperature_min: float = 0.1,
        router_temperature_max: float = 2.0,
        strength_init: float = 0.03,
        strength_max: float = 0.5,
    ):
        super().__init__()
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if not 0.0 < strength_init < strength_max:
            raise ValueError("strength must satisfy 0 < init < max")
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_dim = int(router_dim)
        self.strength_max = float(strength_max)
        self.group_norms = nn.ModuleList(
            [nn.LayerNorm(channels, eps=1e-6) for _ in range(2)]
        )
        self.router = RefinementAwareDynamicRouter(
            channels=channels,
            num_experts=num_experts,
            router_dim=router_dim,
            max_refinement_steps=max_refinement_steps,
            temperature_init=router_temperature_init,
            temperature_min=router_temperature_min,
            temperature_max=router_temperature_max,
        )
        self.experts = nn.ModuleList(
            [
                DirectionGroupReadoutExpert(channels, expert_width, dropout)
                for _ in range(num_experts)
            ]
        )
        strength_hidden = max(4, router_dim // 2)
        self.strength_mlp = nn.Sequential(
            nn.Linear(router_dim + 3, strength_hidden),
            nn.SiLU(),
            nn.Linear(strength_hidden, 1),
        )
        nn.init.normal_(self.strength_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(
            self.strength_mlp[-1].bias,
            _logit(strength_init / strength_max),
        )

    @property
    def residual_scale(self) -> torch.Tensor:
        """Compatibility diagnostic: nominal strength at zero MLP input."""

        return self.strength_max * torch.sigmoid(self.strength_mlp[-1].bias[0])

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.router.prepare_anchor_context(anchor_bftc, valid_time_mask)

    def _ddp_graph_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        anchor = reference.new_zeros(())
        for parameter in self.parameters():
            if parameter.numel() > 0:
                anchor = anchor + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
        return anchor

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        previous_shared_readout: Optional[torch.Tensor],
        prepared_context: Dict[str, torch.Tensor],
        step_index: int,
        return_route_probabilities: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor],
    ]:
        (
            valid_indices,
            router_logits,
            probabilities,
            valid_query,
            refinement_delta,
            router_aux,
        ) = self.router(
            raw_states,
            shared_readout,
            previous_shared_readout,
            prepared_context,
            step_index,
        )
        batch, bands, frames, _ = raw_states.shape
        output = shared_readout.new_zeros(batch * bands * frames, self.channels)
        zero = raw_states.float().new_zeros(())
        empty_vector = raw_states.float().new_zeros(self.num_experts)
        if valid_indices.numel() == 0:
            output = output + self._ddp_graph_anchor(output)
            output = output.reshape(batch, bands, frames, self.channels)
            aux = {
                "balance_loss": zero,
                "router_z_loss": zero,
                "router_entropy": zero,
                "mean_top1_probability": zero,
                "prob_fraction": empty_vector,
                "hard_fraction": empty_vector,
                "expert_counts": empty_vector,
                "probability_sums": empty_vector,
                "num_valid_tokens": zero,
                "adjacent_route_flip": zero,
                "routed_residual_rms": zero,
                "mean_correction_strength": zero,
                "max_correction_strength": zero,
                "straight_through_forward_error": zero,
                **router_aux,
            }
            if return_route_probabilities:
                full_probabilities = raw_states.float().new_zeros(
                    batch,
                    bands,
                    frames,
                    self.num_experts,
                )
                return output, aux, full_probabilities
            return output, aux

        flat_raw = raw_states.reshape(-1, 2 * self.channels).index_select(
            0,
            valid_indices,
        )
        flat_shared = shared_readout.reshape(-1, self.channels).index_select(
            0,
            valid_indices,
        ).float()
        flat_delta = refinement_delta.reshape(-1, self.channels).index_select(
            0,
            valid_indices,
        ).float()
        with torch.autocast(device_type=flat_raw.device.type, enabled=False):
            group_one = self.group_norms[0](flat_raw[:, : self.channels].float())
            group_two = self.group_norms[1](flat_raw[:, self.channels :].float())
            difficulty = torch.stack(
                [
                    flat_delta.square().mean(dim=-1).add(1e-8).sqrt(),
                    flat_shared.square().mean(dim=-1).add(1e-8).sqrt(),
                    flat_raw.float().square().mean(dim=-1).add(1e-8).sqrt(),
                ],
                dim=-1,
            )
            difficulty = torch.log1p(difficulty)
            correction_strength = self.strength_max * torch.sigmoid(
                self.strength_mlp(torch.cat([valid_query.float(), difficulty], dim=-1))
            ).squeeze(-1)

        top1_probability, top1_expert = probabilities.max(dim=-1)
        if self.top_k == 1:
            # Preserve the released top-1 forward and gradient contract exactly:
            # probability selects identity and supplies a straight-through router
            # gradient, while the independent strength head owns amplitude.
            selected_gate = 1.0 + top1_probability - top1_probability.detach()
            for expert_index, expert in enumerate(self.experts):
                local_indices = (top1_expert == expert_index).nonzero(
                    as_tuple=False
                ).squeeze(1)
                if local_indices.numel() == 0:
                    continue
                expert_output = expert(
                    group_one.index_select(0, local_indices),
                    group_two.index_select(0, local_indices),
                )
                amplitude = correction_strength.index_select(0, local_indices)
                amplitude = amplitude * selected_gate.index_select(0, local_indices)
                expert_output = expert_output * amplitude.unsqueeze(-1).to(
                    expert_output.dtype
                )
                global_indices = valid_indices.index_select(0, local_indices)
                output = output.index_copy(
                    0,
                    global_indices,
                    expert_output.to(output.dtype),
                )
            hard_assignment = F.one_hot(
                top1_expert,
                num_classes=self.num_experts,
            ).float()
            straight_through_forward_error = (
                selected_gate.detach() - 1.0
            ).abs().max()
        else:
            # The Top-K sensitivity control keeps one token-wise correction
            # strength.  Within the selected set, router probabilities are
            # renormalized to sum to one, preventing K from multiplying the
            # nominal correction amplitude by construction.
            topk_probability, topk_expert = torch.topk(
                probabilities,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            topk_weight = topk_probability / topk_probability.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-9)
            for expert_index, expert in enumerate(self.experts):
                matches = (topk_expert == expert_index).nonzero(as_tuple=False)
                if matches.numel() == 0:
                    continue
                token_indices = matches[:, 0]
                slot_indices = matches[:, 1]
                expert_output = expert(
                    group_one.index_select(0, token_indices),
                    group_two.index_select(0, token_indices),
                )
                amplitude = correction_strength.index_select(0, token_indices)
                amplitude = amplitude * topk_weight[token_indices, slot_indices]
                expert_output = expert_output * amplitude.unsqueeze(-1).to(
                    expert_output.dtype
                )
                global_indices = valid_indices.index_select(0, token_indices)
                output = output.index_add(
                    0,
                    global_indices,
                    expert_output.to(output.dtype),
                )
            hard_assignment = F.one_hot(
                topk_expert,
                num_classes=self.num_experts,
            ).sum(dim=1).float()
            straight_through_forward_error = zero
        output = output + self._ddp_graph_anchor(output)

        expert_counts = hard_assignment.sum(dim=0)
        probability_sums = probabilities.sum(dim=0)
        hard_fraction = expert_counts / float(valid_indices.numel() * self.top_k)
        prob_fraction = probability_sums / float(valid_indices.numel())
        balance_loss = self.num_experts * torch.sum(hard_fraction * prob_fraction)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        entropy = -(
            probabilities.clamp_min(1e-9)
            * probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()

        full_routes = torch.full(
            (batch * bands * frames,),
            -1,
            device=top1_expert.device,
            dtype=top1_expert.dtype,
        ).index_copy(0, valid_indices, top1_expert).reshape(batch, bands, frames)
        valid = prepared_context["valid_time_mask"]
        if frames > 1:
            pair_mask = (valid[:, 1:] & valid[:, :-1])[:, None, :].expand(
                batch,
                bands,
                frames - 1,
            )
            changed = full_routes[:, :, 1:] != full_routes[:, :, :-1]
            adjacent_route_flip = (
                changed.masked_select(pair_mask).float().mean()
                if bool(pair_mask.any())
                else zero
            )
        else:
            adjacent_route_flip = zero
        routed = output.index_select(0, valid_indices).float()
        routed_rms = routed.square().mean().sqrt()
        output = output.reshape(batch, bands, frames, self.channels)
        aux = {
            "balance_loss": balance_loss,
            "router_z_loss": router_z_loss,
            "router_entropy": entropy,
            "mean_top1_probability": top1_probability.mean(),
            "prob_fraction": prob_fraction,
            "hard_fraction": hard_fraction,
            "expert_counts": expert_counts,
            "probability_sums": probability_sums,
            "num_valid_tokens": router_logits.new_tensor(float(valid_indices.numel())),
            "adjacent_route_flip": adjacent_route_flip,
            "routed_residual_rms": routed_rms,
            "mean_correction_strength": correction_strength.mean(),
            "max_correction_strength": correction_strength.max(),
            "active_experts_per_token": router_logits.new_tensor(float(self.top_k)),
            "straight_through_forward_error": straight_through_forward_error,
            **router_aux,
        }
        if return_route_probabilities:
            full_probabilities = probabilities.new_zeros(
                batch * bands * frames,
                self.num_experts,
            ).index_copy(0, valid_indices, probabilities).reshape(
                batch,
                bands,
                frames,
                self.num_experts,
            )
            return output, aux, full_probabilities
        return output, aux


class RefinementAwareDenseTemporalReadout(DenseTemporalReadout):
    """Dense (non-routed) readout with the refinement-aware call signature."""

    def forward(
        self,
        raw_states: torch.Tensor,
        shared_readout: torch.Tensor,
        previous_shared_readout: Optional[torch.Tensor],
        prepared_context: Dict[str, torch.Tensor],
        step_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del step_index
        output, aux = super().forward(raw_states, shared_readout, prepared_context)
        if previous_shared_readout is None:
            delta = torch.zeros_like(shared_readout)
        else:
            delta = shared_readout - previous_shared_readout.detach()
        valid = prepared_context["valid_time_mask"][:, None, :, None]
        valid_delta = delta.float() * valid
        aux.update(
            {
                "refinement_delta_rms": valid_delta.square().mean().sqrt(),
                "refinement_delta_max": valid_delta.abs().max(),
                "step_embedding_norm": delta.float().new_zeros(()),
                "mean_correction_strength": self.residual_scale.float(),
                "max_correction_strength": self.residual_scale.float(),
                "straight_through_forward_error": delta.float().new_zeros(()),
            }
        )
        return output, aux


class NormClippedStepEmbedding(nn.Module):
    """``nn.Embedding`` whose rows are clipped to a maximum L2 norm.

    The source parameter is **adopted by reference**, never copied or re-drawn,
    so a model built with this wrapper has exactly SEALCore's initialization
    and parameter count.

    Clipping rather than squashing is deliberate. Below the threshold the map is
    the identity, so the mechanism is inert until the embedding actually tries
    to grow past its budget; above it, the direction is preserved and only the
    magnitude is capped. A ``tanh`` would distort every row including small ones
    and can saturate its gradient once driven hard.
    """

    def __init__(self, source: nn.Embedding, max_norm: float):
        super().__init__()
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive; use max_norm=0 on SEAL to disable the cap")
        self.weight = source.weight
        self.max_norm = float(max_norm)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        embedded = F.embedding(index, self.weight)
        norm = embedded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # clamp(max=1) makes this the identity while the row is within budget,
        # which keeps a zero-initialized embedding bit-exact at construction.
        return embedded * (self.max_norm / norm).clamp(max=1.0)
