"""M0: shared-recursive non-causal GTCRN separator with sparse MoE atoms.

This module is a deliberately isolated successor to the V7 single-stage model.
M0 keeps the Single architecture and default fixed-length path; shared helper
layers gained optional masks for variable-length safety.  It keeps the STFT/ERB
encoder, non-causal DPGRNN and no-GSF decoder while changing three architectural
axes:

1. A TIGER-style *single* DPGRNN cell is unrolled ``R`` times with shared
   parameters.  From the second call onward the fixed encoder anchor and the
   preceding state are fused by depthwise 1x1 convolution + PReLU.  Recurrent
   hidden state is explicitly reset (``h=None``) on every refinement call.
2. A Temporal Recurrent Readout MoE (TRR-MoE) operates between the shared
   Temporal BiGRNN and its normalization/attention.  SHAR routes with five
   bounded acoustic evidence branches; selected experts are stateless,
   direction/group-aware gated residual readouts.  Sparse dispatch remains
   gather -> selected expert only -> index_copy with no capacity dropping.
3. The feature decoder feeds a conservation-structured latent-atom head.  Its
   occupancy and ownership are simplexes and its bounded complex correction is
   zero-sum, so group masks analytically sum to ``1 + 0j`` at every TF bin.

``forward`` intentionally returns only separated speech waveforms with shape
``(B, num_sources, L)``.  Differentiable routing/mask/sink tensors from the most
recent call are retained in ``self._last_aux`` for trainer-side auxiliary losses
and research diagnostics.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Package import used by the YAML/entrypoint loader.
    from .gtcrn_ss_noncausal_V7_dpgrnn_single_stage_noise_sink_no_gsf import (
        ConvBlock,
        EncoderV7,
        ERB,
        GRNN,
        GTConvBlock,
        LinearHeadConv,
        NonCausalAttention,
        SFE,
        TRA,
        gLN4D,
    )
except ImportError:  # Allows ``python models/<this_file>.py`` smoke tests.
    from gtcrn_ss_noncausal_V7_dpgrnn_single_stage_noise_sink_no_gsf import (
        ConvBlock,
        EncoderV7,
        ERB,
        GRNN,
        GTConvBlock,
        LinearHeadConv,
        NonCausalAttention,
        SFE,
        TRA,
        gLN4D,
    )


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
            raise ValueError("M0 implements genuine sparse top-1 routing; moe_top_k must be 1")
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


class M0DPGRNNCell(nn.Module):
    """One shared non-causal DPGRNN cell with a routed recurrent readout."""

    def __init__(
        self,
        input_size: int,
        width: int,
        hidden_size: int,
        n_head: int = 2,
        approx_qk_dim: int = 128,
        moe_enabled: bool = True,
        num_experts: int = 6,
        moe_top_k: int = 1,
        moe_expert_width: int = 96,
        moe_dropout: float = 0.0,
        router_dim: int = 24,
        router_temperature_init: float = 0.7,
        router_temperature_min: float = 0.1,
        router_temperature_max: float = 2.0,
        moe_residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if input_size != hidden_size:
            raise ValueError("M0 shared cell requires input_size == hidden_size")
        if hidden_size % 4 != 0:
            raise ValueError("hidden_channels must be divisible by 4 for grouped bidirectional GRNNs")

        self.width = width
        self.hidden_size = hidden_size
        self.moe_enabled = bool(moe_enabled)

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
        self.inter_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.attn = NonCausalAttention(
            hidden_size,
            n_freqs=width,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
        )
        if self.moe_enabled:
            self.temporal_readout = SparseTop1TemporalReadoutMoE(
                channels=hidden_size,
                num_experts=num_experts,
                expert_width=moe_expert_width,
                dropout=moe_dropout,
                top_k=moe_top_k,
                router_dim=router_dim,
                router_temperature_init=router_temperature_init,
                router_temperature_min=router_temperature_min,
                router_temperature_max=router_temperature_max,
                residual_scale_init=moe_residual_scale_init,
            )
        else:
            self.temporal_readout = DenseTemporalReadout(
                channels=hidden_size,
                expert_width=moe_expert_width,
                dropout=moe_dropout,
                residual_scale_init=moe_residual_scale_init,
            )

    def prepare_router_context(
        self,
        fixed_anchor: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute fixed-anchor/global router evidence once per waveform."""

        if fixed_anchor.ndim != 4:
            raise ValueError("fixed_anchor must have shape (B,C,T,F)")
        anchor_bftc = fixed_anchor.permute(0, 3, 2, 1).contiguous()
        return self.temporal_readout.prepare_anchor_context(
            anchor_bftc,
            valid_time_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        fixed_anchor: torch.Tensor,
        router_context: Dict[str, torch.Tensor],
        h: Optional[torch.Tensor] = None,
        valid_time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h is not None:
            raise ValueError("M0 refinement calls must reset recurrent hidden state with h=None")
        if x.ndim != 4:
            raise ValueError(f"DPGRNN input must be (B,C,T,F), got {tuple(x.shape)}")
        b, c, t, f = x.shape
        if c != self.hidden_size or f != self.width:
            raise ValueError(
                f"Expected DPGRNN (C,F)=({self.hidden_size},{self.width}), got ({c},{f})"
            )
        if fixed_anchor.shape != x.shape:
            raise ValueError(
                f"fixed_anchor must match x {tuple(x.shape)}, got {tuple(fixed_anchor.shape)}"
            )
        if valid_time_mask is not None:
            if valid_time_mask.shape != (b, t):
                raise ValueError(
                    f"valid_time_mask must be {(b, t)}, got {tuple(valid_time_mask.shape)}"
                )
            valid_time_mask = valid_time_mask.to(device=x.device, dtype=torch.bool)

        x_tfc = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        time_mask_tfc = None
        if valid_time_mask is not None:
            time_mask_tfc = valid_time_mask[:, :, None, None].to(dtype=x_tfc.dtype)
            x_tfc = x_tfc * time_mask_tfc
        intra_in = x_tfc.reshape(b * t, f, c)
        intra_mix, _ = self.intra_rnn(intra_in, h=None)
        intra_mix = self.intra_fc(intra_mix)
        intra_x = self.intra_ln(intra_mix.reshape(b, t, self.width, self.hidden_size))
        intra_out = x_tfc + intra_x
        if time_mask_tfc is not None:
            intra_out = intra_out * time_mask_tfc

        inter_in = intra_out.permute(0, 2, 1, 3).reshape(b * self.width, t, c)
        inter_lengths = None
        if valid_time_mask is not None:
            frame_lengths = valid_time_mask.sum(dim=1).to(dtype=torch.long)
            inter_lengths = (
                frame_lengths[:, None]
                .expand(b, self.width)
                .contiguous()
                .reshape(b * self.width)
            )
        inter_raw, _ = self.inter_rnn(inter_in, h=None, lengths=inter_lengths)
        shared_readout = self.inter_fc(inter_raw)
        raw_bftc = inter_raw.reshape(b, self.width, t, 2 * c)
        shared_bftc = shared_readout.reshape(b, self.width, t, c)
        readout_delta, moe_aux = self.temporal_readout(
            raw_bftc,
            shared_bftc,
            router_context,
        )
        inter_x = self.inter_ln(
            (shared_bftc + readout_delta).permute(0, 2, 1, 3)
        )
        inter_out = intra_out + inter_x
        if time_mask_tfc is not None:
            inter_out = inter_out * time_mask_tfc

        attn_out = self.attn(
            inter_out.permute(0, 3, 1, 2).contiguous(),
            valid_time_mask=valid_time_mask,
        )
        out = attn_out
        if valid_time_mask is not None:
            out = out * valid_time_mask[:, None, :, None].to(dtype=out.dtype)
        return out, moe_aux


class TigerStyleSharedRecursiveSeparator(nn.Module):
    """Unroll one shared cell with fixed-anchor reinjection for ``R`` steps."""

    def __init__(
        self,
        channels: int,
        width: int,
        num_refinement_steps: int = 4,
        **cell_kwargs,
    ):
        super().__init__()
        if num_refinement_steps < 1:
            raise ValueError("num_refinement_steps must be >= 1")
        self.num_refinement_steps = int(num_refinement_steps)
        self.cell = M0DPGRNNCell(channels, width, channels, **cell_kwargs)
        self.anchor_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True),
            nn.PReLU(),
        )

    def _anchor_fuse_graph_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        """Keep the R=1 ablation DDP-safe without changing its first-step path."""

        dependency = reference.new_zeros(())
        for parameter in self.anchor_fuse.parameters():
            if parameter.numel() > 0:
                dependency = dependency + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
        return dependency

    def forward(
        self,
        encoded: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if encoded.ndim != 4:
            raise ValueError("Shared separator input must have shape (B,C,T,F)")
        # The separator never mutates its input in-place, so a second activation
        # copy is unnecessary. Keeping this reference fixed is sufficient for
        # TIGER-style anchor reinjection and saves one full encoder feature map.
        anchor = encoded
        feature_mask = None
        if valid_time_mask is not None:
            feature_mask = valid_time_mask[:, None, :, None].to(dtype=anchor.dtype)
            anchor = anchor * feature_mask
        state: Optional[torch.Tensor] = None
        step_aux: List[Dict[str, torch.Tensor]] = []
        # Anchor/global router evidence is invariant across refinement steps.
        # Keep it local to this forward (rather than caching on the module) so
        # autograd, re-entrant calls and DDP all retain clean graph ownership.
        router_context = self.cell.prepare_router_context(
            anchor,
            valid_time_mask,
        )

        for step_index in range(self.num_refinement_steps):
            cell_input = anchor if state is None else self.anchor_fuse(anchor + state)
            if feature_mask is not None:
                cell_input = cell_input * feature_mask
            # No DirectionalStateBridge and no hidden carry between refinements.
            state, aux = self.cell(
                cell_input,
                fixed_anchor=anchor,
                router_context=router_context,
                h=None,
                valid_time_mask=valid_time_mask,
            )
            if feature_mask is not None:
                state = state * feature_mask
            aux["step_index"] = aux["num_valid_tokens"].new_tensor(float(step_index))
            step_aux.append(aux)

        if state is None:  # Guard for static type checkers; R is validated >= 1.
            raise RuntimeError("Shared recursive separator produced no state")
        if self.num_refinement_steps == 1:
            # anchor_fuse is intentionally skipped on the first refinement.
            # Give its parameters real zero gradients so an R=1 control works
            # with DDP(find_unused_parameters=False).
            state = state + self._anchor_fuse_graph_dependency(state)
        return state, step_aux


class M0FeatureDecoder(nn.Module):
    """No-GSF V7 decoder whose final output is a fullband mask feature map."""

    def __init__(
        self,
        hidden_channels: int = 64,
        freq_downsample_layers: int = 1,
        mask_head_channels: int = 24,
    ):
        super().__init__()
        if mask_head_channels < 1:
            raise ValueError("mask_head_channels must be >= 1")
        c = hidden_channels
        layers: List[nn.Module] = [
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(10, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(4, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(2, 1), dilation=(1, 1), use_deconv=True),
        ]
        if freq_downsample_layers >= 2:
            layers.append(
                ConvBlock(
                    c,
                    c,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=2,
                    use_deconv=True,
                    is_last=False,
                )
            )
        layers.append(
            LinearHeadConv(
                c,
                mask_head_channels,
                (1, 5),
                stride=(1, 2),
                padding=(0, 2),
                use_deconv=True,
            )
        )
        self.de_convs = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        en_outs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # ``en_outs`` is accepted for Single-decoder API compatibility.  M0 is
        # deliberately the no-GSF ablation, so no encoder skip is fused here.
        del en_outs
        for layer in self.de_convs:
            x = layer(x)
        return x


class ConservationStructuredLatentAtomHead(nn.Module):
    """Map feature atoms to mixture-consistent complex group masks.

    For each time-frequency bin, occupancy ``pi[m]`` is a simplex over atoms
    and ownership ``a[m,k]`` is a simplex over output groups.  The non-negative
    base mask is ``B[k] = sum_m pi[m] a[m,k]``.  A learned complex atom residual
    is centered over ``k`` and magnitude-clipped with a common scale, preserving
    its zero sum.  Consequently ``sum_k M[k] = 1 + 0j`` by construction.
    """

    def __init__(
        self,
        feature_channels: int,
        num_sources: int = 2,
        num_latent_atoms: int = 6,
        noise_sink: bool = True,
        atom_residual_scale: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        if num_latent_atoms < 1:
            raise ValueError("num_latent_atoms must be >= 1")
        if atom_residual_scale < 0.0:
            raise ValueError("atom_residual_scale must be non-negative")

        self.num_sources = int(num_sources)
        self.num_latent_atoms = int(num_latent_atoms)
        self.noise_sink = bool(noise_sink)
        self.group_k = self.num_sources + int(self.noise_sink)
        self.atom_residual_scale = float(atom_residual_scale)
        self.eps = float(eps)

        self.pre = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.PReLU(),
            gLN4D(feature_channels),
        )
        self.occupancy_out = nn.Conv2d(feature_channels, self.num_latent_atoms, kernel_size=1)
        self.ownership_out = nn.Conv2d(
            feature_channels,
            self.num_latent_atoms * self.group_k,
            kernel_size=1,
        )
        self.residual_out = nn.Conv2d(
            feature_channels,
            self.num_latent_atoms * self.group_k * 2,
            kernel_size=1,
        )
        # Start from a real-valued partition; learn phase/interference correction.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def _bounded_zero_sum_residual(
        self,
        logits: torch.Tensor,
        ownership: torch.Tensor,
    ) -> torch.Tensor:
        # logits: (B,M,K,2,T,F); ownership: (B,M,K,T,F)
        if self.atom_residual_scale == 0.0:
            # Preserve a zero-gradient autograd edge to residual_out so the
            # no-complex-correction ablation remains DDP-safe.
            return logits * 0.0
        proposal = torch.tanh(logits) * ownership.unsqueeze(3)
        proposal = proposal * self.atom_residual_scale
        centered = proposal - proposal.mean(dim=2, keepdim=True)

        # Use one scale for every K-vector, preserving its exact zero sum while
        # bounding each complex residual magnitude by atom_residual_scale.
        magnitude = centered.float().square().sum(dim=3).clamp_min(self.eps).sqrt()
        max_magnitude = magnitude.amax(dim=2, keepdim=True)
        scale = (max_magnitude / max(self.atom_residual_scale, self.eps)).clamp_min(1.0)
        return centered / scale.unsqueeze(3).to(centered.dtype)

    @staticmethod
    def _apply_complex_masks(
        mask_real: torch.Tensor,
        mask_imag: torch.Tensor,
        spec: torch.Tensor,
    ) -> torch.Tensor:
        mix_real = spec[:, 0].unsqueeze(1)
        mix_imag = spec[:, 1].unsqueeze(1)
        out_real = mix_real * mask_real - mix_imag * mask_imag
        out_imag = mix_imag * mask_real + mix_real * mask_imag
        return torch.stack([out_real, out_imag], dim=2)

    def forward(
        self,
        feature: torch.Tensor,
        spec: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if feature.ndim != 4 or spec.ndim != 4 or spec.shape[1] != 2:
            raise ValueError("feature/spec must be (B,C,T,F) and (B,2,T,F)")
        b, _, t, f = feature.shape
        if spec.shape[0] != b or spec.shape[2:] != (t, f):
            raise ValueError(
                f"Feature/spec TF shape mismatch: feature={tuple(feature.shape)}, spec={tuple(spec.shape)}"
            )
        if valid_time_mask is not None and valid_time_mask.shape != (b, t):
            raise ValueError(f"valid_time_mask must be {(b, t)}")

        hidden = self.pre(feature)
        occupancy_logits = self.occupancy_out(hidden)
        ownership_logits = self.ownership_out(hidden).reshape(
            b,
            self.num_latent_atoms,
            self.group_k,
            t,
            f,
        )
        residual_logits = self.residual_out(hidden).reshape(
            b,
            self.num_latent_atoms,
            self.group_k,
            2,
            t,
            f,
        )

        # FP32 simplex normalization avoids underflow under mixed precision.
        # Keep the conservation path in FP32 after normalization.  Casting a
        # simplex back to FP16 can make small atoms underflow and makes the sum
        # less exact; the head is cheap relative to the separator activations.
        occupancy = torch.softmax(occupancy_logits.float(), dim=1)
        ownership = torch.softmax(ownership_logits.float(), dim=2)
        base_masks = (occupancy.unsqueeze(2) * ownership).sum(dim=1)

        atom_residual = self._bounded_zero_sum_residual(residual_logits, ownership)
        correction = (occupancy[:, :, None, None] * atom_residual).sum(dim=1)
        mask_real = base_masks + correction[:, :, 0]
        mask_imag = correction[:, :, 1]

        # Analytic construction already conserves the mixture.  Closing the last
        # group from the preceding groups additionally removes FP rounding drift.
        if self.group_k == 1:
            mask_real = torch.ones_like(mask_real)
            mask_imag = torch.zeros_like(mask_imag)
        else:
            mask_real = torch.cat(
                [mask_real[:, :-1], 1.0 - mask_real[:, :-1].sum(dim=1, keepdim=True)],
                dim=1,
            )
            mask_imag = torch.cat(
                [mask_imag[:, :-1], -mask_imag[:, :-1].sum(dim=1, keepdim=True)],
                dim=1,
            )

        grouped_specs = self._apply_complex_masks(mask_real, mask_imag, spec)

        if valid_time_mask is None:
            tf_valid = feature.new_ones((b, 1, t, 1))
        else:
            tf_valid = valid_time_mask.to(device=feature.device, dtype=feature.dtype)
            tf_valid = tf_valid[:, None, :, None]
        mix_magnitude = spec.float().square().sum(dim=1).clamp_min(self.eps).sqrt()
        activity = (mix_magnitude + self.eps) * tf_valid[:, 0].float()
        activity_denom = activity.sum().clamp_min(self.eps)

        atom_utilization = (
            occupancy.float() * activity[:, None]
        ).sum(dim=(0, 2, 3)) / activity_denom

        if self.group_k > 1:
            ownership_entropy_map = -(
                ownership.float().clamp_min(self.eps)
                * ownership.float().clamp_min(self.eps).log()
            ).sum(dim=2) / math.log(self.group_k)
            ownership_weight = occupancy.float() * activity[:, None]
            assignment_entropy_loss = (
                ownership_entropy_map * ownership_weight
            ).sum() / ownership_weight.sum().clamp_min(self.eps)
        else:
            assignment_entropy_loss = feature.float().new_zeros(())

        residual_power = atom_residual.float().square().sum(dim=3)
        residual_weight = occupancy.float().unsqueeze(2) * activity[:, None, None]
        complex_residual_loss = (
            residual_power * residual_weight
        ).sum() / (residual_weight.sum().clamp_min(self.eps) * self.group_k)

        real_sum_error = mask_real.float().sum(dim=1) - 1.0
        imag_sum_error = mask_imag.float().sum(dim=1)
        reconstruction_error = grouped_specs.float().sum(dim=1) - spec.float()

        aux = {
            "grouped_specs": grouped_specs,
            "grouped_mask_real": mask_real,
            "grouped_mask_imag": mask_imag,
            "base_masks": base_masks,
            "occupancy": occupancy,
            "ownership": ownership,
            "atom_complex_residual": atom_residual,
            "complex_correction": correction,
            "atom_utilization": atom_utilization,
            "assignment_entropy_loss": assignment_entropy_loss,
            "complex_residual_loss": complex_residual_loss,
            "mask_sum_real_max_error": real_sum_error.abs().amax(),
            "mask_sum_imag_max_error": imag_sum_error.abs().amax(),
            "mixture_consistency_mse": reconstruction_error.square().mean(),
        }
        return grouped_specs[:, : self.num_sources], aux


class GTCRN_SS_NonCausal_M0_SharedRecursive_MoE_Latent(nn.Module):
    """Complete M0 speech-separation model derived from the Single baseline."""

    def __init__(
        self,
        n_fft: int = 256,
        hop_len: int = 128,
        win_len: int = 256,
        num_sources: int = 2,
        apply_mask_constraint: bool = True,
        num_refinement_steps: int = 4,
        hidden_channels: int = 72,
        freq_downsample_layers: int = 1,
        stft_center: bool = True,
        noise_sink: bool = True,
        architecture_version: str = "m0_trr_shar_v1",
        moe_enabled: bool = True,
        num_experts: int = 6,
        moe_top_k: int = 1,
        moe_expert_width: int = 96,
        moe_dropout: float = 0.0,
        router_dim: int = 24,
        router_temperature_init: float = 0.7,
        router_temperature_min: float = 0.1,
        router_temperature_max: float = 2.0,
        moe_residual_scale_init: float = 0.1,
        step_balance_ratio: float = 0.1,
        num_latent_atoms: int = 6,
        mask_head_channels: int = 24,
        atom_residual_scale: float = 0.5,
        router_z_loss_weight: float = 1e-3,
        prototype_orthogonality_weight: float = 1e-3,
        assignment_entropy_weight: float = 0.0,
        atom_utilization_weight: float = 0.0,
        atom_utilization_floor: float = 0.01,
        complex_residual_weight: float = 0.0,
        refinement_steps: Optional[int] = None,
        latent_masks: Optional[int] = None,
        attention_cache_frames: Optional[int] = None,
        streaming_attention_mode: Optional[str] = None,
    ):
        super().__init__()
        if refinement_steps is not None:
            warnings.warn(
                "refinement_steps is a compatibility alias; use num_refinement_steps",
                DeprecationWarning,
                stacklevel=2,
            )
            num_refinement_steps = int(refinement_steps)
        if latent_masks is not None:
            warnings.warn(
                "latent_masks is a compatibility alias; use num_latent_atoms",
                DeprecationWarning,
                stacklevel=2,
            )
            num_latent_atoms = int(latent_masks)
        if attention_cache_frames is not None or streaming_attention_mode is not None:
            warnings.warn(
                "Streaming attention kwargs have no effect in non-causal M0; ignoring them",
                RuntimeWarning,
                stacklevel=2,
            )
        if not apply_mask_constraint:
            warnings.warn(
                "M0 always enforces analytic complex mixture conservation; "
                "apply_mask_constraint=False is ignored",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_fft <= 0 or hop_len <= 0 or win_len <= 0:
            raise ValueError("n_fft, hop_len and win_len must be positive")
        if win_len > n_fft:
            raise ValueError("win_len cannot exceed n_fft")
        if not stft_center:
            raise ValueError(
                "M0 currently requires stft_center=True: center=False with the "
                "Hann analysis/synthesis window violates PyTorch ISTFT's NOLA "
                "boundary check"
            )
        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        if hidden_channels % 4 != 0:
            raise ValueError("hidden_channels must be divisible by 4")
        if freq_downsample_layers not in (1, 2):
            raise ValueError("M0 currently supports freq_downsample_layers 1 or 2")
        if atom_utilization_floor < 0.0:
            raise ValueError("atom_utilization_floor must be non-negative")
        if architecture_version != "m0_trr_shar_v1":
            raise ValueError(
                "M0 architecture_version must be 'm0_trr_shar_v1'; old post-attention "
                "MoE checkpoints are not strict-resume compatible"
            )
        if not 0.0 <= step_balance_ratio <= 1.0:
            raise ValueError("step_balance_ratio must be in [0, 1]")
        for name, value in {
            "router_z_loss_weight": router_z_loss_weight,
            "prototype_orthogonality_weight": prototype_orthogonality_weight,
            "assignment_entropy_weight": assignment_entropy_weight,
            "atom_utilization_weight": atom_utilization_weight,
            "complex_residual_weight": complex_residual_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)
        self.num_sources = int(num_sources)
        self.hidden_channels = int(hidden_channels)
        self.freq_downsample_layers = int(freq_downsample_layers)
        self.stft_center = bool(stft_center)
        self.noise_sink = bool(noise_sink)
        self.architecture_version = architecture_version
        self.num_refinement_steps = int(num_refinement_steps)
        self.refinement_steps = self.num_refinement_steps  # read-only-style compatibility metadata
        self.num_latent_atoms = int(num_latent_atoms)
        self.latent_masks = self.num_latent_atoms
        self.mask_head_channels = int(mask_head_channels)
        self.moe_enabled = bool(moe_enabled)
        self.num_experts = int(num_experts)
        self.moe_expert_width = int(moe_expert_width)
        self.router_dim = int(router_dim)
        self.step_balance_ratio = float(step_balance_ratio)

        self.router_z_loss_weight = float(router_z_loss_weight)
        self.prototype_orthogonality_weight = float(prototype_orthogonality_weight)
        self.assignment_entropy_weight = float(assignment_entropy_weight)
        self.atom_utilization_weight = float(atom_utilization_weight)
        self.atom_utilization_floor = float(atom_utilization_floor)
        self.complex_residual_weight = float(complex_residual_weight)

        if n_fft >= 512:
            erb_subband_1, erb_subband_2 = 65, 64
            high_lim, fs = 8000, 16000
        else:
            erb_subband_1, erb_subband_2 = 33, 32
            high_lim, fs = 4000, 8000

        dp_width = erb_subband_1
        for _ in range(1, freq_downsample_layers):
            dp_width = (dp_width + 1) // 2

        self.erb = ERB(
            erb_subband_1,
            erb_subband_2,
            nfft=n_fft,
            high_lim=high_lim,
            fs=fs,
        )
        self.sfe = SFE(3, 1)
        self.encoder = EncoderV7(
            hidden_channels=hidden_channels,
            freq_downsample_layers=freq_downsample_layers,
        )
        self.separator = TigerStyleSharedRecursiveSeparator(
            channels=hidden_channels,
            width=dp_width,
            num_refinement_steps=num_refinement_steps,
            moe_enabled=moe_enabled,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
            moe_expert_width=moe_expert_width,
            moe_dropout=moe_dropout,
            router_dim=router_dim,
            router_temperature_init=router_temperature_init,
            router_temperature_min=router_temperature_min,
            router_temperature_max=router_temperature_max,
            moe_residual_scale_init=moe_residual_scale_init,
        )
        self.decoder = M0FeatureDecoder(
            hidden_channels=hidden_channels,
            freq_downsample_layers=freq_downsample_layers,
            mask_head_channels=mask_head_channels,
        )
        self.mask = ConservationStructuredLatentAtomHead(
            feature_channels=mask_head_channels,
            num_sources=num_sources,
            num_latent_atoms=num_latent_atoms,
            noise_sink=noise_sink,
            atom_residual_scale=atom_residual_scale,
        )
        self._last_aux: Optional[Dict[str, object]] = None

    def _iter_gln(self):
        for module in self.modules():
            if isinstance(module, gLN4D):
                yield module

    def _frame_mask(
        self,
        lengths: Optional[torch.Tensor],
        total_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
        if lengths.ndim != 1 or lengths.numel() != batch_size:
            raise ValueError(f"lengths must have shape ({batch_size},), got {tuple(lengths.shape)}")
        if self.stft_center:
            frames = torch.div(lengths, self.hop_len, rounding_mode="floor") + 1
        else:
            frames = torch.div(
                (lengths - self.n_fft).clamp_min(0),
                self.hop_len,
                rounding_mode="floor",
            ) + 1
        if bool(((frames < 1) | (frames > total_frames)).any()):
            raise ValueError("Validated sample lengths produced invalid STFT frame counts")
        indices = torch.arange(total_frames, device=device).unsqueeze(0)
        return indices < frames.unsqueeze(1)

    def _set_gln_mask(self, valid_time_mask: Optional[torch.Tensor]) -> None:
        mask = None if valid_time_mask is None else valid_time_mask[:, None, :, None]
        for module in self.modules():
            if isinstance(module, gLN4D):
                module._mask = mask
            elif isinstance(module, TRA):
                module._mask = valid_time_mask

    def _clear_gln_mask(self) -> None:
        for module in self.modules():
            if isinstance(module, (gLN4D, TRA)):
                module._mask = None

    def _validate_lengths(
        self,
        lengths: Optional[torch.Tensor],
        *,
        batch_size: int,
        n_samples: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Validate true waveform lengths before any non-causal processing."""

        minimum_samples = self.n_fft // 2 + 1 if self.stft_center else self.n_fft
        if lengths is None:
            if n_samples < minimum_samples:
                raise ValueError(
                    f"Input length must be >= {minimum_samples} samples for the configured STFT"
                )
            return None

        raw_lengths = torch.as_tensor(lengths, device=device)
        if raw_lengths.ndim != 1 or raw_lengths.numel() != batch_size:
            raise ValueError(
                f"lengths must have shape ({batch_size},), got {tuple(raw_lengths.shape)}"
            )
        if raw_lengths.dtype == torch.bool:
            raise TypeError("lengths must contain integer sample counts, not booleans")
        if raw_lengths.is_floating_point():
            if not bool(torch.isfinite(raw_lengths).all()):
                raise ValueError("lengths must contain finite sample counts")
            if not bool(torch.equal(raw_lengths, raw_lengths.round())):
                raise ValueError("lengths must contain integer sample counts")

        validated = raw_lengths.to(dtype=torch.long)
        if bool(((validated < minimum_samples) | (validated > n_samples)).any()):
            raise ValueError(
                f"Every length must be within [{minimum_samples}, {n_samples}] samples"
            )
        return validated

    def _stft_with_lengths(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor],
        stft_kwargs: Dict[str, object],
    ) -> torch.Tensor:
        """Compute each cropped STFT before padding its frame axis.

        Cropping before a centered reflect-padded STFT is necessary: merely
        zeroing a padded waveform tail does not reproduce the right boundary
        condition of the original shorter utterance.
        """

        batch_size, n_samples = x.shape
        if lengths is None or bool((lengths == n_samples).all()):
            return torch.stft(x, **stft_kwargs, return_complex=True)

        target_frames = (
            n_samples // self.hop_len + 1
            if self.stft_center
            else (n_samples - self.n_fft) // self.hop_len + 1
        )
        spectra = []
        for batch_index, sample_length in enumerate(lengths.tolist()):
            spectrum = torch.stft(
                x[batch_index, :sample_length],
                **stft_kwargs,
                return_complex=True,
            )
            pad_frames = target_frames - spectrum.shape[-1]
            if pad_frames < 0:
                raise RuntimeError("Per-sample STFT exceeded the padded batch frame count")
            spectra.append(F.pad(spectrum, (0, pad_frames)))
        return torch.stack(spectra, dim=0)

    def _istft_grouped_specs(
        self,
        grouped_specs: torch.Tensor,
        lengths: Optional[torch.Tensor],
        valid_time_mask: Optional[torch.Tensor],
        *,
        n_samples: int,
        stft_kwargs: Dict[str, object],
    ) -> torch.Tensor:
        """Invert valid frames per utterance and right-pad waveform outputs."""

        if lengths is None or bool((lengths == n_samples).all()):
            grouped_waveforms = []
            for group_index in range(self.mask.group_k):
                group_spec = grouped_specs[:, group_index]
                group_complex = torch.complex(group_spec[:, 0], group_spec[:, 1])
                group_complex = group_complex.permute(0, 2, 1).contiguous()
                grouped_waveforms.append(
                    torch.istft(group_complex, length=n_samples, **stft_kwargs)
                )
            return torch.stack(grouped_waveforms, dim=1)

        if valid_time_mask is None:
            raise RuntimeError("Variable-length ISTFT requires a valid frame mask")
        frame_lengths = valid_time_mask.sum(dim=1).to(dtype=torch.long).tolist()
        batch_waveforms = []
        for batch_index, (sample_length, frame_length) in enumerate(
            zip(lengths.tolist(), frame_lengths)
        ):
            group_waveforms = []
            for group_index in range(self.mask.group_k):
                group_spec = grouped_specs[
                    batch_index,
                    group_index,
                    :,
                    :frame_length,
                    :,
                ]
                group_complex = torch.complex(group_spec[0], group_spec[1])
                group_complex = group_complex.permute(1, 0).contiguous()
                waveform = torch.istft(
                    group_complex,
                    length=sample_length,
                    **stft_kwargs,
                )
                group_waveforms.append(F.pad(waveform, (0, n_samples - sample_length)))
            batch_waveforms.append(torch.stack(group_waveforms, dim=0))
        return torch.stack(batch_waveforms, dim=0)

    def set_aux_loss_config(
        self,
        router_z_loss_weight: Optional[float] = None,
        assignment_entropy_weight: Optional[float] = None,
        atom_utilization_weight: Optional[float] = None,
        atom_utilization_floor: Optional[float] = None,
        complex_residual_weight: Optional[float] = None,
    ) -> None:
        """Update internally weighted routing/atom regularizers at runtime."""

        updates = {
            "router_z_loss_weight": router_z_loss_weight,
            "assignment_entropy_weight": assignment_entropy_weight,
            "atom_utilization_weight": atom_utilization_weight,
            "atom_utilization_floor": atom_utilization_floor,
            "complex_residual_weight": complex_residual_weight,
        }
        for name, value in updates.items():
            if value is None:
                continue
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, float(value))

    def _zero(self) -> torch.Tensor:
        return next(self.parameters()).new_zeros((), dtype=torch.float32)

    def clear_m0_aux(self) -> None:
        """Release retained differentiable TF diagnostics after a train step."""

        self._last_aux = None

    def _moe_step_tensors(self, key: str) -> List[torch.Tensor]:
        if self._last_aux is None:
            return []
        steps = self._last_aux.get("moe_steps", [])
        return [step[key] for step in steps if key in step]

    def _weighted_moe_mean(self, key: str) -> torch.Tensor:
        """Average a per-token step statistic using its true valid-token count."""

        if self._last_aux is None:
            return self._zero()
        numerator = self._zero()
        denominator = self._zero()
        for step in self._last_aux.get("moe_steps", []):
            if key not in step or "num_valid_tokens" not in step:
                continue
            count = step["num_valid_tokens"].float()
            numerator = numerator + step[key].float() * count
            denominator = denominator + count
        return numerator / denominator.clamp_min(1.0)

    def _balance_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return global, per-step and configured hybrid Switch balance losses."""

        if not self.moe_enabled or self._last_aux is None:
            zero = self._zero()
            return zero, zero, zero
        steps = self._last_aux.get("moe_steps", [])
        if not steps:
            zero = self._zero()
            return zero, zero, zero

        counts = torch.stack([step["expert_counts"].float() for step in steps]).sum(dim=0)
        probability_sums = torch.stack(
            [step["probability_sums"].float() for step in steps]
        ).sum(dim=0)
        total_tokens = torch.stack(
            [step["num_valid_tokens"].float() for step in steps]
        ).sum()
        denominator = total_tokens.clamp_min(1.0)
        global_hard = counts / denominator
        global_probability = probability_sums / denominator
        global_balance = self.num_experts * torch.sum(global_hard * global_probability)
        step_balance = torch.stack(
            [step["balance_loss"].float() for step in steps]
        ).mean()
        hybrid = (
            (1.0 - self.step_balance_ratio) * global_balance
            + self.step_balance_ratio * step_balance
        )
        return global_balance, step_balance, hybrid

    def _get_balance_loss(self) -> torch.Tensor:
        """Return global + weak-step Switch loss; wrapper applies its weight once."""

        return self._balance_components()[2]

    def _get_routing_aux_loss(self, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return internally weighted router-z and latent-structure losses.

        ``targets`` is accepted for compatibility with ``PITAuxBalanceWrapper``;
        these target-independent structural losses do not consume it.
        """

        del targets
        if self._last_aux is None:
            return self._zero()
        loss = self._zero()
        z_values = self._moe_step_tensors("router_z_loss")
        if z_values and self.router_z_loss_weight > 0.0:
            loss = loss + self.router_z_loss_weight * self._weighted_moe_mean(
                "router_z_loss"
            )
        prototype_values = self._moe_step_tensors("prototype_orthogonality_loss")
        if prototype_values and self.prototype_orthogonality_weight > 0.0:
            # The same shared prototype bank is observed on every refinement;
            # averaging applies this regularizer exactly once rather than R times.
            loss = loss + self.prototype_orthogonality_weight * torch.stack(
                [value.float() for value in prototype_values]
            ).mean()
        loss = loss + self.assignment_entropy_weight * self._last_aux[
            "assignment_entropy_loss"
        ].float()

        utilization = self._last_aux["atom_utilization"].float()
        utilization_loss = torch.relu(
            utilization.new_tensor(self.atom_utilization_floor) - utilization
        ).mean()
        loss = loss + self.atom_utilization_weight * utilization_loss
        loss = loss + self.complex_residual_weight * self._last_aux[
            "complex_residual_loss"
        ].float()
        return loss

    def get_m0_diagnostics(self) -> Dict[str, object]:
        """Return detached statistics from the most recent forward pass."""

        diagnostics: Dict[str, object] = {
            "forward_available": self._last_aux is not None,
            "architecture_version": self.architecture_version,
            "num_refinement_steps": self.num_refinement_steps,
            "shared_cell_count": 1,
            "moe_enabled": self.moe_enabled,
            "num_experts": self.num_experts if self.moe_enabled else 1,
            "num_latent_atoms": self.num_latent_atoms,
            "num_groups": self.mask.group_k,
        }
        if self._last_aux is None:
            return diagnostics

        def detached_mean(key: str) -> torch.Tensor:
            values = self._moe_step_tensors(key)
            if not values:
                return self._zero().detach()
            return torch.stack([value.detach().float() for value in values]).mean(dim=0)

        global_balance, step_balance, hybrid_balance = self._balance_components()
        step_list = self._last_aux.get("moe_steps", [])
        if self.moe_enabled and step_list:
            total_tokens = torch.stack(
                [step["num_valid_tokens"].detach().float() for step in step_list]
            ).sum()
            total_counts = torch.stack(
                [step["expert_counts"].detach().float() for step in step_list]
            ).sum(dim=0)
            total_probability = torch.stack(
                [step["probability_sums"].detach().float() for step in step_list]
            ).sum(dim=0)
            denominator = total_tokens.clamp_min(1.0)
            expert_load = total_counts / denominator
            expert_probability = total_probability / denominator
        else:
            total_tokens = detached_mean("num_valid_tokens")
            total_counts = detached_mean("expert_counts")
            expert_load = detached_mean("hard_fraction")
            expert_probability = detached_mean("prob_fraction")

        diagnostics.update(
            {
                "moe_balance_loss": hybrid_balance.detach(),
                "moe_global_balance_loss": global_balance.detach(),
                "moe_step_balance_loss": step_balance.detach(),
                "router_z_loss": self._weighted_moe_mean("router_z_loss").detach(),
                "router_entropy": detached_mean("router_entropy"),
                "mean_top1_probability": detached_mean("mean_top1_probability"),
                "expert_load": expert_load,
                "expert_probability": expert_probability,
                "expert_counts_total": total_counts,
                "valid_tokens_total": total_tokens,
                "expert_counts_mean_per_step": detached_mean("expert_counts"),
                "valid_tokens_per_step": detached_mean("num_valid_tokens"),
                "evidence_weights": detached_mean("evidence_weights"),
                "router_temperature": detached_mean("router_temperature"),
                "prototype_max_cosine": detached_mean("prototype_max_cosine"),
                "prototype_orthogonality_loss": detached_mean(
                    "prototype_orthogonality_loss"
                ),
                "mean_query_norm": detached_mean("mean_query_norm"),
                "adjacent_route_flip": detached_mean("adjacent_route_flip"),
                "routed_residual_rms": detached_mean("routed_residual_rms"),
                "readout_residual_scale": self.separator.cell.temporal_readout.residual_scale.detach(),
                "routing_aux_loss": self._get_routing_aux_loss().detach(),
                "atom_utilization": self._last_aux["atom_utilization"].detach(),
                "assignment_entropy": self._last_aux["assignment_entropy_loss"].detach(),
                "complex_residual_energy": self._last_aux["complex_residual_loss"].detach(),
                "mask_sum_real_max_error": self._last_aux[
                    "mask_sum_real_max_error"
                ].detach(),
                "mask_sum_imag_max_error": self._last_aux[
                    "mask_sum_imag_max_error"
                ].detach(),
                "mixture_consistency_mse": self._last_aux[
                    "mixture_consistency_mse"
                ].detach(),
            }
        )
        refinement_diagnostics = {}
        for step_index, step in enumerate(self._last_aux.get("moe_steps", [])):
            refinement_diagnostics[f"step_{step_index + 1}"] = {
                key: step[key].detach()
                for key in (
                    "balance_loss",
                    "router_z_loss",
                    "router_entropy",
                    "mean_top1_probability",
                    "hard_fraction",
                    "prob_fraction",
                    "expert_counts",
                    "probability_sums",
                    "num_valid_tokens",
                    "evidence_weights",
                    "router_temperature",
                    "prototype_max_cosine",
                    "prototype_orthogonality_loss",
                    "mean_query_norm",
                    "adjacent_route_flip",
                    "routed_residual_rms",
                )
                if key in step
            }
        diagnostics["refinement"] = refinement_diagnostics
        sink_waveform = self._last_aux.get("sink_waveform")
        if torch.is_tensor(sink_waveform) and sink_waveform.numel() > 0:
            diagnostics["sink_rms"] = sink_waveform.detach().float().square().mean().sqrt()
        return diagnostics

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Input mixture must have shape (B,L), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input mixture must be a floating-point tensor")
        if x.shape[1] < self.n_fft and not self.stft_center:
            raise ValueError("center=False requires input length >= n_fft")

        self._last_aux = None
        device = x.device
        batch_size, n_samples = x.shape
        sample_lengths = self._validate_lengths(
            lengths,
            batch_size=batch_size,
            n_samples=n_samples,
            device=device,
        )
        stft_kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": torch.hann_window(self.win_len, device=device, dtype=x.dtype),
            "onesided": True,
            "center": self.stft_center,
        }

        complex_spec = self._stft_with_lengths(
            x,
            sample_lengths,
            stft_kwargs,
        )  # (B,F,T)
        spec_ri = torch.view_as_real(complex_spec)
        spec_real = spec_ri[..., 0].permute(0, 2, 1)
        spec_imag = spec_ri[..., 1].permute(0, 2, 1)
        spec_magnitude = torch.sqrt(spec_real.square() + spec_imag.square() + 1e-12)
        feature = torch.stack([spec_magnitude, spec_real, spec_imag], dim=1)
        mix_spec = torch.stack([spec_real, spec_imag], dim=1)

        valid_time_mask = self._frame_mask(
            sample_lengths,
            total_frames=feature.shape[2],
            batch_size=batch_size,
            device=device,
        )
        self._set_gln_mask(valid_time_mask)

        try:
            feature = self.erb.bm(feature)
            feature = self.sfe(feature)
            feature, encoder_outputs = self.encoder(feature)
            feature, moe_steps = self.separator(feature, valid_time_mask=valid_time_mask)

            mask_feature_erb = self.decoder(feature, encoder_outputs)
            mask_feature = self.erb.bs(mask_feature_erb)
            if valid_time_mask is not None:
                mask_feature = mask_feature * valid_time_mask[
                    :, None, :, None
                ].to(dtype=mask_feature.dtype)
            if mask_feature.shape[2:] != mix_spec.shape[2:]:
                raise RuntimeError(
                    "Decoder/fullband TF shape does not match mixture: "
                    f"{tuple(mask_feature.shape)} vs {tuple(mix_spec.shape)}"
                )
            _, latent_aux = self.mask(
                mask_feature,
                mix_spec,
                valid_time_mask=valid_time_mask,
            )

            grouped_specs = latent_aux["grouped_specs"]
            grouped_waveforms_tensor = self._istft_grouped_specs(
                grouped_specs,
                sample_lengths,
                valid_time_mask,
                n_samples=n_samples,
                stft_kwargs=stft_kwargs,
            )

            latent_aux["moe_steps"] = moe_steps
            latent_aux["grouped_waveforms"] = grouped_waveforms_tensor
            latent_aux["speech_specs"] = grouped_specs[:, : self.num_sources]
            latent_aux["speech_waveforms"] = grouped_waveforms_tensor[:, : self.num_sources]
            latent_aux["sink_spec"] = grouped_specs[:, self.num_sources :]
            latent_aux["sink_waveform"] = grouped_waveforms_tensor[:, self.num_sources :]
            latent_aux["valid_time_mask"] = valid_time_mask
            latent_aux["lengths"] = sample_lengths
            self._last_aux = latent_aux
            output = grouped_waveforms_tensor[:, : self.num_sources]
        finally:
            self._clear_gln_mask()

        if output.shape != (batch_size, self.num_sources, n_samples):
            raise RuntimeError(
                f"Unexpected M0 output shape {tuple(output.shape)}; "
                f"expected {(batch_size, self.num_sources, n_samples)}"
            )
        return output


if __name__ == "__main__":
    model = GTCRN_SS_NonCausal_M0_SharedRecursive_MoE_Latent(
        n_fft=512,
        hop_len=256,
        win_len=512,
        num_sources=2,
        hidden_channels=72,
        num_refinement_steps=4,
        moe_enabled=True,
        num_experts=6,
        num_latent_atoms=6,
        noise_sink=True,
    ).eval()
    mixture = torch.randn(2, 16000)
    valid_lengths = torch.tensor([16000, 10000])
    with torch.no_grad():
        separated = model(mixture, lengths=valid_lengths)
    print("Input:", mixture.shape, "Output:", separated.shape)
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
    print("Diagnostics:", model.get_m0_diagnostics())
