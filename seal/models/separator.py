"""Shared-weight DPGRNN cell unrolled over refinement steps.

``SharedRecursiveSeparator`` is the SEAL separator. The baseline cell and
separator are built first only to seed SEAL's shared operators with a
paired initialization (see ``SEALCore``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import GRNN, NonCausalAttention
from .routing import (
    DenseTemporalReadout,
    RefinementAwareDenseTemporalReadout,
    RefinementAwareSparseTemporalMoE,
    SparseTop1TemporalReadoutMoE,
    _logit,
)


class BaselineDPGRNNCell(nn.Module):
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
            raise ValueError("shared cell requires input_size == hidden_size")
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
            raise ValueError("refinement calls must reset recurrent hidden state with h=None")
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


class BaselineRecursiveSeparator(nn.Module):
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
        self.cell = BaselineDPGRNNCell(channels, width, channels, **cell_kwargs)
        self.anchor_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True),
            nn.PReLU(),
        )

    def _anchor_fuse_graph_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        """Keep R=1 DDP-safe without changing the first-step path."""

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


class StepAwareFusion(nn.Module):
    """Step-aware anchor/state fusion (SAFR) with a baseline-compatible safety path."""

    def __init__(
        self,
        channels: int,
        fusion_dim: int,
        max_refinement_steps: int = 4,
        blend_init: float = 0.05,
        candidate_scale_init: float = 0.0,
    ):
        super().__init__()
        if not 0.0 < blend_init < 1.0:
            raise ValueError("safr_blend_init must be in (0, 1)")
        if max_refinement_steps < 1:
            raise ValueError("max_refinement_steps must be positive")
        self.channels = int(channels)
        self.max_refinement_steps = int(max_refinement_steps)
        self.num_transition_steps = self.max_refinement_steps - 1
        self.baseline_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=True),
            nn.PReLU(),
        )
        self.anchor_norm = nn.LayerNorm(channels, eps=1e-6)
        self.state_norm = nn.LayerNorm(channels, eps=1e-6)
        self.input_projection = nn.Linear(3 * channels, fusion_dim)
        # SAFR is applied only before refinements 2..R, so it owns R-1
        # transition slots.  Keeping an R-sized table leaves slot zero
        # permanently unused and mislabels transition r-1 -> r as step r.
        self.step_embedding = nn.Embedding(self.num_transition_steps, fusion_dim)
        nn.init.zeros_(self.step_embedding.weight)
        self.output_projection = nn.Linear(fusion_dim, 2 * channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        # Keep the candidate *output* exactly closed through its zero scale,
        # but seed the latent candidate itself.  Initializing both factors to
        # zero would create a bilinear deadlock: neither the candidate nor its
        # scale could receive a first-step gradient.
        nn.init.normal_(
            self.output_projection.weight[channels:],
            mean=0.0,
            std=1e-3,
        )
        self.raw_blend = nn.Parameter(
            torch.full((self.num_transition_steps,), _logit(blend_init))
        )
        self.raw_candidate_scale = nn.Parameter(
            torch.full((self.num_transition_steps,), float(candidate_scale_init))
        )

    def _ddp_graph_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        dependency = reference.new_zeros(())
        for parameter in self.parameters():
            # ``sum`` also covers the legitimate Rmax=1 case, where SAFR has
            # zero transition slots and therefore owns empty parameter rows.
            dependency = dependency + parameter.to(reference.dtype).sum() * 0.0
        return dependency

    def forward(
        self,
        anchor: torch.Tensor,
        previous_state: torch.Tensor,
        step_index: int,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if anchor.shape != previous_state.shape:
            raise ValueError("SAFR anchor and previous_state must match")
        if not 1 <= int(step_index) < self.max_refinement_steps:
            raise ValueError("SAFR is defined only for refinement transitions 1..R-1")
        transition_index = int(step_index) - 1
        baseline = self.baseline_fuse(anchor + previous_state)
        anchor_cl = anchor.permute(0, 2, 3, 1)
        state_cl = previous_state.permute(0, 2, 3, 1)
        anchor_norm = self.anchor_norm(anchor_cl.float())
        state_norm = self.state_norm(state_cl.float())
        joint = torch.cat(
            [anchor_norm, state_norm, state_norm - anchor_norm],
            dim=-1,
        )
        step = torch.tensor(transition_index, device=anchor.device, dtype=torch.long)
        latent = F.silu(self.input_projection(joint) + self.step_embedding(step))
        gate_logits, candidate = self.output_projection(latent).chunk(2, dim=-1)
        trust_gate = torch.sigmoid(gate_logits)
        candidate_scale = torch.tanh(self.raw_candidate_scale[transition_index])
        saf_r = (
            anchor_cl
            + trust_gate.to(anchor_cl.dtype) * (state_cl - anchor_cl)
            + candidate_scale.to(anchor_cl.dtype) * torch.tanh(candidate).to(anchor_cl.dtype)
        ).permute(0, 3, 1, 2).contiguous()
        blend = torch.sigmoid(self.raw_blend[transition_index])
        output = baseline + blend.to(baseline.dtype) * (saf_r - baseline)
        if valid_time_mask is not None:
            output = output * valid_time_mask[:, None, :, None].to(output.dtype)
        valid = (
            torch.ones_like(trust_gate[..., :1], dtype=torch.bool)
            if valid_time_mask is None
            else valid_time_mask[:, :, None, None].expand_as(trust_gate[..., :1])
        )
        selected_gate = trust_gate.masked_select(valid.expand_as(trust_gate)).float()
        candidate_valid = candidate.masked_select(valid.expand_as(candidate)).float()
        zero = anchor.float().new_zeros(())
        diagnostics = {
            "safr_applied": anchor.float().new_ones(()),
            "safr_blend": blend.float(),
            "safr_gate_mean": selected_gate.mean() if selected_gate.numel() else zero,
            "safr_gate_min": selected_gate.min() if selected_gate.numel() else zero,
            "safr_gate_max": selected_gate.max() if selected_gate.numel() else zero,
            "safr_candidate_rms": (
                candidate_valid.square().mean().sqrt()
                if candidate_valid.numel()
                else zero
            ),
            "safr_candidate_scale": candidate_scale.float(),
        }
        return output, diagnostics


class SharedDPGRNNCell(nn.Module):
    """One shared non-causal DPGRNN cell with a RADR temporal readout."""

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
        max_refinement_steps: int = 4,
        router_temperature_init: float = 0.7,
        router_temperature_min: float = 0.1,
        router_temperature_max: float = 2.0,
        moe_strength_init: float = 0.03,
        moe_strength_max: float = 0.5,
    ):
        super().__init__()
        if input_size != hidden_size:
            raise ValueError("shared cell requires input_size == hidden_size")
        if hidden_size % 4 != 0:
            raise ValueError("hidden_channels must be divisible by 4")
        self.width = int(width)
        self.hidden_size = int(hidden_size)
        self.moe_enabled = bool(moe_enabled)
        self.intra_rnn = GRNN(input_size, hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = GRNN(hidden_size, hidden_size, bidirectional=True)
        self.inter_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.attn = NonCausalAttention(
            hidden_size,
            n_freqs=width,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
        )
        if self.moe_enabled:
            self.temporal_readout = RefinementAwareSparseTemporalMoE(
                channels=hidden_size,
                num_experts=num_experts,
                expert_width=moe_expert_width,
                dropout=moe_dropout,
                top_k=moe_top_k,
                router_dim=router_dim,
                max_refinement_steps=max_refinement_steps,
                router_temperature_init=router_temperature_init,
                router_temperature_min=router_temperature_min,
                router_temperature_max=router_temperature_max,
                strength_init=moe_strength_init,
                strength_max=moe_strength_max,
            )
        else:
            self.temporal_readout = RefinementAwareDenseTemporalReadout(
                channels=hidden_size,
                expert_width=moe_expert_width,
                dropout=moe_dropout,
                residual_scale_init=moe_strength_init,
            )

    def prepare_router_context(
        self,
        fixed_anchor: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
        previous_shared_readout: Optional[torch.Tensor],
        step_index: int,
        h: Optional[torch.Tensor] = None,
        valid_time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if h is not None:
            raise ValueError("refinement calls must use h=None")
        if x.ndim != 4:
            raise ValueError("DPGRNN input must have shape (B,C,T,F)")
        b, c, t, f = x.shape
        if c != self.hidden_size or f != self.width:
            raise ValueError("DPGRNN input has unexpected channel/frequency shape")
        if fixed_anchor.shape != x.shape:
            raise ValueError("fixed_anchor must match x")
        if valid_time_mask is not None:
            valid_time_mask = valid_time_mask.to(device=x.device, dtype=torch.bool)

        x_tfc = x.permute(0, 2, 3, 1)
        time_mask_tfc = None
        if valid_time_mask is not None:
            time_mask_tfc = valid_time_mask[:, :, None, None].to(x_tfc.dtype)
            x_tfc = x_tfc * time_mask_tfc
        intra_in = x_tfc.reshape(b * t, f, c)
        intra_mix, _ = self.intra_rnn(intra_in, h=None)
        intra_mix = self.intra_fc(intra_mix)
        intra_x = self.intra_ln(intra_mix.reshape(b, t, f, c))
        intra_out = x_tfc + intra_x
        if time_mask_tfc is not None:
            intra_out = intra_out * time_mask_tfc

        inter_in = intra_out.permute(0, 2, 1, 3).reshape(b * f, t, c)
        inter_lengths = None
        if valid_time_mask is not None:
            frame_lengths = valid_time_mask.sum(dim=1).to(dtype=torch.long)
            inter_lengths = frame_lengths[:, None].expand(b, f).reshape(b * f)
        inter_raw, _ = self.inter_rnn(inter_in, h=None, lengths=inter_lengths)
        shared_readout = self.inter_fc(inter_raw)
        raw_bftc = inter_raw.reshape(b, f, t, 2 * c)
        shared_bftc = shared_readout.reshape(b, f, t, c)
        readout_value, moe_aux = self.temporal_readout(
            raw_bftc,
            shared_bftc,
            previous_shared_readout,
            router_context,
            step_index,
        )
        readout_mode = getattr(
            self.temporal_readout,
            "readout_mode",
            "delta",
        )
        if readout_mode == "delta":
            inter_readout = shared_bftc + readout_value
        elif readout_mode == "full":
            inter_readout = readout_value
        else:
            raise ValueError(
                f"Unsupported temporal readout mode {readout_mode!r}"
            )
        inter_x = self.inter_ln(
            inter_readout.permute(0, 2, 1, 3)
        )
        inter_out = intra_out + inter_x
        if time_mask_tfc is not None:
            inter_out = inter_out * time_mask_tfc
        out = self.attn(
            inter_out.permute(0, 3, 1, 2).contiguous(),
            valid_time_mask=valid_time_mask,
        )
        if valid_time_mask is not None:
            out = out * valid_time_mask[:, None, :, None].to(out.dtype)
        return out, shared_bftc, moe_aux


class SharedRecursiveSeparator(nn.Module):
    """One shared RADR cell unrolled with fixed-anchor SAFR reinjection."""

    def __init__(
        self,
        channels: int,
        width: int,
        num_refinement_steps: int = 4,
        max_refinement_steps: int = 4,
        router_dim: int = 24,
        saf_r_blend_init: float = 0.05,
        saf_r_candidate_scale_init: float = 0.0,
        **cell_kwargs,
    ):
        super().__init__()
        if not 1 <= num_refinement_steps <= max_refinement_steps:
            raise ValueError("num_refinement_steps must be within max_refinement_steps")
        self.num_refinement_steps = int(num_refinement_steps)
        self.max_refinement_steps = int(max_refinement_steps)
        self.fusion = StepAwareFusion(
            channels=channels,
            fusion_dim=router_dim,
            max_refinement_steps=max_refinement_steps,
            blend_init=saf_r_blend_init,
            candidate_scale_init=saf_r_candidate_scale_init,
        )
        self.cell = SharedDPGRNNCell(
            channels,
            width,
            channels,
            router_dim=router_dim,
            max_refinement_steps=max_refinement_steps,
            **cell_kwargs,
        )

    def forward(
        self,
        encoded: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if encoded.ndim != 4:
            raise ValueError("separator input must have shape (B,C,T,F)")
        anchor = encoded
        feature_mask = None
        if valid_time_mask is not None:
            feature_mask = valid_time_mask[:, None, :, None].to(anchor.dtype)
            anchor = anchor * feature_mask
        router_context = self.cell.prepare_router_context(anchor, valid_time_mask)
        state: Optional[torch.Tensor] = None
        previous_readout: Optional[torch.Tensor] = None
        step_aux: List[Dict[str, torch.Tensor]] = []
        for step_index in range(self.num_refinement_steps):
            if state is None:
                cell_input = anchor
                zero = anchor.float().new_zeros(())
                fusion_aux = {
                    "safr_applied": zero,
                    "safr_blend": zero,
                    "safr_gate_mean": zero,
                    "safr_gate_min": zero,
                    "safr_gate_max": zero,
                    "safr_candidate_rms": zero,
                    "safr_candidate_scale": zero,
                }
            else:
                cell_input, fusion_aux = self.fusion(
                    anchor,
                    state,
                    step_index,
                    valid_time_mask,
                )
            state, shared_readout, aux = self.cell(
                cell_input,
                fixed_anchor=anchor,
                router_context=router_context,
                previous_shared_readout=previous_readout,
                step_index=step_index,
                h=None,
                valid_time_mask=valid_time_mask,
            )
            previous_readout = shared_readout
            if feature_mask is not None:
                state = state * feature_mask
            aux.update(fusion_aux)
            aux["step_index"] = aux["num_valid_tokens"].new_tensor(float(step_index))
            step_aux.append(aux)
        if state is None:
            raise RuntimeError("shared recursive separator produced no state")
        if self.num_refinement_steps == 1:
            state = state + self.fusion._ddp_graph_anchor(state)
        return state, step_aux
