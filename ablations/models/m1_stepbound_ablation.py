"""Single-factor ablation adaptors for the public SEAL ICASSP model.

The canonical model is intentionally left untouched.  This module constructs
the released M1-StepBound graph first and then replaces only the mechanism
named by an ablation arm.  Parameters belonging to a disabled path remain in
the autograd graph through a zero-valued dependency, which keeps DDP safe and
makes capacity accounting explicit.
"""

from __future__ import annotations

from types import MethodType
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gtcrn_ss_noncausal_M1_core import (
    BaselineCompatibleSAFR,
    ConservedLatentAdditiveResidualHead,
    RefinementAwareDynamicRouter,
    RefinementAwareSparseTemporalMoE,
)
from models.gtcrn_ss_noncausal_M1_stepbound import (
    GTCRN_SS_NonCausal_M1_StepBound,
)


def _zero_parameter_dependency(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Keep disabled parameters in the DDP graph without scanning full tensors."""

    dependency = reference.new_zeros(())
    for parameter in module.parameters():
        if parameter.numel():
            dependency = (
                dependency
                + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
            )
    return dependency


class SAFRControl(nn.Module):
    """Nested controls for separating anchor reinjection from adaptive SAFR."""

    MODES = {"baseline_only", "state_only", "no_candidate"}

    def __init__(self, source: BaselineCompatibleSAFR, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"SAFR mode must be one of {sorted(self.MODES)}")
        self.source = source
        self.mode = mode

    def _ddp_graph_anchor(self, reference: torch.Tensor) -> torch.Tensor:
        return _zero_parameter_dependency(self.source, reference)

    def forward(
        self,
        anchor: torch.Tensor,
        previous_state: torch.Tensor,
        step_index: int,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        source = self.source
        if anchor.shape != previous_state.shape:
            raise ValueError("SAFR anchor and previous_state must match")
        if not 1 <= int(step_index) < source.max_refinement_steps:
            raise ValueError("SAFR is defined only for refinement transitions 1..R-1")

        zero = anchor.float().new_zeros(())
        if self.mode in {"baseline_only", "state_only"}:
            fusion_input = (
                anchor + previous_state
                if self.mode == "baseline_only"
                else previous_state
            )
            output = source.baseline_fuse(fusion_input)
            output = output + self._ddp_graph_anchor(output)
            if valid_time_mask is not None:
                output = output * valid_time_mask[:, None, :, None].to(output.dtype)
            return output, {
                "safr_applied": zero,
                "safr_blend": zero,
                "safr_gate_mean": zero,
                "safr_gate_min": zero,
                "safr_gate_max": zero,
                "safr_candidate_rms": zero,
                "safr_candidate_scale": zero,
            }

        # Keep the adaptive anchor/state trust gate but close only the learned
        # candidate correction.  This is the middle point between conventional
        # reinjection and the complete SAFR branch.
        transition_index = int(step_index) - 1
        baseline = source.baseline_fuse(anchor + previous_state)
        anchor_cl = anchor.permute(0, 2, 3, 1)
        state_cl = previous_state.permute(0, 2, 3, 1)
        anchor_norm = source.anchor_norm(anchor_cl.float())
        state_norm = source.state_norm(state_cl.float())
        joint = torch.cat(
            [anchor_norm, state_norm, state_norm - anchor_norm],
            dim=-1,
        )
        step = torch.tensor(transition_index, device=anchor.device, dtype=torch.long)
        latent = F.silu(source.input_projection(joint) + source.step_embedding(step))
        gate_logits, candidate = source.output_projection(latent).chunk(2, dim=-1)
        trust_gate = torch.sigmoid(gate_logits)
        saf_r = (
            anchor_cl + trust_gate.to(anchor_cl.dtype) * (state_cl - anchor_cl)
        ).permute(0, 3, 1, 2).contiguous()
        blend = torch.sigmoid(source.raw_blend[transition_index])
        output = baseline + blend.to(baseline.dtype) * (saf_r - baseline)
        output = output + self._ddp_graph_anchor(output)
        if valid_time_mask is not None:
            output = output * valid_time_mask[:, None, :, None].to(output.dtype)
        valid = (
            torch.ones_like(trust_gate[..., :1], dtype=torch.bool)
            if valid_time_mask is None
            else valid_time_mask[:, :, None, None].expand_as(trust_gate[..., :1])
        )
        selected_gate = trust_gate.masked_select(valid.expand_as(trust_gate)).float()
        candidate_valid = candidate.masked_select(valid.expand_as(candidate)).float()
        return output, {
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
            "safr_candidate_scale": zero,
        }


class FixedHashedRouter(nn.Module):
    """Deterministic position-hash routing without acoustic adaptivity.

    The assignment is a function of the absolute ``(band, frame)`` router
    coordinate alone, so it is a stable property of the router grid: one
    coordinate reaches the same expert in every batch, at every batch slot,
    and for utterances of any duration.  Unlike a flattened round-robin
    assignment, the integer mix avoids a short periodic expert pattern along
    time/frequency.  This is parameter-inventory matched, but not
    functional-capacity matched: the learned router is deliberately idle.
    """

    # Injective packing stride for ``band * STRIDE + frame``.  It must exceed
    # any frame count the model will ever see, so the packed coordinate never
    # depends on the current utterance length.
    COORDINATE_STRIDE = 1_000_003

    def __init__(self, source: RefinementAwareDynamicRouter):
        super().__init__()
        self.source = source
        self.num_experts = source.num_experts

    @property
    def step_embedding(self) -> nn.Module:
        return self.source.step_embedding

    @property
    def step_branch_bias(self) -> nn.Parameter:
        return self.source.step_branch_bias

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.source.prepare_anchor_context(anchor_bftc, valid_time_mask)

    def forward(self, *args, **kwargs):
        raw_states = args[0] if args else kwargs["raw_states"]
        if raw_states.ndim != 4:
            raise ValueError("raw_states must have shape (B,F,T,2C)")
        bands = int(raw_states.shape[1])
        frames = int(raw_states.shape[2])
        if frames >= self.COORDINATE_STRIDE:
            raise ValueError(
                "frame count exceeds COORDINATE_STRIDE; the packed router "
                "coordinate would stop being injective"
            )
        (
            valid_indices,
            source_logits,
            _source_probabilities,
            valid_query,
            refinement_delta,
            router_aux,
        ) = self.source(*args, **kwargs)
        token_count = int(valid_indices.numel())
        fixed_logits = source_logits.new_full((token_count, self.num_experts), -8.0)
        if token_count:
            # ``valid_indices`` is flattened over ``(batch, band, frame)``.
            # Hashing it directly folds the batch slot and the utterance length
            # into the assignment, so one ``(band, frame)`` would reach a
            # different expert in another batch or at another duration.  That
            # is not a fixed assignment: every expert would instead see a fresh
            # random 1/K slice of all token types and the experts would
            # converge to the same average function, which is a weaker control
            # than the intended non-adaptive one.  Recover the absolute
            # time-frequency coordinate before hashing.
            within = torch.remainder(
                valid_indices.to(torch.int64),
                bands * frames,
            )
            band = torch.div(within, frames, rounding_mode="floor")
            frame = torch.remainder(within, frames)
            positions = band * self.COORDINATE_STRIDE + frame
            hashed = positions ^ (positions >> 16)
            hashed = hashed * 0x45D9F3B
            hashed = hashed ^ (hashed >> 16)
            assignment = torch.remainder(hashed, self.num_experts)
            fixed_logits.scatter_(1, assignment[:, None], 8.0)
        fixed_logits = fixed_logits + _zero_parameter_dependency(
            self.source, source_logits
        )
        probabilities = torch.softmax(fixed_logits, dim=-1)
        router_aux = dict(router_aux)
        router_aux["fixed_hashed_routing"] = source_logits.new_ones(())
        return (
            valid_indices,
            fixed_logits,
            probabilities,
            valid_query,
            refinement_delta,
            router_aux,
        )


class GlobalStrengthHead(nn.Module):
    """Use one learned strength while preserving the full parameter inventory."""

    def __init__(self, source: nn.Sequential):
        super().__init__()
        self.source = source

    @property
    def global_logit(self) -> torch.Tensor:
        # Reuse the canonical final bias instead of adding a parameter.  The
        # existing residual_scale diagnostic therefore reports the true value.
        return self.source[-1].bias[0]

    def __getitem__(self, index):
        # M1's compatibility diagnostic reads strength_mlp[-1].bias.
        return self.source[index]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dependency = _zero_parameter_dependency(self.source, inputs)
        output = inputs.new_ones(inputs.shape[:-1] + (1,))
        return output * self.global_logit.to(inputs.dtype) + dependency


class _ProbabilityCache:
    def __init__(self) -> None:
        self.probabilities: Optional[torch.Tensor] = None
        self.utterance_indices: Optional[torch.Tensor] = None


class ProbabilityCachingRouter(nn.Module):
    """Expose the current route probabilities to a controlled strength head."""

    def __init__(self, source: nn.Module, cache: _ProbabilityCache):
        super().__init__()
        self.source = source
        self.cache = cache

    @property
    def step_embedding(self) -> nn.Module:
        return self.source.step_embedding

    @property
    def step_branch_bias(self) -> nn.Parameter:
        return self.source.step_branch_bias

    @property
    def num_experts(self) -> int:
        return int(self.source.num_experts)

    @property
    def router_dim(self) -> int:
        return int(self.source.router_dim)

    @property
    def prototypes(self) -> nn.Parameter:
        return self.source.prototypes

    @property
    def temperature(self) -> torch.Tensor:
        return self.source.temperature

    def prepare_anchor_context(
        self,
        anchor_bftc: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.source.prepare_anchor_context(anchor_bftc, valid_time_mask)

    def forward(self, *args, **kwargs):
        result = self.source(*args, **kwargs)
        self.cache.probabilities = result[2]
        raw_states = args[0] if args else kwargs["raw_states"]
        _, bands, frames, _ = raw_states.shape
        self.cache.utterance_indices = torch.div(
            result[0],
            bands * frames,
            rounding_mode="floor",
        )
        return result


class MeanMatchedTop1TiedStrengthHead(nn.Module):
    """Tie token-to-token amplitude variation to routing confidence.

    A direct ``strength_max * p_top1`` control changes both the coupling and
    the average correction magnitude (with six experts its floor is already
    ``strength_max / 6``).  That is not a clean role-separation test.  This
    head instead reuses the canonical final bias as one learned global mean and
    lets top-1 confidence determine only a bounded, zero-mean modulation around
    it.  Centering is done independently within each utterance, so output does
    not depend on other batch members.  Consequently every utterance-token mean
    is exactly the learned global mean, the parameter inventory is unchanged,
    and any Full-vs-control difference is not explained by a trivial
    mean-amplitude shift.
    """

    MODULATION_FRACTION = 0.95

    def __init__(self, source: nn.Sequential, cache: _ProbabilityCache):
        super().__init__()
        self.source = source
        self.cache = cache

    def __getitem__(self, index):
        return self.source[index]

    @property
    def global_logit(self) -> torch.Tensor:
        return self.source[-1].bias[0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = self.cache.probabilities
        utterance_indices = self.cache.utterance_indices
        if probabilities is None or utterance_indices is None:
            raise RuntimeError("router probabilities are unavailable")
        self.cache.probabilities = None
        self.cache.utterance_indices = None
        if probabilities.shape[0] != inputs.shape[0]:
            raise RuntimeError("router/strength token counts differ")
        top1 = probabilities.max(dim=-1).values.float()
        global_fraction = torch.sigmoid(self.global_logit.float())
        safe_radius = torch.minimum(global_fraction, 1.0 - global_fraction)
        fraction = torch.empty_like(top1)
        for utterance_index in utterance_indices.unique(sorted=True):
            selected = utterance_indices == utterance_index
            confidence = top1.masked_select(selected)
            centered = confidence - confidence.mean()
            # Detaching only the normalizing radius avoids using cross-token
            # scale changes as a second amplitude predictor.  The centered
            # confidence remains differentiable through the router.
            radius = centered.detach().abs().max().clamp_min(1e-6)
            normalized = centered / radius
            current = global_fraction + (
                self.MODULATION_FRACTION * safe_radius * normalized
            )
            fraction.masked_scatter_(selected, current)
        fraction = fraction.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(fraction) - torch.log1p(-fraction)
        dependency = _zero_parameter_dependency(self.source, inputs)
        return logits[:, None].to(inputs.dtype) + dependency


class DirectTop1TiedStrengthHead(nn.Module):
    """Naive secondary control: ``strength = strength_max * p_top1``.

    This intentionally retains the mean-amplitude confound and therefore must
    not be used as the primary evidence for identity/amplitude separation.
    """

    def __init__(self, source: nn.Sequential, cache: _ProbabilityCache):
        super().__init__()
        self.source = source
        self.cache = cache

    def __getitem__(self, index):
        return self.source[index]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = self.cache.probabilities
        if probabilities is None:
            raise RuntimeError("router probabilities are unavailable")
        self.cache.probabilities = None
        self.cache.utterance_indices = None
        if probabilities.shape[0] != inputs.shape[0]:
            raise RuntimeError("router/strength token counts differ")
        top1 = probabilities.max(dim=-1).values.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(top1) - torch.log1p(-top1)
        dependency = _zero_parameter_dependency(self.source, inputs)
        return logits[:, None].to(inputs.dtype) + dependency


class ObservationAblation(nn.Module):
    """Disable observation injection or remove its phase reliability weight."""

    MODES = {"off", "unit_reliability"}

    def __init__(self, source: nn.Module, mode: str):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"observation mode must be one of {sorted(self.MODES)}")
        self.source = source
        self.mode = mode

    def _disabled(
        self,
        decoder_feature: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fused = decoder_feature + _zero_parameter_dependency(self.source, decoder_feature)
        if valid_time_mask is not None:
            fused = fused * valid_time_mask[:, None, :, None].to(fused.dtype)
        zero = decoder_feature.float().new_zeros(())
        return fused, {
            "observation_scale": zero,
            "observation_gate_abs_mean": zero,
            "observation_gate_abs_max": zero,
            "phase_reliability_mean": zero,
            "observation_injection_rms": zero,
            "observation_feature_rms": zero,
        }

    def _unit_reliability(
        self,
        decoder_feature: torch.Tensor,
        mix_spec: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        source = self.source
        if mix_spec.ndim != 4 or mix_spec.shape[1] != 2:
            raise ValueError("mix_spec must have shape (B,2,T,F)")
        if (
            decoder_feature.shape[0] != mix_spec.shape[0]
            or decoder_feature.shape[2:] != mix_spec.shape[2:]
        ):
            raise ValueError("decoder feature and mixture STFT must share B/T/F")
        real = mix_spec[:, 0].float()
        imag = mix_spec[:, 1].float()
        magnitude = real.square().add(imag.square()).add(1e-12).sqrt()
        if valid_time_mask is None:
            valid = torch.ones(
                magnitude.shape[0],
                magnitude.shape[1],
                device=magnitude.device,
                dtype=torch.bool,
            )
        else:
            valid = valid_time_mask.to(device=magnitude.device, dtype=torch.bool)
        valid_tf = valid[:, :, None].expand_as(magnitude)
        reliability = torch.ones_like(magnitude)
        observation = torch.stack(
            [
                torch.log1p(magnitude),
                real / magnitude.clamp_min(1e-8),
                imag / magnitude.clamp_min(1e-8),
            ],
            dim=1,
        )
        observation = observation * valid[:, None, :, None]
        projected = source.projection(observation)
        projected = projected * valid[:, None, :, None].to(projected.dtype)
        obs_feature = source.norm(source.activation(source.depthwise(projected)))
        gate = torch.tanh(source.decoder_gate(decoder_feature))
        injection = source.scale.to(decoder_feature.dtype) * gate * obs_feature.to(
            decoder_feature.dtype
        )
        fused = decoder_feature + injection
        fused = fused * valid[:, None, :, None].to(fused.dtype)
        valid_gate = gate.masked_select(valid[:, None, :, None].expand_as(gate)).float()
        valid_reliability = reliability.masked_select(valid_tf).float()
        zero = decoder_feature.float().new_zeros(())
        return fused, {
            "observation_scale": source.scale.float(),
            "observation_gate_abs_mean": (
                valid_gate.abs().mean() if valid_gate.numel() else zero
            ),
            "observation_gate_abs_max": (
                valid_gate.abs().max() if valid_gate.numel() else zero
            ),
            "phase_reliability_mean": (
                valid_reliability.mean() if valid_reliability.numel() else zero
            ),
            "observation_injection_rms": injection.float().square().mean().sqrt(),
            "observation_feature_rms": obs_feature.float().square().mean().sqrt(),
        }

    def forward(
        self,
        decoder_feature: torch.Tensor,
        mix_spec: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.mode == "off":
            return self._disabled(decoder_feature, valid_time_mask)
        return self._unit_reliability(decoder_feature, mix_spec, valid_time_mask)


def _ablation_additive_group_residuals(
    self: ConservedLatentAdditiveResidualHead,
    feature: torch.Tensor,
    spec: torch.Tensor,
    base_masks: torch.Tensor,
    valid: torch.Tensor,
    additive_logit_delta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """M1 additive head with one controlled physical-assumption substitution."""

    b, _, t, f = feature.shape
    if not self.additive_residual_enabled:
        grouped = feature.float().new_zeros(b, self.group_k, 2, t, f)
        return grouped, spec.float().new_zeros(b, t, f)

    valid_feature = valid[:, None, :, None].to(feature.dtype)
    hidden = self.additive_pre(feature) * valid_feature
    hidden = F.silu(self.additive_depthwise(hidden)) * valid_feature
    logits = self.additive_output(hidden).reshape(b, self.num_sources, 2, t, f)
    if additive_logit_delta is not None:
        expected_shape = (b, self.num_sources, 2, t, f)
        if tuple(additive_logit_delta.shape) != expected_shape:
            raise ValueError(
                "additive_logit_delta must have shape "
                f"{expected_shape}, got {tuple(additive_logit_delta.shape)}"
            )
        logits = logits + additive_logit_delta.to(logits.device, logits.dtype)

    spec_fp32 = spec.float()
    mix_power = spec_fp32.square().sum(dim=1)
    mix_power = mix_power * valid[:, :, None].to(mix_power.dtype)
    if self.ablation_reference_mode == "center_bin":
        local_power = mix_power
    else:
        local_power = F.avg_pool2d(
            mix_power[:, None], kernel_size=3, stride=1, padding=1
        )[:, 0]
    local_reference = local_power.add(self.eps).sqrt()

    if self.ablation_source_prior_mode == "none":
        source_prior = torch.ones_like(base_masks[:, : self.num_sources])
    else:
        source_prior = base_masks[:, : self.num_sources].clamp_min(self.eps).sqrt()
    speech_raw = (
        torch.tanh(logits.float())
        * source_prior[:, :, None]
        * local_reference[:, None, None]
        * self.additive_residual_scale.float()
    )
    sink_raw = -speech_raw.sum(dim=1, keepdim=True)
    grouped_raw = torch.cat([speech_raw, sink_raw], dim=1)
    magnitudes = grouped_raw.square().sum(dim=2).add(self.eps).sqrt()
    max_magnitude = magnitudes.amax(dim=1)
    budget = self.additive_residual_scale.float() * local_reference
    common_scale = (max_magnitude / budget.clamp_min(self.eps)).clamp_min(1.0)
    grouped = grouped_raw / common_scale[:, None, None]
    grouped = grouped * (local_power > self.eps)[:, None, None]
    grouped = grouped * valid[:, None, None, :, None]
    return grouped, local_reference


class GTCRN_SS_NonCausal_M1_StepBound_Ablation(
    GTCRN_SS_NonCausal_M1_StepBound
):
    """Configurable, single-factor adaptor around canonical M1-StepBound."""

    DEFAULTS = {
        "safr_mode": "full",
        "routing_mode": "dynamic",
        "router_evidence_mode": "full",
        "strength_mode": "adaptive",
        "router_step_signal": "bounded",
        "first_delta_semantics": "absent",
        "observation_mode": "reliability_gated",
        "residual_reference_mode": "local_rms",
        "source_prior_mode": "sqrt_mask",
    }
    ALLOWED = {
        "safr_mode": {"full", "baseline_only", "state_only", "no_candidate"},
        "routing_mode": {"dynamic", "fixed_hashed"},
        "router_evidence_mode": {"full", "no_delta", "no_progress"},
        "strength_mode": {
            "adaptive",
            "global_scalar",
            "top1_mean_matched",
            "top1_probability_direct",
        },
        "router_step_signal": {
            "bounded",
            "no_embedding",
            "no_branch_bias",
            "none",
        },
        "first_delta_semantics": {"absent", "zero_observed"},
        "observation_mode": {"reliability_gated", "off", "unit_reliability"},
        "residual_reference_mode": {"local_rms", "center_bin"},
        "source_prior_mode": {"sqrt_mask", "none"},
    }

    def __init__(
        self,
        *args,
        ablation_id: str,
        saf_r_mode: str = "full",
        routing_mode: str = "dynamic",
        router_evidence_mode: str = "full",
        strength_mode: str = "adaptive",
        router_step_signal: str = "bounded",
        first_delta_semantics: str = "absent",
        observation_mode: str = "reliability_gated",
        residual_reference_mode: str = "local_rms",
        source_prior_mode: str = "sqrt_mask",
        **kwargs,
    ):
        settings = {
            "safr_mode": saf_r_mode,
            "routing_mode": routing_mode,
            "router_evidence_mode": router_evidence_mode,
            "strength_mode": strength_mode,
            "router_step_signal": router_step_signal,
            "first_delta_semantics": first_delta_semantics,
            "observation_mode": observation_mode,
            "residual_reference_mode": residual_reference_mode,
            "source_prior_mode": source_prior_mode,
        }
        for name, value in settings.items():
            if value not in self.ALLOWED[name]:
                raise ValueError(
                    f"{name}={value!r}; expected one of {sorted(self.ALLOWED[name])}"
                )
        changed = [
            name for name, value in settings.items() if value != self.DEFAULTS[name]
        ]
        if len(changed) != 1:
            raise ValueError(
                "Each training arm must change exactly one mechanism; changed "
                + (", ".join(changed) if changed else "none")
            )

        super().__init__(*args, **kwargs)
        self.ablation_id = str(ablation_id)
        self.ablation_settings = settings

        readouts = [
            module
            for module in self.modules()
            if isinstance(module, RefinementAwareSparseTemporalMoE)
        ]
        if router_evidence_mode != "full":
            disabled = (
                ("delta",)
                if router_evidence_mode == "no_delta"
                else ("delta", "trajectory")
            )
            for readout in readouts:
                readout.router.disabled_evidence_branches = disabled

        if router_step_signal in {"no_embedding", "none"}:
            for readout in readouts:
                router = readout.router
                embedding = router.step_embedding
                with torch.no_grad():
                    embedding.weight.zero_()
                embedding.weight.requires_grad_(False)

        if router_step_signal in {"no_branch_bias", "none"}:
            for readout in readouts:
                router = readout.router
                with torch.no_grad():
                    router.step_branch_bias.zero_()
                router.step_branch_bias.requires_grad_(False)

        if first_delta_semantics == "zero_observed":
            for readout in readouts:
                # The canonical router owns an explicit semantic hook.  This
                # keeps the forward zero constant and detached while allowing
                # the delta branch's own affine parameters to learn what an
                # observed zero means.
                readout.router.first_delta_semantics = "zero_observed"

        if routing_mode == "fixed_hashed":
            for readout in readouts:
                readout.router = FixedHashedRouter(readout.router)
            self.router_z_loss_weight = 0.0
            self.prototype_orthogonality_weight = 0.0

        if strength_mode == "global_scalar":
            for readout in readouts:
                readout.strength_mlp = GlobalStrengthHead(readout.strength_mlp)

        if strength_mode in {"top1_mean_matched", "top1_probability_direct"}:
            for readout in readouts:
                cache = _ProbabilityCache()
                readout.router = ProbabilityCachingRouter(readout.router, cache)
                head_class = (
                    MeanMatchedTop1TiedStrengthHead
                    if strength_mode == "top1_mean_matched"
                    else DirectTop1TiedStrengthHead
                )
                readout.strength_mlp = head_class(readout.strength_mlp, cache)

        if saf_r_mode != "full":
            if not isinstance(self.separator.fusion, BaselineCompatibleSAFR):
                raise TypeError("Unexpected SAFR implementation")
            self.separator.fusion = SAFRControl(self.separator.fusion, saf_r_mode)

        if observation_mode in ObservationAblation.MODES:
            self.observation = ObservationAblation(self.observation, observation_mode)

        if residual_reference_mode != "local_rms" or source_prior_mode != "sqrt_mask":
            if not isinstance(self.mask, ConservedLatentAdditiveResidualHead):
                raise TypeError("Residual physical ablations require latent_additive head")
            self.mask.ablation_reference_mode = residual_reference_mode
            self.mask.ablation_source_prior_mode = source_prior_mode
            self.mask._additive_group_residuals = MethodType(
                _ablation_additive_group_residuals,
                self.mask,
            )

    def ablation_report(self) -> Dict[str, object]:
        return {
            "ablation_id": self.ablation_id,
            "settings": dict(self.ablation_settings),
            "step_embedding_max_norm": self.step_embedding_max_norm,
        }

    def get_m1_diagnostics(self) -> Dict[str, object]:
        diagnostics = super().get_m1_diagnostics()
        diagnostics["ablation"] = self.ablation_report()
        return diagnostics
