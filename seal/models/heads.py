"""Mixture-closed reconstruction heads.

Group masks over two speakers plus a residual sink sum to ``1 + 0j`` at
every time-frequency bin, so the separated sources always add back up to
the mixture.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import gLN4D
from .routing import _logit



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
            # configuration without the complex correction remains DDP-safe.
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


class FullBandObservationAdapter(nn.Module):
    """Inject reliable raw-STFT detail only through a decoder-conditioned gate."""

    def __init__(
        self,
        feature_channels: int,
        scale_init: float = 0.1,
        phase_floor_ratio: float = 0.01,
    ):
        super().__init__()
        if not 0.0 < scale_init < 1.0:
            raise ValueError("observation_scale_init must be in (0, 1)")
        if phase_floor_ratio <= 0.0:
            raise ValueError("phase_floor_ratio must be positive")
        self.feature_channels = int(feature_channels)
        self.phase_floor_ratio = float(phase_floor_ratio)
        self.projection = nn.Conv2d(3, feature_channels, 1)
        self.depthwise = nn.Conv2d(
            feature_channels,
            feature_channels,
            3,
            padding=1,
            groups=feature_channels,
        )
        self.activation = nn.PReLU()
        # Reuse the repository's length-aware global normalization so a
        # right-padded utterance is identical to the same utterance evaluated
        # alone.  Plain GroupNorm would include padded TF bins in its moments.
        self.norm = gLN4D(feature_channels, eps=1e-6)
        self.decoder_gate = nn.Conv2d(
            feature_channels,
            feature_channels,
            1,
            bias=False,
        )
        nn.init.normal_(self.decoder_gate.weight, mean=0.0, std=1e-3)
        self.raw_scale = nn.Parameter(torch.tensor(_logit(scale_init)))

    @property
    def scale(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_scale)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        mix_spec: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if mix_spec.ndim != 4 or mix_spec.shape[1] != 2:
            raise ValueError("mix_spec must have shape (B,2,T,F)")
        if decoder_feature.shape[0] != mix_spec.shape[0] or decoder_feature.shape[2:] != mix_spec.shape[2:]:
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
        denominator = valid_tf.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean_magnitude = (magnitude * valid_tf).sum(dim=(1, 2), keepdim=True) / denominator
        phase_floor = mean_magnitude * self.phase_floor_ratio + 1e-8
        reliability = magnitude / (magnitude + phase_floor)
        observation = torch.stack(
            [
                torch.log1p(magnitude),
                reliability * real / magnitude.clamp_min(1e-8),
                reliability * imag / magnitude.clamp_min(1e-8),
            ],
            dim=1,
        )
        observation = observation * valid[:, None, :, None]
        # Mask immediately after the biased pointwise projection.  Otherwise
        # its bias would populate padded frames and the following 3x3 filter
        # could leak that artificial value back into the final valid frame.
        projected = self.projection(observation)
        projected = projected * valid[:, None, :, None].to(projected.dtype)
        obs_feature = self.norm(self.activation(self.depthwise(projected)))
        gate = torch.tanh(self.decoder_gate(decoder_feature))
        injection = self.scale.to(decoder_feature.dtype) * gate * obs_feature.to(
            decoder_feature.dtype
        )
        fused = decoder_feature + injection
        fused = fused * valid[:, None, :, None].to(fused.dtype)
        valid_gate = gate.masked_select(valid[:, None, :, None].expand_as(gate)).float()
        valid_reliability = reliability.masked_select(valid_tf).float()
        zero = decoder_feature.float().new_zeros(())
        return fused, {
            "observation_scale": self.scale.float(),
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


class ConservedLatentAdditiveResidualHead(
    ConservationStructuredLatentAtomHead
):
    """Keep the six-atom latent mask and add a bounded spectral residual.

    The inherited path is exactly the baseline ``occupancy x ownership``
    complex-mask construction, so it starts from the same stable point as
    the baseline head.  A separate zero-initialized branch predicts two additive
    complex speech corrections.  The sink receives their analytic negative,
    so mixture closure remains exact while the speech estimates are no longer
    restricted to ``M_k * X`` in destructive-cancellation bins.

    The six atoms are intentionally described as latent allocation
    components, not as identifiable phonetic or physical sources.  Their
    energy-weighted contribution and ownership remain exposed for empirical
    stability tests.
    """

    def __init__(
        self,
        feature_channels: int,
        num_sources: int = 2,
        num_latent_atoms: int = 6,
        noise_sink: bool = True,
        atom_residual_scale: float = 0.1,
        additive_residual_enabled: bool = True,
        additive_residual_scale_init: float = 0.1,
        additive_residual_scale_max: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__(
            feature_channels=feature_channels,
            num_sources=num_sources,
            num_latent_atoms=num_latent_atoms,
            noise_sink=noise_sink,
            atom_residual_scale=atom_residual_scale,
            eps=eps,
        )
        if num_sources != 2 or not noise_sink:
            raise ValueError(
                "latent-additive head requires two speech sources plus a sink"
            )
        if additive_residual_enabled and not (
            0.0 <= additive_residual_scale_init < additive_residual_scale_max
        ):
            raise ValueError(
                "additive residual scale must satisfy 0 <= init < max"
            )
        self.feature_channels = int(feature_channels)
        self.additive_residual_enabled = bool(additive_residual_enabled)
        self.additive_residual_scale_max = float(additive_residual_scale_max)

        if self.additive_residual_enabled:
            self.additive_pre = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 1),
                nn.PReLU(),
                gLN4D(feature_channels, eps=1e-6),
            )
            self.additive_depthwise = nn.Conv2d(
                feature_channels,
                feature_channels,
                3,
                padding=1,
                groups=feature_channels,
            )
            self.additive_output = nn.Conv2d(
                feature_channels,
                num_sources * 2,
                1,
            )
            # At initialization the complete head is exactly the baseline head.
            # The output layer receives a task gradient on the first update;
            # the preceding residual feature path joins on subsequent updates.
            nn.init.zeros_(self.additive_output.weight)
            nn.init.zeros_(self.additive_output.bias)
            ratio = max(
                additive_residual_scale_init / additive_residual_scale_max,
                1e-6,
            )
            self.raw_additive_residual_scale = nn.Parameter(
                torch.tensor(_logit(ratio))
            )
        else:
            self.register_buffer(
                "disabled_additive_residual_scale",
                torch.zeros(()),
                persistent=False,
            )

    @property
    def additive_residual_scale(self) -> torch.Tensor:
        if not self.additive_residual_enabled:
            return self.disabled_additive_residual_scale
        return self.additive_residual_scale_max * torch.sigmoid(
            self.raw_additive_residual_scale
        )

    def copy_baseline_mask_initialization(
        self,
        source: ConservationStructuredLatentAtomHead,
    ) -> None:
        """Copy every baseline mask operator with identical shape and role."""

        if source.num_sources != self.num_sources:
            raise ValueError("baseline/SEAL mask source counts differ")
        if source.num_latent_atoms != self.num_latent_atoms:
            raise ValueError("baseline/SEAL latent atom counts differ")
        if source.noise_sink != self.noise_sink:
            raise ValueError("baseline/SEAL mask sink settings differ")
        for name in ("pre", "occupancy_out", "ownership_out", "residual_out"):
            getattr(self, name).load_state_dict(
                getattr(source, name).state_dict(),
                strict=True,
            )

    def _additive_group_residuals(
        self,
        feature: torch.Tensor,
        spec: torch.Tensor,
        base_masks: torch.Tensor,
        valid: torch.Tensor,
        additive_logit_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, t, f = feature.shape
        if not self.additive_residual_enabled:
            grouped = feature.float().new_zeros(
                b,
                self.group_k,
                2,
                t,
                f,
            )
            return grouped, spec.float().new_zeros(b, t, f)

        valid_feature = valid[:, None, :, None].to(feature.dtype)
        hidden = self.additive_pre(feature) * valid_feature
        hidden = F.silu(self.additive_depthwise(hidden)) * valid_feature
        logits = self.additive_output(hidden).reshape(
            b,
            self.num_sources,
            2,
            t,
            f,
        )
        if additive_logit_delta is not None:
            expected_shape = (b, self.num_sources, 2, t, f)
            if tuple(additive_logit_delta.shape) != expected_shape:
                raise ValueError(
                    "additive_logit_delta must have shape "
                    f"{expected_shape}, got {tuple(additive_logit_delta.shape)}"
                )
            logits = logits + additive_logit_delta.to(
                device=logits.device,
                dtype=logits.dtype,
            )

        spec_fp32 = spec.float()
        mix_power = spec_fp32.square().sum(dim=1)
        mix_power = mix_power * valid[:, :, None].to(mix_power.dtype)
        local_power = F.avg_pool2d(
            mix_power[:, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[:, 0]
        local_reference = local_power.add(self.eps).sqrt()

        # Base-mask conditioning keeps the correction tied to the source
        # allocation while the local RMS reference can remain non-zero when
        # the centre mixture bin is cancelled by opposing sources.
        source_prior = (
            base_masks[:, : self.num_sources].clamp_min(self.eps).sqrt()
        )
        speech_raw = (
            torch.tanh(logits.float())
            * source_prior[:, :, None]
            * local_reference[:, None, None]
            * self.additive_residual_scale.float()
        )
        sink_raw = -speech_raw.sum(dim=1, keepdim=True)
        grouped_raw = torch.cat([speech_raw, sink_raw], dim=1)

        # One common scale over all groups preserves the exact zero sum while
        # bounding every complex group correction by rho * local RMS.
        magnitudes = grouped_raw.square().sum(dim=2).add(self.eps).sqrt()
        max_magnitude = magnitudes.amax(dim=1)
        budget = self.additive_residual_scale.float() * local_reference
        common_scale = (max_magnitude / budget.clamp_min(self.eps)).clamp_min(1.0)
        grouped = grouped_raw / common_scale[:, None, None]
        grouped = grouped * (local_power > self.eps)[:, None, None]
        grouped = grouped * valid[:, None, None, :, None]
        return grouped, local_reference

    def forward(
        self,
        feature: torch.Tensor,
        spec: torch.Tensor,
        valid_time_mask: Optional[torch.Tensor] = None,
        additive_logit_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if feature.ndim != 4 or spec.ndim != 4 or spec.shape[1] != 2:
            raise ValueError("feature/spec must be (B,C,T,F) and (B,2,T,F)")
        b, _, t, f = feature.shape
        if valid_time_mask is None:
            valid = torch.ones(b, t, device=feature.device, dtype=torch.bool)
        else:
            if valid_time_mask.shape != (b, t):
                raise ValueError("valid_time_mask has an unexpected shape")
            valid = valid_time_mask.to(device=feature.device, dtype=torch.bool)

        _, base_aux = super().forward(
            feature,
            spec,
            valid_time_mask=valid,
        )
        base_grouped_specs = base_aux["grouped_specs"].float()
        base_masks = base_aux["base_masks"].float()
        additive_grouped, local_reference = self._additive_group_residuals(
            feature,
            spec,
            base_masks,
            valid,
            additive_logit_delta=additive_logit_delta,
        )
        speech_specs = (
            base_grouped_specs[:, : self.num_sources]
            + additive_grouped[:, : self.num_sources]
        )
        # Close from the speech estimates rather than accumulating the two
        # independently exact paths, eliminating their final FP summation drift.
        sink_spec = spec.float() - speech_specs.sum(dim=1)
        grouped_specs = torch.cat([speech_specs, sink_spec[:, None]], dim=1)

        mix_magnitude = spec.float().square().sum(dim=1).add(self.eps).sqrt()
        valid_tf = valid[:, :, None].expand(b, t, f)
        activity = mix_magnitude * valid_tf
        activity_denom = activity.sum().clamp_min(self.eps)
        responsibilities = base_masks
        responsibility_utilization = (
            responsibilities * activity[:, None]
        ).sum(dim=(0, 2, 3)) / activity_denom
        entropy_map = -(
            responsibilities.clamp_min(self.eps)
            * responsibilities.clamp_min(self.eps).log()
        ).sum(dim=1) / math.log(self.group_k)
        responsibility_entropy_loss = (
            entropy_map * activity
        ).sum() / activity_denom

        speech_residuals = additive_grouped[:, : self.num_sources]
        residual_energy = speech_residuals.square().sum(dim=2)
        group_residual_magnitude = additive_grouped.square().sum(dim=2).sqrt()
        group_residual_ratio = group_residual_magnitude / local_reference[
            :, None
        ].clamp_min(self.eps)
        valid_residual_ratio = (
            valid[:, None, :, None]
            & (local_reference[:, None] > self.eps)
        )
        peak_group_residual_ratio = torch.where(
            valid_residual_ratio,
            group_residual_ratio,
            torch.zeros_like(group_residual_ratio),
        ).amax()
        normalized_residual_energy = residual_energy / (
            local_reference[:, None].square() + self.eps
        )
        additive_complex_residual_loss = (
            normalized_residual_energy * activity[:, None]
        ).sum() / (activity_denom * self.num_sources)
        multiplicative_complex_residual_loss = base_aux[
            "complex_residual_loss"
        ].float()
        # Both terms are dimensionless: the baseline regularizes its complex mask
        # correction, while the additive term is normalized by local power.
        complex_residual_loss = (
            multiplicative_complex_residual_loss
            + additive_complex_residual_loss
        )
        base_rms = base_grouped_specs[:, : self.num_sources].square().mean().sqrt()
        residual_rms = speech_residuals.square().mean().sqrt()
        reconstruction_error = grouped_specs.sum(dim=1) - spec.float()
        atom_group_contribution = (
            base_aux["occupancy"].float().unsqueeze(2)
            * base_aux["ownership"].float()
        )
        atom_group_profile = (
            atom_group_contribution * activity[:, None, None]
        ).sum(dim=(0, 3, 4)) / activity_denom
        atom_profile = atom_group_profile.sum(dim=1)
        effective_atom_count = torch.exp(
            -(
                atom_profile.clamp_min(self.eps)
                * atom_profile.clamp_min(self.eps).log()
            ).sum()
        )
        conditional_atom_roles = atom_group_profile / atom_profile[:, None].clamp_min(
            self.eps
        )
        top_roles = conditional_atom_roles.topk(k=2, dim=1).values
        atom_role_margin = top_roles[:, 0] - top_roles[:, 1]

        aux = dict(base_aux)
        aux.update(
            {
                "grouped_specs": grouped_specs,
                "responsibilities": responsibilities,
                "base_grouped_specs": base_grouped_specs,
                "additive_complex_residuals": speech_residuals,
                "additive_grouped_residuals": additive_grouped,
                "local_residual_reference": local_reference,
                "atom_group_contribution": atom_group_contribution,
                "atom_group_profile": atom_group_profile,
                "effective_atom_count": effective_atom_count,
                "atom_role_margin": atom_role_margin,
                "responsibility_utilization": responsibility_utilization,
                "responsibility_entropy_loss": responsibility_entropy_loss,
                "complex_residual_loss": complex_residual_loss,
                "multiplicative_complex_residual_loss": (
                    multiplicative_complex_residual_loss
                ),
                "additive_complex_residual_loss": (
                    additive_complex_residual_loss
                ),
                "residual_to_base_rms_ratio": (
                    residual_rms / base_rms.clamp_min(self.eps)
                ),
                "peak_group_residual_to_local_reference_ratio": (
                    peak_group_residual_ratio
                ),
                "latent_additive_residual_scale": (
                    self.additive_residual_scale.float()
                ),
                # Compatibility key used by existing diagnostics consumers.
                "csr_residual_scale": self.additive_residual_scale.float(),
                "mixture_consistency_mse": reconstruction_error.square().mean(),
                "mixture_consistency_max_error": reconstruction_error.abs().max(),
            }
        )
        return grouped_specs[:, : self.num_sources], aux
