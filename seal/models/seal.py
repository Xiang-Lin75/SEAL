"""SEAL: mixture-closed reconstruction and refinement-aware routing."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ERB, SFE, TRA, Encoder, FeatureDecoder, gLN4D
from .heads import (
    ConservationStructuredLatentAtomHead,
    ConservedLatentAdditiveResidualHead,
    FullBandObservationAdapter,
)
from .routing import NormClippedStepEmbedding
from .separator import BaselineRecursiveSeparator, SharedRecursiveSeparator


class SEALBaseline(nn.Module):
    """Shared-recursive baseline that SEAL builds on.

    It owns everything SEAL keeps unchanged: the STFT/ERB front-end, the
    encoder and decoder, variable-length handling, and the routing
    auxiliary losses. Its separator uses a Top-1 temporal readout MoE and its
    mask is the six-atom latent head whose group masks sum to ``1 + 0j``.

    ``forward`` returns separated speech waveforms of shape
    ``(B, num_sources, L)``. Differentiable routing, mask and sink tensors from
    the most recent call are kept in ``self._last_aux`` for the trainer's
    auxiliary losses and diagnostics.
    """

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
        architecture_version: str = "seal_baseline_v1",
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
    ):
        super().__init__()
        if not apply_mask_constraint:
            warnings.warn(
                "SEAL always enforces analytic complex mixture conservation; "
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
                "SEAL requires stft_center=True: center=False with the "
                "Hann analysis/synthesis window violates PyTorch ISTFT's NOLA "
                "boundary check"
            )
        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        if hidden_channels % 4 != 0:
            raise ValueError("hidden_channels must be divisible by 4")
        if freq_downsample_layers not in (1, 2):
            raise ValueError("freq_downsample_layers must be 1 or 2")
        if atom_utilization_floor < 0.0:
            raise ValueError("atom_utilization_floor must be non-negative")
        if architecture_version != "seal_baseline_v1":
            raise ValueError("architecture_version must be 'seal_baseline_v1'")
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
        self.num_latent_atoms = int(num_latent_atoms)
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
        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            freq_downsample_layers=freq_downsample_layers,
        )
        self.separator = BaselineRecursiveSeparator(
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
        self.decoder = FeatureDecoder(
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

    def clear_aux(self) -> None:
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

    def get_diagnostics(self) -> Dict[str, object]:
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
                f"Unexpected output shape {tuple(output.shape)}; "
                f"expected {(batch_size, self.num_sources, n_samples)}"
            )
        return output


class SEALCore(SEALBaseline):
    """SEAL without the step-cue norm cap.

    Relative to :class:`SEALBaseline` it replaces the separator and the mask:

    * step-aware anchor/state fusion (SAFR) between refinement steps;
    * refinement-aware dynamic routing (RADR), which separates the hard expert
      choice from a bounded correction strength;
    * a reliability-gated full-band observation adapter;
    * the six-atom latent mask followed by a zero-initialized, bounded
      additive complex correction.

    The front-end, encoder and decoder are inherited unchanged.
    """

    @staticmethod
    def _copy_baseline_initialization(
        source_separator: nn.Module,
        target_separator: SharedRecursiveSeparator,
        *,
        moe_enabled: bool,
    ) -> None:
        """Copy the baseline's initial weights into SEAL's matching operators.

        SEAL is constructed after an inherited, randomly initialized baseline
        graph. Operators with identical shape *and* role start from the same
        weights; SEAL-only parameters stay freshly initialized. This is still a
        from-scratch initialization, not checkpoint warm-starting.
        """

        source_cell = source_separator.cell
        target_cell = target_separator.cell
        for name in (
            "intra_rnn",
            "intra_fc",
            "intra_ln",
            "inter_rnn",
            "inter_fc",
            "inter_ln",
            "attn",
        ):
            getattr(target_cell, name).load_state_dict(
                getattr(source_cell, name).state_dict(),
                strict=True,
            )
        target_separator.fusion.baseline_fuse.load_state_dict(
            source_separator.anchor_fuse.state_dict(),
            strict=True,
        )

        source_readout = source_cell.temporal_readout
        target_readout = target_cell.temporal_readout
        if not moe_enabled:
            target_readout.load_state_dict(source_readout.state_dict(), strict=True)
            return

        target_readout.group_norms.load_state_dict(
            source_readout.group_norms.state_dict(),
            strict=True,
        )
        target_readout.experts.load_state_dict(
            source_readout.experts.state_dict(),
            strict=True,
        )
        source_router = source_readout.router
        target_router = target_readout.router
        for name in (
            "local_norm",
            "local_projection",
            "anchor_norm",
            "anchor_projection",
            "trajectory_norm",
            "trajectory_projection",
            "global_norm",
            "global_projection",
        ):
            getattr(target_router, name).load_state_dict(
                getattr(source_router, name).state_dict(),
                strict=True,
            )
        with torch.no_grad():
            target_router.prototypes.copy_(source_router.prototypes)
            target_router.raw_temperature.copy_(source_router.raw_temperature)

    def __init__(
        self,
        n_fft: int = 256,
        hop_len: int = 128,
        win_len: int = 256,
        num_sources: int = 2,
        apply_mask_constraint: bool = True,
        num_refinement_steps: int = 4,
        max_refinement_steps: int = 4,
        hidden_channels: int = 72,
        freq_downsample_layers: int = 1,
        stft_center: bool = True,
        noise_sink: bool = True,
        architecture_version: str = "seal_core_v1",
        mask_head_type: str = "latent_additive",
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
        moe_strength_init: float = 0.03,
        moe_strength_max: float = 0.5,
        step_balance_ratio: float = 0.1,
        mask_head_channels: int = 24,
        saf_r_blend_init: float = 0.05,
        saf_r_candidate_scale_init: float = 0.0,
        observation_scale_init: float = 0.1,
        observation_phase_floor_ratio: float = 0.01,
        latent_additive_residual_scale_init: float = 0.1,
        latent_additive_residual_scale_max: float = 0.5,
        router_z_loss_weight: float = 1e-3,
        prototype_orthogonality_weight: float = 1e-3,
        responsibility_entropy_weight: float = 0.0,
        responsibility_utilization_weight: float = 0.0,
        responsibility_utilization_floor: float = 0.01,
        complex_residual_weight: float = 0.0,
        paired_initialization: bool = True,
        num_latent_atoms: int = 6,
        atom_residual_scale: float = 0.1,
    ):
        if architecture_version != "seal_core_v1":
            raise ValueError("architecture_version must be 'seal_core_v1'")
        if mask_head_type != "latent_additive":
            raise ValueError("mask_head_type must be 'latent_additive'")
        if int(num_latent_atoms) < 1:
            raise ValueError("num_latent_atoms must be positive")
        requested_moe_top_k = int(moe_top_k)
        if not 1 <= requested_moe_top_k <= int(num_experts):
            raise ValueError(
                "moe_top_k must satisfy 1 <= moe_top_k <= num_experts"
            )
        # Build the baseline graph first, then replace only its separator and
        # mask. The decoder is identical, so the inherited instance is kept.
        # Replaced modules are no longer registered and add no parameters.
        super().__init__(
            n_fft=n_fft,
            hop_len=hop_len,
            win_len=win_len,
            num_sources=num_sources,
            apply_mask_constraint=apply_mask_constraint,
            num_refinement_steps=num_refinement_steps,
            hidden_channels=hidden_channels,
            freq_downsample_layers=freq_downsample_layers,
            stft_center=stft_center,
            noise_sink=noise_sink,
            architecture_version="seal_baseline_v1",
            moe_enabled=moe_enabled,
            num_experts=num_experts,
            # The baseline separator is only an initialization source and
            # implements Top-1 exclusively; SEAL's separator below receives the
            # requested moe_top_k.
            moe_top_k=1,
            moe_expert_width=moe_expert_width,
            moe_dropout=moe_dropout,
            router_dim=router_dim,
            router_temperature_init=router_temperature_init,
            router_temperature_min=router_temperature_min,
            router_temperature_max=router_temperature_max,
            moe_residual_scale_init=moe_residual_scale_init,
            step_balance_ratio=step_balance_ratio,
            num_latent_atoms=int(num_latent_atoms),
            mask_head_channels=mask_head_channels,
            atom_residual_scale=float(atom_residual_scale),
            router_z_loss_weight=router_z_loss_weight,
            prototype_orthogonality_weight=prototype_orthogonality_weight,
            assignment_entropy_weight=0.0,
            atom_utilization_weight=0.0,
            atom_utilization_floor=0.0,
            complex_residual_weight=complex_residual_weight,
        )
        baseline_separator = self.separator
        baseline_mask = self.mask
        if num_sources != 2 or not noise_sink:
            raise ValueError("SEAL requires two speech sources plus a sink")
        if not 1 <= num_refinement_steps <= max_refinement_steps:
            raise ValueError("num_refinement_steps must be within max_refinement_steps")

        if n_fft >= 512:
            dp_width = 65
        else:
            dp_width = 33
        for _ in range(1, freq_downsample_layers):
            dp_width = (dp_width + 1) // 2
        self.separator = SharedRecursiveSeparator(
            channels=hidden_channels,
            width=dp_width,
            num_refinement_steps=num_refinement_steps,
            max_refinement_steps=max_refinement_steps,
            router_dim=router_dim,
            saf_r_blend_init=saf_r_blend_init,
            saf_r_candidate_scale_init=saf_r_candidate_scale_init,
            moe_enabled=moe_enabled,
            num_experts=num_experts,
            moe_top_k=requested_moe_top_k,
            moe_expert_width=moe_expert_width,
            moe_dropout=moe_dropout,
            router_temperature_init=router_temperature_init,
            router_temperature_min=router_temperature_min,
            router_temperature_max=router_temperature_max,
            moe_strength_init=moe_strength_init,
            moe_strength_max=moe_strength_max,
        )
        if paired_initialization:
            self._copy_baseline_initialization(
                baseline_separator,
                self.separator,
                moe_enabled=moe_enabled,
            )
        self.observation = FullBandObservationAdapter(
            feature_channels=mask_head_channels,
            scale_init=observation_scale_init,
            phase_floor_ratio=observation_phase_floor_ratio,
        )
        self.mask = ConservedLatentAdditiveResidualHead(
            feature_channels=mask_head_channels,
            num_sources=num_sources,
            num_latent_atoms=int(num_latent_atoms),
            noise_sink=noise_sink,
            atom_residual_scale=float(atom_residual_scale),
            additive_residual_enabled=True,
            additive_residual_scale_init=latent_additive_residual_scale_init,
            additive_residual_scale_max=latent_additive_residual_scale_max,
        )
        if paired_initialization:
            self.mask.copy_baseline_mask_initialization(baseline_mask)
        self.architecture_version = architecture_version
        self.mask_head_type = str(mask_head_type)
        self.paired_initialization = bool(paired_initialization)
        self.max_refinement_steps = int(max_refinement_steps)
        self.num_latent_atoms = int(num_latent_atoms)
        self.responsibility_entropy_weight = float(responsibility_entropy_weight)
        self.responsibility_utilization_weight = float(
            responsibility_utilization_weight
        )
        self.responsibility_utilization_floor = float(
            responsibility_utilization_floor
        )
        self.complex_residual_weight = float(complex_residual_weight)

    def set_aux_loss_config(
        self,
        router_z_loss_weight: Optional[float] = None,
        responsibility_entropy_weight: Optional[float] = None,
        responsibility_utilization_weight: Optional[float] = None,
        responsibility_utilization_floor: Optional[float] = None,
        complex_residual_weight: Optional[float] = None,
    ) -> None:
        """Update internally weighted routing/latent regularizers at runtime."""

        updates = {
            "router_z_loss_weight": router_z_loss_weight,
            "responsibility_entropy_weight": responsibility_entropy_weight,
            "responsibility_utilization_weight": responsibility_utilization_weight,
            "responsibility_utilization_floor": responsibility_utilization_floor,
            "complex_residual_weight": complex_residual_weight,
        }
        for name, value in updates.items():
            if value is None:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, float(value))

    def _get_routing_aux_loss(
        self,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            loss = loss + self.prototype_orthogonality_weight * torch.stack(
                [value.float() for value in prototype_values]
            ).mean()
        loss = loss + self.responsibility_entropy_weight * self._last_aux[
            "responsibility_entropy_loss"
        ].float()
        utilization = self._last_aux["responsibility_utilization"].float()
        utilization_loss = torch.relu(
            utilization.new_tensor(self.responsibility_utilization_floor)
            - utilization
        ).mean()
        loss = loss + self.responsibility_utilization_weight * utilization_loss
        loss = loss + self.complex_residual_weight * self._last_aux[
            "complex_residual_loss"
        ].float()
        return loss

    def get_diagnostics(self) -> Dict[str, object]:
        diagnostics: Dict[str, object] = {
            "forward_available": self._last_aux is not None,
            "architecture_version": self.architecture_version,
            "mask_head_type": self.mask_head_type,
            "num_refinement_steps": self.num_refinement_steps,
            "shared_cell_count": 1,
            "moe_enabled": self.moe_enabled,
            "num_experts": self.num_experts if self.moe_enabled else 1,
            "num_groups": self.mask.group_k,
            "latent_atoms_present": self.num_latent_atoms > 0,
            "num_latent_atoms": self.num_latent_atoms,
            "recurrent_source_memory_present": False,
            "frequency_attention_present": False,
            "dax_msa_present": False,
            "paired_initialization": self.paired_initialization,
        }
        if self._last_aux is None:
            return diagnostics

        def detached_mean(key: str) -> torch.Tensor:
            values = self._moe_step_tensors(key)
            if not values:
                return self._zero().detach()
            return torch.stack([value.detach().float() for value in values]).mean(dim=0)

        global_balance, step_balance, hybrid_balance = self._balance_components()
        steps = self._last_aux.get("moe_steps", [])
        if self.moe_enabled and steps:
            total_tokens = torch.stack(
                [step["num_valid_tokens"].detach().float() for step in steps]
            ).sum()
            total_counts = torch.stack(
                [step["expert_counts"].detach().float() for step in steps]
            ).sum(dim=0)
            total_probability = torch.stack(
                [step["probability_sums"].detach().float() for step in steps]
            ).sum(dim=0)
            expert_load = total_counts / total_tokens.clamp_min(1.0)
            expert_probability = total_probability / total_tokens.clamp_min(1.0)
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
                "evidence_weights": detached_mean("evidence_weights"),
                "router_temperature": detached_mean("router_temperature"),
                "prototype_orthogonality_loss": detached_mean(
                    "prototype_orthogonality_loss"
                ),
                "refinement_delta_rms": detached_mean("refinement_delta_rms"),
                "delta_evidence_rms": detached_mean("delta_evidence_rms"),
                "mean_correction_strength": detached_mean(
                    "mean_correction_strength"
                ),
                "max_correction_strength": detached_mean(
                    "max_correction_strength"
                ),
                "safr_blend": detached_mean("safr_blend"),
                "safr_gate_mean": detached_mean("safr_gate_mean"),
                "responsibility_utilization": self._last_aux[
                    "responsibility_utilization"
                ].detach(),
                "responsibility_entropy": self._last_aux[
                    "responsibility_entropy_loss"
                ].detach(),
                "residual_to_base_rms_ratio": self._last_aux[
                    "residual_to_base_rms_ratio"
                ].detach(),
                "peak_group_residual_to_local_reference_ratio": self._last_aux[
                    "peak_group_residual_to_local_reference_ratio"
                ].detach(),
                "csr_residual_scale": self._last_aux[
                    "csr_residual_scale"
                ].detach(),
                "mixture_consistency_mse": self._last_aux[
                    "mixture_consistency_mse"
                ].detach(),
                "mixture_consistency_max_error": self._last_aux[
                    "mixture_consistency_max_error"
                ].detach(),
                "observation_scale": self._last_aux["observation_scale"].detach(),
                "observation_gate_abs_mean": self._last_aux[
                    "observation_gate_abs_mean"
                ].detach(),
                "observation_injection_rms": self._last_aux[
                    "observation_injection_rms"
                ].detach(),
                "routing_aux_loss": self._get_routing_aux_loss().detach(),
            }
        )
        if "atom_utilization" in self._last_aux:
            diagnostics.update(
                {
                    "atom_utilization": self._last_aux[
                        "atom_utilization"
                    ].detach(),
                    "atom_assignment_entropy": self._last_aux[
                        "assignment_entropy_loss"
                    ].detach(),
                    "latent_additive_residual_scale": self._last_aux[
                        "latent_additive_residual_scale"
                    ].detach(),
                    "atom_group_profile": self._last_aux[
                        "atom_group_profile"
                    ].detach(),
                    "effective_atom_count": self._last_aux[
                        "effective_atom_count"
                    ].detach(),
                    "atom_role_margin": self._last_aux[
                        "atom_role_margin"
                    ].detach(),
                    "multiplicative_complex_residual_loss": self._last_aux[
                        "multiplicative_complex_residual_loss"
                    ].detach(),
                    "additive_complex_residual_loss": self._last_aux[
                        "additive_complex_residual_loss"
                    ].detach(),
                }
            )
        diagnostics["refinement"] = {
            f"step_{index + 1}": {
                key: value.detach()
                for key, value in step.items()
                if torch.is_tensor(value) and value.numel() <= max(self.num_experts, 5)
            }
            for index, step in enumerate(steps)
        }
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
            raise TypeError("Input mixture must be floating point")
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
        complex_spec = self._stft_with_lengths(x, sample_lengths, stft_kwargs)
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
            feature, moe_steps = self.separator(
                feature,
                valid_time_mask=valid_time_mask,
            )
            mask_feature_erb = self.decoder(feature, encoder_outputs)
            mask_feature = self.erb.bs(mask_feature_erb)
            if valid_time_mask is not None:
                mask_feature = mask_feature * valid_time_mask[:, None, :, None].to(
                    mask_feature.dtype
                )
            if mask_feature.shape[2:] != mix_spec.shape[2:]:
                raise RuntimeError("Decoder/fullband TF shape does not match mixture")
            mask_feature, observation_aux = self.observation(
                mask_feature,
                mix_spec,
                valid_time_mask,
            )
            _, head_aux = self.mask(
                mask_feature,
                mix_spec,
                valid_time_mask=valid_time_mask,
            )
            grouped_specs = head_aux["grouped_specs"]
            grouped_waveforms = self._istft_grouped_specs(
                grouped_specs,
                sample_lengths,
                valid_time_mask,
                n_samples=n_samples,
                stft_kwargs=stft_kwargs,
            )
            head_aux.update(observation_aux)
            head_aux["moe_steps"] = moe_steps
            head_aux["grouped_waveforms"] = grouped_waveforms
            head_aux["speech_specs"] = grouped_specs[:, : self.num_sources]
            head_aux["speech_waveforms"] = grouped_waveforms[:, : self.num_sources]
            head_aux["sink_spec"] = grouped_specs[:, self.num_sources :]
            head_aux["sink_waveform"] = grouped_waveforms[:, self.num_sources :]
            head_aux["valid_time_mask"] = valid_time_mask
            head_aux["lengths"] = sample_lengths
            self._last_aux = head_aux
            output = grouped_waveforms[:, : self.num_sources]
        finally:
            self._clear_gln_mask()
        if output.shape != (batch_size, self.num_sources, n_samples):
            raise RuntimeError(
                f"Unexpected output shape {tuple(output.shape)}; expected "
                f"{(batch_size, self.num_sources, n_samples)}"
            )
        return output


class SEAL(SEALCore):
    """SEAL: :class:`SEALCore` with a norm-capped router step cue.

    The refinement-aware router forms
    ``normalize(acoustic_evidence + step_embedding[s])``. SEAL caps the L2 norm
    of each router step-embedding row at ``step_embedding_max_norm`` so the
    step identity can break ties between experts without overriding a
    confident acoustic route. The cap adds no parameters, preserves direction,
    and is the identity below the budget. ``0.15`` is the paper operating
    point, not a claim of global optimality; ``0.0`` disables the cap.
    """

    def __init__(
        self,
        *args,
        step_embedding_max_norm: float = 0.15,
        **kwargs,
    ):
        requested_version = kwargs.pop("architecture_version", None) or "seal_v1"
        if requested_version != "seal_v1":
            raise ValueError("architecture_version must be 'seal_v1'")
        kwargs["architecture_version"] = "seal_core_v1"
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
        # owns a separate one that feeds SAFR, not the expert choice.
        for name, module in self.named_modules():
            if not name.endswith("temporal_readout.router"):
                continue
            source = getattr(module, "step_embedding", None)
            if not isinstance(source, nn.Embedding):
                raise RuntimeError(
                    f"{name}.step_embedding is {type(source).__name__}, "
                    "expected nn.Embedding"
                )
            module.step_embedding = NormClippedStepEmbedding(
                source, self.step_embedding_max_norm
            )
            self.bounded_routers.append(name)

        if not self.bounded_routers:
            raise RuntimeError(
                "no temporal_readout.router found; the step-cue cap requires "
                "moe_enabled=True"
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

    def get_diagnostics(self) -> Dict[str, object]:
        diagnostics = super().get_diagnostics()
        # Nested numeric fields are flattened by the trainer and written every
        # epoch.  This shows whether the cap actually bound the learned
        # embedding instead of relying on the configured threshold alone.
        diagnostics["step_bound"] = self._step_bound_tensor_diagnostics()
        return diagnostics
