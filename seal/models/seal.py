from __future__ import annotations
"""SEAL Model (Standalone version)"""

from typing import Dict, List
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Optional, Tuple, Union
from typing import Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

"""
GTCRN-SS V7 dense DPE **non-causal, single-stage** model (full-attention).

Single-stage variant of
``gtcrn_ss_noncausal_V7_dpgrnn_progressive_car_only_noise_sink_no_gsf``:
the refine encoder + Stage-2 refinement path are removed, so the network is
just  Encoder -> recursive DPGRNN blocks (Stage 1) -> Decoder -> mask -> iSTFT.

``forward`` returns a **single** tensor ``(B, num_sources, L)`` (not a tuple),
which the trainer consumes via its single-output loss branch.

All the non-causal properties of the two-stage model are kept:
  - full (non-causal) temporal self-attention,
  - symmetric (centered) temporal convolutions,
  - global layer norm ``gLN4D`` with optional length masking,
  - bidirectional inter-frame GRU,
  - gated ``DirectionalStateBridge`` for the progressive within-stage memory,
  - bidirectional temporal recurrent attention (BiTRA).

Legacy streaming-only kwargs are accepted but warn-and-ignored.
"""

from itertools import permutations



class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (
            (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        )
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (
                (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) / (bins[i + 1] - bins[i] + 1e-12)
            )
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (
                (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12)
                / (bins[i + 2] - bins[i + 1] + 1e-12)
            )

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x):
        return self.unfold(x).reshape(x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])


class gLN4D(nn.Module):
    """Global (non-causal) layer norm over (C, T, F) with optional length mask.

    If ``self._mask`` (shape ``(B, 1, T, 1)``, 1 = valid frame) is set, mean/var
    are computed only over valid time frames so that zero-padding does not
    distort the statistics. When ``None`` (default), plain global stats over the
    full tensor are used (exact for fixed-length or batch_size==1 inference).
    """

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._mask = None  # optional (B, 1, T, 1) time mask, set externally

    def forward(self, x):
        if self._mask is None:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        else:
            m = self._mask.to(dtype=x.dtype)              # (B, 1, T, 1)
            b, c, t, f = x.shape
            valid_frames = m.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            denom = valid_frames * c * f
            mean = (x * m).sum(dim=(1, 2, 3), keepdim=True) / denom
            var = (((x - mean) ** 2) * m).sum(dim=(1, 2, 3), keepdim=True) / denom
        x_hat = (x - mean) / (var + self.eps).sqrt()
        output = x_hat * self.gain + self.bias
        # A masked normalization must also keep padded frames at zero.
        # Otherwise learned bias/gain can recreate non-zero padding which is
        # then consumed by later non-causal convolutions or recurrent layers.
        if self._mask is not None:
            output = output * m
        return output


class LearnableSigmoid(nn.Module):
    def __init__(self, n_channels, n_freqs):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(n_channels, 1, n_freqs))

    def forward(self, x):
        return torch.sigmoid(self.slope * x)


class TRA(nn.Module):
    """Bidirectional temporal recurrent attention gate (BiTRA)."""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels, 1, batch_first=True, bidirectional=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()
        self._mask = None  # optional (B,T), set by the owning M0 model

    def forward(self, x):
        zt = torch.mean(x.pow(2), dim=-1)              # (B, C, T)
        sequence = zt.transpose(1, 2)
        if self._mask is None:
            out = self.att_gru(sequence)[0]            # (B, T, 2C)
        else:
            mask = self._mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != (x.shape[0], x.shape[2]):
                raise ValueError(
                    f"TRA mask must be {(x.shape[0], x.shape[2])}, got {tuple(mask.shape)}"
                )
            lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.att_gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.shape[2],
            )
        at = self.att_fc(out).transpose(1, 2)          # (B, C, T)
        at = self.att_act(at)
        output = x * at[..., None]
        if self._mask is not None:
            output = output * mask[:, None, :, None].to(dtype=output.dtype)
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.Identity() if is_last else gLN4D(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)
        self.point_conv1 = conv_module(in_channels // 2 * 3, hidden_channels, 1)
        self.point_bn1 = gLN4D(hidden_channels)
        self.point_act = nn.PReLU()
        self.depth_conv = conv_module(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.depth_bn = gLN4D(hidden_channels)
        self.depth_act = nn.PReLU()
        self.point_conv2 = conv_module(hidden_channels, in_channels // 2, 1)
        self.point_bn2 = gLN4D(in_channels // 2)
        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        b, c, g, t, f = x.shape
        return x.view(b, c * g, t, f)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))

        # Non-causal: symmetric (centered) time padding instead of left-only.
        pad_l = self.pad_size // 2
        pad_r = self.pad_size - pad_l
        h1 = nn.functional.pad(h1, [0, 0, pad_l, pad_r])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))

        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)
        return self.shuffle(h1, x2)


class LinearHeadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)

    def forward(self, x):
        return self.conv(x)


class NonCausalAttention(nn.Module):
    """Full (non-causal) multi-head self-attention over the time axis."""

    def __init__(self, emb_dim, n_freqs, n_head=2, approx_qk_dim=128):
        super().__init__()
        self.n_head = n_head
        self.emb_dim = emb_dim
        proj_dim = math.ceil(approx_qk_dim / n_freqs)
        if emb_dim % n_head != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by n_head ({n_head})")

        for ii in range(n_head):
            self.add_module(
                f"attn_conv_Q_{ii}",
                nn.Sequential(nn.Conv2d(emb_dim, proj_dim, 1), nn.PReLU(), gLN4D(proj_dim)),
            )
            self.add_module(
                f"attn_conv_K_{ii}",
                nn.Sequential(nn.Conv2d(emb_dim, proj_dim, 1), nn.PReLU(), gLN4D(proj_dim)),
            )
            self.add_module(
                f"attn_conv_V_{ii}",
                nn.Sequential(nn.Conv2d(emb_dim, emb_dim // n_head, 1), nn.PReLU(), gLN4D(emb_dim // n_head)),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(nn.Conv2d(emb_dim, emb_dim, 1), nn.PReLU(), gLN4D(emb_dim)),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x, valid_time_mask=None):
        b, _, t, f = x.shape
        if valid_time_mask is not None:
            if valid_time_mask.shape != (b, t):
                raise ValueError(
                    f"valid_time_mask must be {(b, t)}, got {tuple(valid_time_mask.shape)}"
                )
            valid_time_mask = valid_time_mask.to(device=x.device, dtype=torch.bool)

        all_q, all_k, all_v = [], [], []
        for ii in range(self.n_head):
            all_q.append(self[f"attn_conv_Q_{ii}"](x))
            all_k.append(self[f"attn_conv_K_{ii}"](x))
            all_v.append(self[f"attn_conv_V_{ii}"](x))

        q = torch.cat(all_q, dim=0)
        k = torch.cat(all_k, dim=0)
        v = torch.cat(all_v, dim=0)

        q = q.transpose(1, 2).flatten(start_dim=2)
        k = k.transpose(1, 2).flatten(start_dim=2)
        v = v.transpose(1, 2)
        old_shape = v.shape
        v = v.flatten(start_dim=2)
        attn_dim = q.shape[-1]

        # Non-causal: full attention, no causal mask.
        attn_mat = torch.matmul(q, k.transpose(1, 2)) / (attn_dim ** 0.5)
        if valid_time_mask is not None:
            # q/k were concatenated head-major: [head0 batch, head1 batch, ...].
            head_mask = valid_time_mask.repeat(self.n_head, 1)
            attn_mat = attn_mat.masked_fill(
                ~head_mask[:, None, :],
                torch.finfo(attn_mat.dtype).min,
            )
        attn_mat = torch.softmax(attn_mat, dim=-1)
        v = torch.matmul(attn_mat, v)

        v = v.reshape(old_shape).transpose(1, 2)
        emb_dim_v = v.shape[1]
        batch = v.view(self.n_head, b, emb_dim_v, t, f).transpose(0, 1).contiguous()
        batch = batch.view(b, self.n_head * emb_dim_v, t, f)
        batch = self["attn_concat_proj"](batch)
        output = batch + x
        if valid_time_mask is not None:
            output = output * valid_time_mask[:, None, :, None].to(dtype=output.dtype)
        return output


class GatedMaskSS_V2(nn.Module):
    def __init__(self, num_sources=2, apply_constraint=True, nfft=512, noise_sink=False):
        super().__init__()
        self.num_sources = num_sources
        self.apply_constraint = apply_constraint
        self.noise_sink = noise_sink
        self.internal_k = num_sources + 1 if noise_sink else num_sources
        self.gate_act = LearnableSigmoid(n_channels=2 * self.internal_k, n_freqs=nfft // 2 + 1)

    def constrain_mask_sum(self, mask_real, mask_imag):
        k = self.internal_k
        real_sum = mask_real.sum(dim=1, keepdim=True)
        imag_sum = mask_imag.sum(dim=1, keepdim=True)
        mask_real = mask_real - (real_sum - 1) / k
        mask_imag = mask_imag - imag_sum / k
        return mask_real, mask_imag

    def forward(self, mask_output, spec):
        b, _, t, f = mask_output.shape
        k = self.internal_k
        mask_output = mask_output.reshape(b, 2, 2, k, t, f)
        mask = mask_output[:, 0]

        gate_logits = mask_output[:, 1].reshape(b, 2 * k, t, f)
        gate = self.gate_act(gate_logits).reshape(b, 2, k, t, f)

        gated_mask = mask * gate
        mask_real = gated_mask[:, 0]
        mask_imag = gated_mask[:, 1]

        if self.apply_constraint:
            mask_real, mask_imag = self.constrain_mask_sum(mask_real, mask_imag)

        outputs = []
        for src_idx in range(self.num_sources):
            m_real = mask_real[:, src_idx]
            m_imag = mask_imag[:, src_idx]
            s_real = spec[:, 0] * m_real - spec[:, 1] * m_imag
            s_imag = spec[:, 1] * m_real + spec[:, 0] * m_imag
            outputs.append(torch.stack([s_real, s_imag], dim=1))
        return torch.stack(outputs, dim=1)


class EncoderV7(nn.Module):
    def __init__(self, hidden_channels=64, freq_downsample_layers=1):
        super().__init__()
        c = hidden_channels
        layers = [
            ConvBlock(3 * 3, c, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=False, is_last=False),
        ]
        if freq_downsample_layers >= 2:
            layers.append(
                ConvBlock(c, c, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False)
            )
        layers.extend(
            [
                GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
                GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
                GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False),
            ]
        )
        self.en_convs = nn.ModuleList(layers)

    def forward(self, x):
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class DecoderSS_V4(nn.Module):
    def __init__(self, num_sources=2, hidden_channels=64, freq_downsample_layers=1, noise_sink=False, skip_channels=None):
        super().__init__()
        c = hidden_channels
        skip_c = skip_channels if skip_channels is not None else c
        self.noise_sink = noise_sink
        internal_k = num_sources + 1 if noise_sink else num_sources
        out_channels = 2 * 2 * internal_k

        de_layers = [
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(2 * 5, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(2 * 2, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(c, c, (3, 3), stride=(1, 1), padding=(2 * 1, 1), dilation=(1, 1), use_deconv=True),
        ]
        if freq_downsample_layers >= 2:
            de_layers.append(
                ConvBlock(c, c, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False)
            )
        de_layers.append(
            LinearHeadConv(c, out_channels, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True)
        )

        self.de_convs = nn.ModuleList(de_layers)
        # No-GSF ablation: remove learned gated skip fusion while keeping the decoder stack unchanged.
        self.skip_fuse = None

    def forward(self, x, en_outs):
        n_layers = len(self.de_convs)
        for i in range(n_layers):
            if self.skip_fuse is not None:
                skip = en_outs[n_layers - 1 - i]
                x = self.skip_fuse[i](x_local=skip, x_global=x)
            x = self.de_convs[i](x)
        return x


class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, 1, batch_first=True, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, 1, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, h=None, lengths=None):
        if h is None:
            if self.bidirectional:
                h = torch.zeros(2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(1, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        if lengths is None:
            y1, h1 = self.rnn1(x1, h1)
            y2, h2 = self.rnn2(x2, h2)
        else:
            lengths = torch.as_tensor(lengths, device=x.device)
            if lengths.ndim != 1 or lengths.numel() != x.shape[0]:
                raise ValueError(
                    f"GRNN lengths must have shape ({x.shape[0]},), got {tuple(lengths.shape)}"
                )
            lengths = lengths.to(dtype=torch.long)
            if bool(((lengths < 1) | (lengths > x.shape[1])).any()):
                raise ValueError(f"GRNN lengths must be within [1, {x.shape[1]}]")

            def run_packed(rnn, values, hidden):
                packed = nn.utils.rnn.pack_padded_sequence(
                    values,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_y, new_hidden = rnn(packed, hidden)
                padded_y, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_y,
                    batch_first=True,
                    total_length=x.shape[1],
                )
                return padded_y, new_hidden

            y1, h1 = run_packed(self.rnn1, x1, h1)
            y2, h2 = run_packed(self.rnn2, x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DirectionalStateBridge(nn.Module):
    """Explicit gated fusion of a bidirectional GRU's two terminal states,
    replacing the ill-posed length-2 MemGRU. Input/output shape: (2, BF, C)."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Linear(2 * channels, 2 * channels)
        self.proj = nn.Linear(2 * channels, 2 * channels)
        self.norm = nn.LayerNorm(2 * channels)

    def forward(self, h):
        if h is None:
            return None
        two, bf, c = h.shape
        joint = h.permute(1, 0, 2).reshape(bf, 2 * c)
        gate = torch.sigmoid(self.gate(joint))
        proposal = torch.tanh(self.proj(joint))
        joint = self.norm(joint + gate * proposal)
        return joint.reshape(bf, 2, c).permute(1, 0, 2).contiguous()


class DPGRNNBlock(nn.Module):
    def __init__(self, input_size, width, hidden_size, n_head=2, approx_qk_dim=128):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True)
        self.inter_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.attn = NonCausalAttention(hidden_size, n_freqs=width, n_head=n_head, approx_qk_dim=approx_qk_dim)
        self.gcfn = nn.Identity()

    def forward(self, x, h=None):
        b, c, t, f = x.shape
        x_tfc = x.permute(0, 2, 3, 1)  # (B, T, F, C)
        intra_in = x_tfc.reshape(b * t, f, c)

        intra_mix, _ = self.intra_rnn(intra_in)
        intra_mix = self.intra_fc(intra_mix)

        intra_x = self.intra_ln(intra_mix.reshape(b, t, self.width, self.hidden_size))
        intra_out = x_tfc + intra_x

        inter_in = intra_out.permute(0, 2, 1, 3).reshape(b * self.width, t, self.hidden_size)
        inter_x, h = self.inter_rnn(inter_in, h=h)
        inter_x = self.inter_fc(inter_x).reshape(b, self.width, t, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        inter_bctf = inter_out.permute(0, 3, 1, 2).contiguous()
        attn_out = self.attn(inter_bctf)
        return self.gcfn(attn_out), h


class GTCRN_SS_NonCausal_V7_DPGRNN_SingleStage_FullAttn_NoiseSink_NoGSF(nn.Module):
    def __init__(
        self,
        n_fft=256,
        hop_len=128,
        win_len=256,
        num_sources=2,
        apply_mask_constraint=True,
        num_dpgrnn=3,
        num_iterations=3,
        hidden_channels=72,
        freq_downsample_layers=1,
        stft_center=True,
        noise_sink=False,
        attention_cache_frames=None,        # legacy streaming kwarg (unused)
        streaming_attention_mode=None,      # legacy streaming kwarg (unused)
    ):
        super().__init__()
        if attention_cache_frames is not None or streaming_attention_mode is not None:
            warnings.warn(
                "attention_cache_frames / streaming_attention_mode are streaming-only "
                "options and have no effect in the non-causal model; ignoring them.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.num_sources = num_sources
        self.hidden_channels = hidden_channels
        self.freq_downsample_layers = freq_downsample_layers
        self.stft_center = stft_center
        self.num_iterations = num_iterations

        if n_fft >= 512:
            erb_subband_1 = 65
            erb_subband_2 = 64
            high_lim = 8000
            fs = 16000
        else:
            erb_subband_1 = 33
            erb_subband_2 = 32
            high_lim = 4000
            fs = 8000

        dp_width = erb_subband_1
        for _ in range(1, freq_downsample_layers):
            dp_width = (dp_width + 1) // 2

        self.erb = ERB(erb_subband_1, erb_subband_2, nfft=n_fft, high_lim=high_lim, fs=fs)
        self.sfe = SFE(3, 1)
        self.encoder = EncoderV7(hidden_channels=hidden_channels, freq_downsample_layers=freq_downsample_layers)

        # --- Single stage ---
        self.blocks = nn.ModuleList([
            DPGRNNBlock(hidden_channels, dp_width, hidden_channels)
            for _ in range(num_dpgrnn)
        ])
        self.mem = DirectionalStateBridge(hidden_channels)
        self.decoder = DecoderSS_V4(num_sources=num_sources, hidden_channels=hidden_channels, freq_downsample_layers=freq_downsample_layers, noise_sink=noise_sink)
        self.mask = GatedMaskSS_V2(num_sources=num_sources, apply_constraint=apply_mask_constraint, nfft=n_fft, noise_sink=noise_sink)

    # ---- length-masked global LayerNorm plumbing -------------------------
    def _iter_gln(self):
        for m in self.modules():
            if isinstance(m, gLN4D):
                yield m

    def _set_gln_mask(self, lengths, total_frames, device):
        lengths = lengths.to(device=device).float()
        frames = torch.floor(lengths / self.hop_len).long() + 1
        frames = frames.clamp(min=1, max=total_frames)
        idx = torch.arange(total_frames, device=device).unsqueeze(0)
        mask = (idx < frames.unsqueeze(1)).view(-1, 1, total_frames, 1)
        for m in self._iter_gln():
            m._mask = mask

    def _clear_gln_mask(self):
        for m in self._iter_gln():
            m._mask = None

    def _run_recursive_stage(self, feat, blocks, mem_bridge, num_iterations):
        mem = None
        for iteration in range(num_iterations):
            for i, block in enumerate(blocks):
                feat, h = block(feat, h=mem)
                is_last = (iteration == num_iterations - 1 and i == len(blocks) - 1)
                if not is_last:
                    mem = mem_bridge(h)
        return feat

    def forward(self, x, lengths=None):
        device = x.device
        n_samples = x.shape[1]

        stft_kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": torch.hann_window(self.win_len, device=device, dtype=x.dtype),
            "onesided": True,
            "center": self.stft_center,
        }

        spec = torch.stft(x, **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        spec = spec.permute(0, 3, 2, 1)

        total_frames = feat.shape[2]
        if lengths is not None:
            self._set_gln_mask(lengths, total_frames, device)
        else:
            self._clear_gln_mask()

        try:
            feat = self.erb.bm(feat)
            feat = self.sfe(feat)
            feat, en_outs = self.encoder(feat)

            feat = self._run_recursive_stage(feat, self.blocks, self.mem, self.num_iterations)

            m_feat = self.decoder(feat, en_outs)
            m = self.erb.bs(m_feat)
            specs_enh = self.mask(m, spec)

            out = []
            for src_idx in range(self.num_sources):
                spec_k = specs_enh[:, src_idx]
                spec_k = spec_k.permute(0, 3, 2, 1)
                spec_k = torch.complex(spec_k[..., 0], spec_k[..., 1])
                out.append(torch.istft(spec_k, length=n_samples, **stft_kwargs))
            out = torch.stack(out, dim=1)
        finally:
            self._clear_gln_mask()

        return out


if __name__ == "__main__":
    model = GTCRN_SS_NonCausal_V7_DPGRNN_SingleStage_FullAttn_NoiseSink_NoGSF(
        num_sources=2,
        hidden_channels=72,
        num_dpgrnn=3,
        num_iterations=3,
        freq_downsample_layers=1,
        n_fft=512,
        hop_len=256,
        win_len=512,
        stft_center=True,
        noise_sink=True,
    ).eval()
    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 10000])
    with torch.no_grad():
        y = model(x)
        ym = model(x, lengths=lengths)
    print("Input:", x.shape)
    print("Output:", y.shape, "| masked:", ym.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params:", n_params)

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

"""M1-Core: role-decoupled shared-recursive non-causal speech separator.

M1 deliberately keeps M0's STFT/ERB/SFE/encoder, grouped bidirectional
frequency/temporal GRNNs, temporal top-1 expert bank, non-causal temporal
attention and no-GSF decoder.  It changes only four code-level bottlenecks:

* baseline-compatible step-aware anchor/state fusion (SAFR);
* refinement-aware dynamic routing with independent correction strength
  (RADR), where the first-step delta is exactly zero and later deltas compare
  the same pre-MoE readout stage;
* decoder-conditioned, reliability-weighted full-band observation injection;
* an M0-compatible six-atom latent mask followed by a zero-initialized
  additive complex residual, preserving M0's optimization path while adding
  an escape from the finite multiplicative-mask cancellation ceiling.

The intentionally deferred candidates -- DAX-MSA, frequency attention,
recurrent source memory, source-memory feedback and route-transition matrices
-- are not present in this file.  This keeps the first M1 experiment
attributable and prevents source identity from leaking into the acoustic MoE.
"""






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
    """Dense ablation with the M1 call signature and delta diagnostics."""

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


class M1SharedDPGRNNCell(nn.Module):
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
            raise ValueError("M1 shared cell requires input_size == hidden_size")
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
            raise ValueError("M1 refinement calls must use h=None")
        if x.ndim != 4:
            raise ValueError("DPGRNN input must have shape (B,C,T,F)")
        b, c, t, f = x.shape
        if c != self.hidden_size or f != self.width:
            raise ValueError("M1 DPGRNN input has unexpected channel/frequency shape")
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


class BaselineCompatibleSAFR(nn.Module):
    """Step-aware anchor/state trust fusion with an explicit M0 safety path."""

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


class M1SharedRecursiveSeparator(nn.Module):
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
        self.fusion = BaselineCompatibleSAFR(
            channels=channels,
            fusion_dim=router_dim,
            max_refinement_steps=max_refinement_steps,
            blend_init=saf_r_blend_init,
            candidate_scale_init=saf_r_candidate_scale_init,
        )
        self.cell = M1SharedDPGRNNCell(
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
            raise ValueError("M1 separator input must have shape (B,C,T,F)")
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
            raise RuntimeError("M1 shared recursive separator produced no state")
        if self.num_refinement_steps == 1:
            state = state + self.fusion._ddp_graph_anchor(state)
        return state, step_aux


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


class ConservedSpectralResidualHead(nn.Module):
    """Static-query responsibilities plus two additive complex residual edges."""

    def __init__(
        self,
        feature_channels: int,
        query_dim: int,
        num_sources: int = 2,
        noise_sink: bool = True,
        residual_scale_init: float = 0.1,
        residual_scale_max: float = 0.5,
        responsibility_temperature_init: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if num_sources != 2 or not noise_sink:
            raise ValueError("M1 CSR currently requires two speech sources plus a sink")
        if not 0.0 <= residual_scale_init < residual_scale_max:
            raise ValueError("CSR residual scale must satisfy 0 <= init < max")
        if responsibility_temperature_init <= 0.0:
            raise ValueError("responsibility temperature must be positive")
        self.num_sources = int(num_sources)
        self.group_k = self.num_sources + 1
        self.feature_channels = int(feature_channels)
        self.query_dim = int(query_dim)
        self.residual_scale_max = float(residual_scale_max)
        self.eps = float(eps)
        self.pre = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.PReLU(),
            gLN4D(feature_channels, eps=1e-6),
        )
        self.output_key = nn.Conv2d(feature_channels, query_dim, 1)
        self.speech_seed = nn.Parameter(torch.empty(query_dim))
        self.noise_seed = nn.Parameter(torch.empty(query_dim))
        nn.init.normal_(self.speech_seed, mean=0.0, std=1.0 / math.sqrt(query_dim))
        nn.init.normal_(self.noise_seed, mean=0.0, std=1.0 / math.sqrt(query_dim))
        self.shared_speech_bias = nn.Parameter(torch.zeros(()))
        self.noise_bias = nn.Parameter(torch.zeros(()))
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.expm1(responsibility_temperature_init)))
        )
        self.edge_conditioner = nn.Linear(query_dim, 2 * feature_channels)
        self.edge_depthwise = nn.Conv2d(
            feature_channels,
            feature_channels,
            3,
            padding=1,
            groups=feature_channels,
        )
        self.edge_output = nn.Conv2d(feature_channels, 2, 1)
        nn.init.zeros_(self.edge_output.weight)
        nn.init.zeros_(self.edge_output.bias)
        ratio = max(residual_scale_init / residual_scale_max, 1e-6)
        self.raw_residual_scale = nn.Parameter(torch.tensor(_logit(ratio)))

    @property
    def responsibility_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature).add(1e-4)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.residual_scale_max * torch.sigmoid(self.raw_residual_scale)

    def _queries(self) -> torch.Tensor:
        return F.normalize(
            torch.stack(
                [self.speech_seed, -self.speech_seed, self.noise_seed],
                dim=0,
            ),
            p=2.0,
            dim=-1,
            eps=1e-6,
        )

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
            raise ValueError("feature and spec TF shapes differ")
        if valid_time_mask is None:
            valid = torch.ones(b, t, device=feature.device, dtype=torch.bool)
        else:
            if valid_time_mask.shape != (b, t):
                raise ValueError("valid_time_mask has an unexpected shape")
            valid = valid_time_mask.to(device=feature.device, dtype=torch.bool)

        hidden = self.pre(feature)
        keys = self.output_key(hidden).float()
        queries = self._queries().float()
        scores = torch.einsum("bdtf,kd->bktf", keys, queries)
        biases = torch.stack(
            [self.shared_speech_bias, self.shared_speech_bias, self.noise_bias]
        ).float()
        scores = scores / self.responsibility_temperature.float()
        positive = F.softplus(scores + biases[None, :, None, None]).add(self.eps)
        responsibilities = positive / positive.sum(dim=1, keepdim=True)

        spec_fp32 = spec.float()
        base_specs = responsibilities.unsqueeze(2) * spec_fp32.unsqueeze(1)
        speech_conditions = torch.stack(
            [queries[0] - queries[2], queries[1] - queries[2]],
            dim=0,
        )
        film = self.edge_conditioner(speech_conditions).float()
        gain, bias = film.chunk(2, dim=-1)
        edge_hidden = hidden.float().unsqueeze(1) * (
            1.0 + 0.1 * torch.tanh(gain)[None, :, :, None, None]
        ) + 0.1 * bias[None, :, :, None, None]
        # The FiLM bias is source-conditioned but not time-conditioned.  Clear
        # it on padded frames before the spatial residual filter so the last
        # valid frame has the same right boundary as single-utterance inference.
        edge_hidden = edge_hidden * valid[:, None, None, :, None].to(
            edge_hidden.dtype
        )
        edge_hidden = edge_hidden.reshape(b * self.num_sources, self.feature_channels, t, f)
        edge_hidden = F.silu(self.edge_depthwise(edge_hidden))
        residual_logits = self.edge_output(edge_hidden).reshape(
            b,
            self.num_sources,
            2,
            t,
            f,
        )
        mix_magnitude = spec_fp32.square().sum(dim=1).add(self.eps).sqrt()
        local_reference = F.avg_pool2d(
            mix_magnitude[:, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[:, 0]
        residuals = (
            torch.tanh(residual_logits.float())
            * self.residual_scale.float()
            * local_reference[:, None, None]
        )
        residuals = residuals * valid[:, None, None, :, None]

        speech_one = base_specs[:, 0] + residuals[:, 0]
        speech_two = base_specs[:, 1] + residuals[:, 1]
        sink = spec_fp32 - speech_one - speech_two
        grouped_specs = torch.stack([speech_one, speech_two, sink], dim=1)

        valid_tf = valid[:, :, None].expand(b, t, f)
        activity = mix_magnitude * valid_tf
        activity_denom = activity.sum().clamp_min(self.eps)
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
        residual_energy = residuals.square().sum(dim=2)
        complex_residual_loss = (
            residual_energy * activity[:, None]
        ).sum() / (activity_denom * self.num_sources)
        base_rms = base_specs[:, : self.num_sources].square().mean().sqrt()
        residual_rms = residuals.square().mean().sqrt()
        reconstruction_error = grouped_specs.sum(dim=1) - spec_fp32
        query_cosine = queries @ queries.transpose(0, 1)
        aux = {
            "grouped_specs": grouped_specs,
            "responsibilities": responsibilities,
            "base_grouped_specs": base_specs,
            "additive_complex_residuals": residuals,
            "responsibility_utilization": responsibility_utilization,
            "responsibility_entropy_loss": responsibility_entropy_loss,
            "complex_residual_loss": complex_residual_loss,
            "residual_to_base_rms_ratio": residual_rms / base_rms.clamp_min(self.eps),
            "csr_residual_scale": self.residual_scale.float(),
            "responsibility_temperature": self.responsibility_temperature.float(),
            "query_speech_cosine": query_cosine[0, 1],
            "query_noise_max_cosine": query_cosine[:2, 2].abs().max(),
            "mixture_consistency_mse": reconstruction_error.square().mean(),
            "mixture_consistency_max_error": reconstruction_error.abs().max(),
        }
        return grouped_specs[:, : self.num_sources], aux


class ConservedLatentAdditiveResidualHead(
    ConservationStructuredLatentAtomHead
):
    """Preserve the M0 latent mask and add a bounded spectral residual.

    The inherited path is exactly M0's ``occupancy x ownership`` complex-mask
    construction.  Its output is therefore a stable, paired M0-compatible
    starting point.  A separate zero-initialized branch predicts two additive
    complex speech corrections.  The sink receives their analytic negative,
    so mixture closure remains exact while the speech estimates are no longer
    restricted to ``M_k * X`` in destructive-cancellation bins.

    The six M0 atoms are intentionally described as latent allocation
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
                "M1 latent-additive head requires two speech sources plus a sink"
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
            # At initialization the complete head is exactly the M0 head.
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

    def copy_m0_mask_initialization(
        self,
        source: ConservationStructuredLatentAtomHead,
    ) -> None:
        """Copy every M0 mask operator with identical shape and semantics."""

        if source.num_sources != self.num_sources:
            raise ValueError("M0/M1 mask source counts differ")
        if source.num_latent_atoms != self.num_latent_atoms:
            raise ValueError("M0/M1 latent atom counts differ")
        if source.noise_sink != self.noise_sink:
            raise ValueError("M0/M1 mask sink settings differ")
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
        # Both terms are dimensionless: M0 regularizes its complex mask
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


class GTCRN_SS_NonCausal_M1_Core(
    GTCRN_SS_NonCausal_M0_SharedRecursive_MoE_Latent
):
    """Complete lean M1 graph with the M0 trainer/public-output contract."""

    @staticmethod
    def _copy_compatible_m0_initialization(
        source_separator: nn.Module,
        target_separator: M1SharedRecursiveSeparator,
        *,
        moe_enabled: bool,
    ) -> None:
        """Pair identical M0/M1 operators without loading a trained checkpoint.

        M1 is constructed after an inherited, randomly initialized M0 graph.
        Copying only operators with identical shape *and* semantics makes a
        same-seed M0/M1 comparison isolate the new routing/fusion/head choices.
        M1-only parameters remain freshly initialized; this is still a
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
        architecture_version: str = "m1_core_radr_latent_additive_v1",
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
        csr_residual_scale_init: float = 0.1,
        csr_residual_scale_max: float = 0.5,
        responsibility_temperature_init: float = 1.0,
        latent_additive_residual_scale_init: float = 0.1,
        latent_additive_residual_scale_max: float = 0.5,
        router_z_loss_weight: float = 1e-3,
        prototype_orthogonality_weight: float = 1e-3,
        responsibility_entropy_weight: float = 0.0,
        responsibility_utilization_weight: float = 0.0,
        responsibility_utilization_floor: float = 0.01,
        complex_residual_weight: float = 0.0,
        paired_m0_initialization: bool = True,
        # M0 latent coordinates retained by the default M0+ head.
        num_latent_atoms: Optional[int] = 6,
        atom_residual_scale: Optional[float] = 0.1,
        assignment_entropy_weight: Optional[float] = None,
        atom_utilization_weight: Optional[float] = None,
        atom_utilization_floor: Optional[float] = None,
        refinement_steps: Optional[int] = None,
        latent_masks: Optional[int] = None,
        attention_cache_frames: Optional[int] = None,
        streaming_attention_mode: Optional[str] = None,
    ):
        if architecture_version != "m1_core_radr_latent_additive_v1":
            raise ValueError(
                "M1 architecture_version must be "
                "'m1_core_radr_latent_additive_v1'"
            )
        allowed_mask_heads = {"latent_additive", "m0_latent_control", "csr"}
        if mask_head_type not in allowed_mask_heads:
            raise ValueError(
                f"mask_head_type must be one of {sorted(allowed_mask_heads)}"
            )
        if refinement_steps is not None:
            num_refinement_steps = int(refinement_steps)
        if latent_masks is not None:
            if num_latent_atoms not in (None, 6, int(latent_masks)):
                raise ValueError("num_latent_atoms and latent_masks disagree")
            num_latent_atoms = int(latent_masks)
        if num_latent_atoms is None:
            num_latent_atoms = 6
        if int(num_latent_atoms) < 1:
            raise ValueError("num_latent_atoms must be positive")
        requested_moe_top_k = int(moe_top_k)
        if not 1 <= requested_moe_top_k <= int(num_experts):
            raise ValueError(
                "moe_top_k must satisfy 1 <= moe_top_k <= num_experts"
            )
        if atom_residual_scale is None:
            atom_residual_scale = 0.1
        if assignment_entropy_weight is not None:
            responsibility_entropy_weight = float(assignment_entropy_weight)
        if atom_utilization_weight is not None:
            responsibility_utilization_weight = float(atom_utilization_weight)
        if atom_utilization_floor is not None:
            responsibility_utilization_floor = float(atom_utilization_floor)
        # Build and validate the tested frontend/helpers, then replace only the
        # M0-specific separator and mask.  The decoder graph is identical in M0
        # and M1, so retaining the inherited instance avoids an unnecessary
        # reinitialization confound.  Superseded modules are no longer
        # registered and therefore do not contribute parameters.
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
            architecture_version="m0_trr_shar_v1",
            moe_enabled=moe_enabled,
            num_experts=num_experts,
            # The inherited M0 separator is only a temporary initialization
            # source and implements Top-1 exclusively.  M1 replaces it below
            # with the requested Top-K readout, so never expose a diagnostic
            # K > 1 to the superseded M0 constructor.
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
            attention_cache_frames=attention_cache_frames,
            streaming_attention_mode=streaming_attention_mode,
        )
        inherited_m0_separator = self.separator
        inherited_m0_mask = self.mask
        if num_sources != 2 or not noise_sink:
            raise ValueError("M1-Core currently requires two speech sources plus sink")
        if not 1 <= num_refinement_steps <= max_refinement_steps:
            raise ValueError("num_refinement_steps must be within max_refinement_steps")

        if n_fft >= 512:
            dp_width = 65
        else:
            dp_width = 33
        for _ in range(1, freq_downsample_layers):
            dp_width = (dp_width + 1) // 2
        self.separator = M1SharedRecursiveSeparator(
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
        if paired_m0_initialization:
            self._copy_compatible_m0_initialization(
                inherited_m0_separator,
                self.separator,
                moe_enabled=moe_enabled,
            )
        self.observation = FullBandObservationAdapter(
            feature_channels=mask_head_channels,
            scale_init=observation_scale_init,
            phase_floor_ratio=observation_phase_floor_ratio,
        )
        if mask_head_type in {"latent_additive", "m0_latent_control"}:
            self.mask = ConservedLatentAdditiveResidualHead(
                feature_channels=mask_head_channels,
                num_sources=num_sources,
                num_latent_atoms=int(num_latent_atoms),
                noise_sink=noise_sink,
                atom_residual_scale=float(atom_residual_scale),
                additive_residual_enabled=(mask_head_type == "latent_additive"),
                additive_residual_scale_init=latent_additive_residual_scale_init,
                additive_residual_scale_max=latent_additive_residual_scale_max,
            )
            if paired_m0_initialization:
                self.mask.copy_m0_mask_initialization(inherited_m0_mask)
        else:
            self.mask = ConservedSpectralResidualHead(
                feature_channels=mask_head_channels,
                query_dim=router_dim,
                num_sources=num_sources,
                noise_sink=noise_sink,
                residual_scale_init=csr_residual_scale_init,
                residual_scale_max=csr_residual_scale_max,
                responsibility_temperature_init=responsibility_temperature_init,
            )
        self.architecture_version = architecture_version
        self.mask_head_type = str(mask_head_type)
        self.paired_m0_initialization = bool(paired_m0_initialization)
        self.max_refinement_steps = int(max_refinement_steps)
        self.num_latent_atoms = (
            int(num_latent_atoms) if mask_head_type != "csr" else 0
        )
        self.latent_masks = self.num_latent_atoms
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
        **compatibility,
    ) -> None:
        aliases = {
            "assignment_entropy_weight": "responsibility_entropy_weight",
            "atom_utilization_weight": "responsibility_utilization_weight",
            "atom_utilization_floor": "responsibility_utilization_floor",
        }
        updates = {
            "router_z_loss_weight": router_z_loss_weight,
            "responsibility_entropy_weight": responsibility_entropy_weight,
            "responsibility_utilization_weight": responsibility_utilization_weight,
            "responsibility_utilization_floor": responsibility_utilization_floor,
            "complex_residual_weight": complex_residual_weight,
        }
        for old_name, new_name in aliases.items():
            if compatibility.get(old_name) is not None:
                updates[new_name] = compatibility[old_name]
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

    def get_m1_diagnostics(self) -> Dict[str, object]:
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
            "paired_m0_initialization": self.paired_m0_initialization,
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

    # The existing trainer discovers this historical method name dynamically.
    def get_m0_diagnostics(self) -> Dict[str, object]:
        return self.get_m1_diagnostics()

    def clear_m1_aux(self) -> None:
        self._last_aux = None

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
            _, csr_aux = self.mask(
                mask_feature,
                mix_spec,
                valid_time_mask=valid_time_mask,
            )
            grouped_specs = csr_aux["grouped_specs"]
            grouped_waveforms = self._istft_grouped_specs(
                grouped_specs,
                sample_lengths,
                valid_time_mask,
                n_samples=n_samples,
                stft_kwargs=stft_kwargs,
            )
            csr_aux.update(observation_aux)
            csr_aux["moe_steps"] = moe_steps
            csr_aux["grouped_waveforms"] = grouped_waveforms
            csr_aux["speech_specs"] = grouped_specs[:, : self.num_sources]
            csr_aux["speech_waveforms"] = grouped_waveforms[:, : self.num_sources]
            csr_aux["sink_spec"] = grouped_specs[:, self.num_sources :]
            csr_aux["sink_waveform"] = grouped_waveforms[:, self.num_sources :]
            csr_aux["valid_time_mask"] = valid_time_mask
            csr_aux["lengths"] = sample_lengths
            self._last_aux = csr_aux
            output = grouped_waveforms[:, : self.num_sources]
        finally:
            self._clear_gln_mask()
        if output.shape != (batch_size, self.num_sources, n_samples):
            raise RuntimeError(
                f"Unexpected M1 output shape {tuple(output.shape)}; expected "
                f"{(batch_size, self.num_sources, n_samples)}"
            )
        return output


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GTCRN_SS_NonCausal_M1_Core(
        n_fft=512,
        hop_len=256,
        win_len=512,
        hidden_channels=72,
        num_refinement_steps=4,
        num_experts=6,
        moe_expert_width=96,
        router_dim=24,
        mask_head_channels=24,
    ).eval()
    mixture = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 10000])
    with torch.no_grad():
        separated = model(mixture, lengths=lengths)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print("output:", tuple(separated.shape))
    print("total parameters:", total)
    print("trainable parameters:", trainable)

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

