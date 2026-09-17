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

import math
import warnings
from itertools import permutations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


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
