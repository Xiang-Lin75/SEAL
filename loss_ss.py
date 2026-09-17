"""
Loss functions for Speech Separation with PIT (Permutation Invariant Training).

Supports:
- PIT SI-SNR Loss (most common for separation)
- PIT L1 Loss
- PIT Hybrid Loss (spectral + time domain)
"""
import weakref
from itertools import permutations
from typing import Tuple, List, Optional

import torch
import torch.nn as nn


def si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    
    Args:
        estimate: (*, L) estimated signal
        target: (*, L) target signal
        eps: Small value for numerical stability
        
    Returns:
        si_snr: (*,) SI-SNR in dB (higher is better)
    """
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # s_target = <s', s> * s / ||s||^2
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot * target / s_target_energy
    
    # e_noise = s' - s_target
    e_noise = estimate - s_target
    
    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_snr_value = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    
    return si_snr_value


class PITLoss(nn.Module):
    """
    Permutation Invariant Training (PIT) Loss wrapper.
    
    Finds the optimal permutation of estimated sources to minimize loss.
    
    Args:
        loss_fn: Base loss function (takes estimate, target, returns scalar per sample)
        num_sources: Number of sources
    """
    def __init__(self, loss_fn: nn.Module, num_sources: int = 2):
        super().__init__()
        self.loss_fn = loss_fn
        self.num_sources = num_sources
        self.perms = list(permutations(range(num_sources)))
    
    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L) estimated signals
            targets: (B, K, L) target signals
            
        Returns:
            loss: Scalar loss value
            best_perms: List of best permutations for each sample
        """
        B, K, L = estimates.shape
        
        # Compute loss for each permutation
        perm_losses = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            # Compute loss per source and average
            loss_per_src = []
            for k in range(K):
                loss_k = self.loss_fn(perm_est[:, k], targets[:, k])  # (B,)
                loss_per_src.append(loss_k)
            perm_loss = torch.stack(loss_per_src, dim=1).mean(dim=1)  # (B,)
            perm_losses.append(perm_loss)
        
        perm_losses = torch.stack(perm_losses, dim=1)  # (B, num_perms)
        
        # Find best permutation for each batch
        min_loss, min_idx = perm_losses.min(dim=1)  # (B,)
        
        best_perms = [self.perms[i] for i in min_idx.tolist()]
        
        return min_loss.mean(), best_perms


class NegSISNRLoss(nn.Module):
    """Negative SI-SNR loss (for minimization)."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, L) or (L,)
            target: (B, L) or (L,)
            
        Returns:
            loss: (B,) or scalar, negative SI-SNR
        """
        return -si_snr(estimate, target, self.eps)


class PITSISNRLoss(nn.Module):
    """
    PIT with SI-SNR loss.
    
    This is the most common loss for speech separation.
    """
    def __init__(self, num_sources: int = 2, eps: float = 1e-8):
        super().__init__()
        self.num_sources = num_sources
        self.eps = eps
        self.perms = list(permutations(range(num_sources)))
    
    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L) estimated signals
            targets: (B, K, L) target signals
            
        Returns:
            loss: Scalar loss value (negative mean SI-SNR)
            best_perms: List of best permutations for each sample
        """
        B, K, L = estimates.shape
        
        # Compute SI-SNR for each permutation (sum over sources)
        perm_si_snrs = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            # Sum SI-SNR over sources
            total_si_snr = torch.zeros(B, device=estimates.device)
            for k in range(K):
                total_si_snr += si_snr(perm_est[:, k], targets[:, k], self.eps)
            perm_si_snrs.append(total_si_snr)
        
        perm_si_snrs = torch.stack(perm_si_snrs, dim=1)  # (B, num_perms)
        
        # Find best permutation (maximum SI-SNR)
        max_si_snr, max_idx = perm_si_snrs.max(dim=1)  # (B,)
        
        best_perms = [self.perms[i] for i in max_idx.tolist()]
        
        # Return negative SI-SNR as loss (for minimization)
        return -max_si_snr.mean(), best_perms


class PITL1Loss(nn.Module):
    """
    PIT with L1 loss (Mean Absolute Error).
    """
    def __init__(self, num_sources: int = 2):
        super().__init__()
        self.num_sources = num_sources
        self.perms = list(permutations(range(num_sources)))
        self.l1 = nn.L1Loss(reduction='none')

    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L)
            targets: (B, K, L)
        """
        B, K, L = estimates.shape
        
        perm_losses = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            # Sum L1 loss over sources
            total_l1 = torch.zeros(B, device=estimates.device)
            for k in range(K):
                # Mean over time dimension
                total_l1 += self.l1(perm_est[:, k], targets[:, k]).mean(dim=-1)
            perm_losses.append(total_l1) # (B,)
        
        perm_losses = torch.stack(perm_losses, dim=1)  # (B, num_perms)
        
        # Find best permutation (minimum Loss)
        min_loss, min_idx = perm_losses.min(dim=1)  # (B,)
        
        best_perms = [self.perms[i] for i in min_idx.tolist()]
        
        return min_loss.mean(), best_perms


class PITHybridLoss(nn.Module):
    """
    PIT with Hybrid Loss (spectral + time domain).
    
    Combines:
    - Compressed spectral loss (real + imag + magnitude)
    - SI-SNR loss
    
    Args:
        num_sources: Number of sources
        n_fft: FFT size
        hop_len: Hop length
        win_len: Window length
        compress_factor: Spectral compression factor
        lambda_spec: Weight for spectral loss
        lambda_sisnr: Weight for SI-SNR loss
    """
    def __init__(
        self,
        num_sources: int = 2,
        n_fft: int = 512,
        hop_len: int = 256,
        win_len: int = 512,
        compress_factor: float = 0.3,
        lambda_spec: float = 1.0,
        lambda_sisnr: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.c = compress_factor
        self.lambda_spec = lambda_spec
        self.lambda_sisnr = lambda_sisnr
        self.eps = eps
        
        self.perms = list(permutations(range(num_sources)))
        self.register_buffer("window", torch.hann_window(win_len))
    
    def _compute_spectral_loss(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute compressed spectral loss."""
        # Ensure window is on same device as input
        window = self.window.to(estimate.device)
        est_stft = torch.stft(
            estimate, self.n_fft, self.hop_len, self.win_len, 
            window, return_complex=True
        )
        tgt_stft = torch.stft(
            target, self.n_fft, self.hop_len, self.win_len,
            window, return_complex=True
        )
        
        est_mag = torch.abs(est_stft).clamp(self.eps)
        tgt_mag = torch.abs(tgt_stft).clamp(self.eps)
        
        # Compress
        est_stft_c = est_stft / est_mag ** (1 - self.c)
        tgt_stft_c = tgt_stft / tgt_mag ** (1 - self.c)
        
        # Losses
        real_loss = (est_stft_c.real - tgt_stft_c.real).pow(2).mean(dim=(-2, -1))
        imag_loss = (est_stft_c.imag - tgt_stft_c.imag).pow(2).mean(dim=(-2, -1))
        mag_loss = (est_mag ** self.c - tgt_mag ** self.c).pow(2).mean(dim=(-2, -1))
        
        return real_loss + imag_loss + mag_loss
    
    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L) estimated signals
            targets: (B, K, L) target signals
            
        Returns:
            loss: Scalar loss value
            best_perms: List of best permutations for each sample
        """
        B, K, L = estimates.shape
        
        # Compute loss for each permutation
        perm_losses = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            total_loss = torch.zeros(B, device=estimates.device)
            
            for k in range(K):
                # Spectral loss
                spec_loss = self._compute_spectral_loss(perm_est[:, k], targets[:, k])
                # SI-SNR loss
                sisnr_loss = -si_snr(perm_est[:, k], targets[:, k], self.eps)
                
                total_loss += self.lambda_spec * spec_loss + self.lambda_sisnr * sisnr_loss
            
            perm_losses.append(total_loss / K)
        
        perm_losses = torch.stack(perm_losses, dim=1)  # (B, num_perms)
        
        # Find best permutation (minimum loss)
        min_loss, min_idx = perm_losses.min(dim=1)  # (B,)
        
        best_perms = [self.perms[i] for i in min_idx.tolist()]
        
        return min_loss.mean(), best_perms


def snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-Noise Ratio (SNR) - used by TIGER for training.
    Unlike SI-SNR, this doesn't do scale normalization.
    
    Args:
        estimate: (*, L) estimated signal
        target: (*, L) target signal
        eps: Small value for numerical stability
        
    Returns:
        snr: (*,) SNR in dB (higher is better)
    """
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    
    # e_noise = estimate - target
    e_noise = estimate - target
    
    # SNR = 10 * log10(||target||^2 / ||e_noise||^2)
    snr_value = 10 * torch.log10(
        (target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    
    return snr_value


class PITSNRLoss(nn.Module):
    """
    PIT with SNR loss - TIGER's default training loss.
    
    SNR is less sensitive to scale changes compared to SI-SNR,
    which may help with stability during training.
    """
    def __init__(self, num_sources: int = 2, eps: float = 1e-8):
        super().__init__()
        self.num_sources = num_sources
        self.eps = eps
        self.perms = list(permutations(range(num_sources)))
    
    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L) estimated signals
            targets: (B, K, L) target signals
            
        Returns:
            loss: Scalar loss value (negative mean SNR)
            best_perms: List of best permutations for each sample
        """
        B, K, L = estimates.shape
        
        # Compute SNR for each permutation (sum over sources)
        perm_snrs = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            # Sum SNR over sources
            total_snr = torch.zeros(B, device=estimates.device)
            for k in range(K):
                total_snr += snr(perm_est[:, k], targets[:, k], self.eps)
            perm_snrs.append(total_snr)
        
        perm_snrs = torch.stack(perm_snrs, dim=1)  # (B, num_perms)
        
        # Find best permutation (maximum SNR)
        max_snr, max_idx = perm_snrs.max(dim=1)  # (B,)
        
        best_perms = [self.perms[i] for i in max_idx.tolist()]
        
        # Return negative SNR as loss (for minimization)
        return -max_snr.mean() / K, best_perms


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for spectral refinement.
    
    Uses multiple FFT sizes to capture both fine-grained and coarse spectral details.
    This helps the model learn better frequency representations.
    
    Args:
        fft_sizes: List of FFT sizes to use
        hop_sizes: List of hop sizes corresponding to FFT sizes
        win_sizes: List of window sizes corresponding to FFT sizes
    """
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [50, 120, 240],
        win_sizes: List[int] = [240, 600, 1200],
        eps: float = 1e-8,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.eps = eps
        
        # Register windows as buffers
        for i, win_size in enumerate(win_sizes):
            self.register_buffer(f"window_{i}", torch.hann_window(win_size))
    
    def _stft_loss(self, estimate: torch.Tensor, target: torch.Tensor, fft_idx: int) -> torch.Tensor:
        """Compute single-resolution STFT loss."""
        n_fft = self.fft_sizes[fft_idx]
        hop_size = self.hop_sizes[fft_idx]
        win_size = self.win_sizes[fft_idx]
        window = getattr(self, f"window_{fft_idx}")
        
        # Move window to same device as input
        if window.device != estimate.device:
            window = window.to(estimate.device)
        
        est_stft = torch.stft(
            estimate, n_fft, hop_size, win_size, window, return_complex=True
        )
        tgt_stft = torch.stft(
            target, n_fft, hop_size, win_size, window, return_complex=True
        )
        
        # Magnitude loss (L1)
        est_mag = torch.abs(est_stft)
        tgt_mag = torch.abs(tgt_stft)
        mag_loss = (est_mag - tgt_mag).abs().mean(dim=(-2, -1))
        
        # Log-Magnitude Loss (L1) - More robust than Spectral Convergence on silence
        log_mag_loss = (torch.log(est_mag + self.eps) - torch.log(tgt_mag + self.eps)).abs().mean(dim=(-2, -1))
        
        return mag_loss + log_mag_loss
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, L) or (B, K, L) estimated signal
            target: (B, L) or (B, K, L) target signal
            
        Returns:
            loss: (B,) or (B, K) multi-resolution STFT loss
        """
        total_loss = 0
        for i in range(len(self.fft_sizes)):
            total_loss = total_loss + self._stft_loss(estimate, target, i)
        return total_loss / len(self.fft_sizes)



class FreqMAEWavL1Loss(nn.Module):
    """
    TIGER's Frequency MAE + Waveform L1 Loss.
    
    Exact implementation from TIGER's Look2Hear/losses/matrix.py.
    Combines:
    - Frequency domain L1 (Real + Imag) 
    - Time domain L1 (Waveform)
    
    Args:
        win: FFT window size (TIGER default: 2048)
        stride: Hop length (TIGER default: 512)
    """
    def __init__(self, win: int = 2048, stride: int = 512):
        super().__init__()
        self.win = win
        self.stride = stride
        self.register_buffer("window", torch.hann_window(win))
    
    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimate: (B, K, L) estimated signals
            target: (B, K, L) target signals
            
        Returns:
            loss: (B,) combined loss
        """
        B, nsrc, L = estimate.shape
        
        # Move window to same device
        window = self.window.to(estimate.device)
        
        # Flatten for STFT: (B*K, L)
        est_flat = estimate.reshape(-1, L)
        tgt_flat = target.reshape(-1, L)
        
        est_spec = torch.stft(
            est_flat, n_fft=self.win, hop_length=self.stride,
            window=window, return_complex=True
        )
        tgt_spec = torch.stft(
            tgt_flat, n_fft=self.win, hop_length=self.stride,
            window=window, return_complex=True
        )
        
        # Frequency L1 (Real + Imag) - Mean over Time and Freq
        freq_L1 = (est_spec.real - tgt_spec.real).abs().mean(dim=(1, 2)) + \
                  (est_spec.imag - tgt_spec.imag).abs().mean(dim=(1, 2))
        freq_L1 = freq_L1.reshape(B, nsrc).mean(dim=-1)
        
        # Waveform L1
        wave_L1 = (estimate - target).abs().mean(dim=-1).mean(dim=-1)
        
        return freq_L1 + wave_L1


class PITTigerLoss(nn.Module):
    """
    Complete TIGER-style PIT Loss.
    
    Combines:
    - SNR Loss (main loss for permutation)
    - Multi-resolution STFT Loss (spectral refinement)
    - Optional Freq MAE + Wave L1 Loss
    
    Args:
        num_sources: Number of sources
        use_snr: Use SNR instead of SI-SNR
        lambda_snr: Weight for SNR/SI-SNR loss
        lambda_stft: Weight for multi-resolution STFT loss
        use_freq_wav_loss: Whether to use Freq MAE + Wave L1 loss
        lambda_freq_wav: Weight for Freq MAE + Wave L1 loss
    """
    def __init__(
        self,
        num_sources: int = 2,
        use_snr: bool = True,
        lambda_snr: float = 1.0,
        lambda_stft: float = 1.0,
        use_freq_wav_loss: bool = False,
        lambda_freq_wav: float = 0.5,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.use_snr = use_snr
        self.lambda_snr = lambda_snr
        self.lambda_stft = lambda_stft
        self.use_freq_wav_loss = use_freq_wav_loss
        self.lambda_freq_wav = lambda_freq_wav
        self.eps = eps
        
        self.perms = list(permutations(range(num_sources)))
        
        # Multi-resolution STFT loss
        stft_kwargs = {}
        if fft_sizes is not None: stft_kwargs['fft_sizes'] = fft_sizes
        if hop_sizes is not None: stft_kwargs['hop_sizes'] = hop_sizes
        if win_sizes is not None: stft_kwargs['win_sizes'] = win_sizes
        
        # Initialize only if needed (fixes potential TypeErrors if stft_loss is disabled)
        if self.lambda_stft > 0:
            self.stft_loss = MultiResolutionSTFTLoss(**stft_kwargs)
        else:
            self.stft_loss = None
        
        # Optional Freq MAE + Wave L1 loss
        if use_freq_wav_loss:
            # Use provided FFT/Hop sizes for TIGER loss if available, else default
            win = 2048
            stride = 512
            if fft_sizes is not None and len(fft_sizes) > 0: win = fft_sizes[-1] # Use largest FFT
            if hop_sizes is not None and len(hop_sizes) > 0: stride = hop_sizes[-1] # Use largest Hop
            self.freq_wav_loss = FreqMAEWavL1Loss(win=win, stride=stride)
    
    def forward(
        self, 
        estimates: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Args:
            estimates: (B, K, L) estimated signals
            targets: (B, K, L) target signals
            
        Returns:
            loss: Scalar loss value
            best_perms: List of best permutations for each sample
        """
        B, K, L = estimates.shape
        sdr_fn = snr if self.use_snr else si_snr
        
        # Step 1: Find best permutation using SNR/SI-SNR
        perm_sdrs = []
        for perm in self.perms:
            perm_est = estimates[:, perm, :]
            total_sdr = torch.zeros(B, device=estimates.device)
            for k in range(K):
                # Ensure targets are sufficiently long
                tgt = targets[:, k]
                est = perm_est[:, k]
                if est.shape[-1] != tgt.shape[-1]:
                   min_len = min(est.shape[-1], tgt.shape[-1])
                   est = est[..., :min_len]
                   tgt = tgt[..., :min_len]
                
                total_sdr += sdr_fn(est, tgt, self.eps)
            perm_sdrs.append(total_sdr)
        
        perm_sdrs = torch.stack(perm_sdrs, dim=1)  # (B, num_perms)
        max_sdr, max_idx = perm_sdrs.max(dim=1)
        
        best_perms = [self.perms[i] for i in max_idx.tolist()]
        
        # Step 2: Reorder estimates according to best permutation
        reordered_est = torch.stack([
            estimates[b, best_perms[b], :] for b in range(B)
        ])
        
        # Step 3: Compute total loss
        # SDR loss
        sdr_loss = -max_sdr.mean() / K
        
        # Multi-resolution STFT loss
        stft_loss = 0
        if self.lambda_stft > 0 and self.stft_loss is not None:
            for k in range(K):
                stft_loss = stft_loss + self.stft_loss(reordered_est[:, k], targets[:, k]).mean()
            stft_loss = stft_loss / K
        
        total_loss = self.lambda_snr * sdr_loss + self.lambda_stft * stft_loss
        
        # Optional Freq MAE + Wave L1 loss
        if self.use_freq_wav_loss:
            freq_wav_loss = self.freq_wav_loss(reordered_est, targets).mean()
            total_loss = total_loss + self.lambda_freq_wav * freq_wav_loss
        
        return total_loss, best_perms


class PITAuxBalanceWrapper(nn.Module):
    """
    Wrap an existing PIT loss and add model-side auxiliary balance loss.

    This is intended for sparse MoE models whose routing statistics live inside
    the model (for example `_get_balance_loss()` on the model object). The
    wrapped base PIT loss remains unchanged; this class only adds:

        total_loss = (
            base_loss
            + balance_loss_weight * model._get_balance_loss()
            + model._get_routing_aux_loss(targets)
        )

    ``_get_balance_loss()`` is an unweighted load-balancing loss, so the
    wrapper applies ``balance_loss_weight`` exactly once. In contrast,
    ``_get_routing_aux_loss()`` must return an already-weighted additive term;
    the wrapper deliberately does not apply another multiplier.

    Usage:
        base_loss = PITSISNRLoss(num_sources=2)
        loss_fn = PITAuxBalanceWrapper(base_loss, balance_loss_weight=0.01)
        loss_fn.attach_model(model)
    """
    def __init__(self, base_loss: nn.Module, balance_loss_weight: float = 0.01):
        super().__init__()
        self.base_loss = base_loss
        self.balance_loss_weight = balance_loss_weight
        # Do not assign the model itself as an attribute: nn.Module would then
        # register it as a child of the loss wrapper, duplicating the complete
        # model in wrapper.parameters()/state_dict(). The trainer owns the model;
        # this wrapper only needs a non-owning lookup for auxiliary losses.
        self._attached_model_ref = None
        self.latest_components = {}

    def attach_model(self, model: nn.Module):
        self._attached_model_ref = weakref.ref(model)

    def _resolve_model(self):
        if self._attached_model_ref is None:
            return None
        attached_model = self._attached_model_ref()
        if attached_model is None:
            return None
        if hasattr(attached_model, "module"):
            return attached_model.module
        return attached_model

    def _get_balance_loss(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        model = self._resolve_model()
        if model is None or not hasattr(model, "_get_balance_loss"):
            return reference_tensor.new_tensor(0.0)

        balance_loss = model._get_balance_loss()
        if not torch.is_tensor(balance_loss):
            balance_loss = reference_tensor.new_tensor(float(balance_loss))
        return balance_loss

    def _get_routing_aux_loss(
        self,
        reference_tensor: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        model = self._resolve_model()
        if model is None or not hasattr(model, "_get_routing_aux_loss"):
            return reference_tensor.new_tensor(0.0)

        routing_aux_loss = model._get_routing_aux_loss(targets)
        if not torch.is_tensor(routing_aux_loss):
            routing_aux_loss = reference_tensor.new_tensor(float(routing_aux_loss))
        return routing_aux_loss

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        base_loss, best_perms = self.base_loss(estimates, targets)
        balance_loss = self._get_balance_loss(base_loss)
        routing_aux_loss = self._get_routing_aux_loss(base_loss, targets)
        weighted_balance_loss = self.balance_loss_weight * balance_loss
        total_loss = base_loss + weighted_balance_loss + routing_aux_loss
        self.latest_components = {
            "base_pit": base_loss.detach(),
            "raw_balance": balance_loss.detach(),
            "weighted_balance": weighted_balance_loss.detach(),
            "weighted_router_latent": routing_aux_loss.detach(),
            "total": total_loss.detach(),
        }
        return total_loss, best_perms


if __name__ == "__main__":
    # Test losses
    print("Testing PIT Loss Functions")
    print("=" * 50)
    
    B, K, L = 4, 2, 16000
    estimates = torch.randn(B, K, L)
    targets = torch.randn(B, K, L)
    
    # Test SI-SNR
    print("\n1. Testing SI-SNR:")
    snr = si_snr(estimates[:, 0], targets[:, 0])
    print(f"   SI-SNR shape: {snr.shape}, values: {snr}")
    
    # Test PIT SI-SNR Loss
    print("\n2. Testing PITSISNRLoss:")
    loss_fn = PITSISNRLoss(num_sources=2)
    loss, perms = loss_fn(estimates, targets)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Best perms: {perms[:2]}...")
    
    # Test gradient
    estimates.requires_grad = True
    loss, _ = loss_fn(estimates, targets)
    loss.backward()
    print(f"   Gradient computed: {estimates.grad is not None}")
    
    # Test Hybrid Loss
    print("\n3. Testing PITHybridLoss:")
    estimates = torch.randn(B, K, L, requires_grad=True)
    hybrid_loss = PITHybridLoss(num_sources=2)
    loss, perms = hybrid_loss(estimates, targets)
    print(f"   Loss: {loss.item():.4f}")
    loss.backward()
    print(f"   Gradient computed: {estimates.grad is not None}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
