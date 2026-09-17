"""Evaluation-only metrics for permutation-invariant speech separation.

This module deliberately does not contain any training losses.  It exposes
both the historical metrics used by this repository and the definitions used
by TIGER's public ``MetricsTracker``:

* TIGER SI-SDR uses utterance-level PIT with ``zero_mean=False``;
* TIGER BSS-SDR uses :func:`fast_bss_eval.sdr_pit_loss`, whose PIT assignment
  is independent of the SI-SDR assignment.
* PESQ/STOI use the primary no-zero-mean SI-SDR assignment and then average
  over the two sources, so they cannot silently choose an easier permutation.

The distinction matters when comparing fixed checkpoints, but it must not
silently change the objective used to train those checkpoints.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, Tuple

import torch


DEFAULT_EPS = 1e-8
DEFAULT_BSS_FILTER_LENGTH = 512
DEFAULT_PESQ_MODE = "wb"


def _validate_source_tensors(estimates: torch.Tensor, targets: torch.Tensor) -> None:
    if estimates.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            "PIT metrics expect estimates and targets with shape (B, K, L), "
            f"got {tuple(estimates.shape)} and {tuple(targets.shape)}"
        )
    if estimates.shape != targets.shape:
        raise ValueError(
            "PIT metrics require matching estimate/target shapes, "
            f"got {tuple(estimates.shape)} and {tuple(targets.shape)}"
        )
    if estimates.shape[1] < 1 or estimates.shape[2] < 1:
        raise ValueError("PIT metrics require at least one source and one sample")


def si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    zero_mean: bool = False,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Return single-source SI-SDR over the final (time) dimension.

    ``zero_mean=False`` matches TIGER's public test metric.  Passing
    ``zero_mean=True`` reproduces this repository's historical ``compute_si_snr``
    helper and remains useful for comparing against already recorded runs.
    """

    if estimate.shape != target.shape:
        raise ValueError(
            f"SI-SDR requires matching shapes, got {estimate.shape} and {target.shape}"
        )
    if estimate.ndim < 1 or estimate.shape[-1] < 1:
        raise ValueError("SI-SDR requires a non-empty time dimension")
    if zero_mean:
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = target.square().sum(dim=-1, keepdim=True) + eps
    projection = dot * target / target_energy
    error = estimate - projection
    ratio = projection.square().sum(dim=-1) / (
        error.square().sum(dim=-1) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


def snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Return plain SNR, retained for backward-compatible diagnostics.

    This is not the filter-invariant BSS-SDR reported by TIGER and is therefore
    named ``SNR``/``SNRi`` everywhere new code emits it.
    """

    if estimate.shape != target.shape:
        raise ValueError(
            f"SNR requires matching shapes, got {estimate.shape} and {target.shape}"
        )
    error = estimate - target
    ratio = target.square().sum(dim=-1) / (error.square().sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def pairwise_si_sdr(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    *,
    zero_mean: bool = False,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Return all estimate/target SI-SDR pairs with shape ``(B, K, K)``.

    Axis 1 indexes estimates and axis 2 indexes targets.  The equations match
    TIGER's ``PairwiseNegSDR('sisdr', zero_mean=False)`` apart from returning a
    positive score instead of a negative loss.
    """

    _validate_source_tensors(estimates, targets)
    if zero_mean:
        estimates = estimates - estimates.mean(dim=-1, keepdim=True)
        targets = targets - targets.mean(dim=-1, keepdim=True)

    expanded_estimates = estimates.unsqueeze(2)
    expanded_targets = targets.unsqueeze(1)
    dot = (expanded_estimates * expanded_targets).sum(dim=-1, keepdim=True)
    target_energy = expanded_targets.square().sum(dim=-1, keepdim=True) + eps
    projection = dot * expanded_targets / target_energy
    error = expanded_estimates - projection
    ratio = projection.square().sum(dim=-1) / (
        error.square().sum(dim=-1) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


def pit_si_sdr(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    *,
    zero_mean: bool = False,
    eps: float = DEFAULT_EPS,
    return_aligned: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute utterance-level PIT SI-SDR, averaging over sources.

    When ``return_aligned`` is true, the result is ``(scores, aligned,
    permutations)``.  Each permutation maps target index to the selected
    estimate index.  PIT is solved independently for every batch item.
    """

    pair_scores = pairwise_si_sdr(
        estimates,
        targets,
        zero_mean=zero_mean,
        eps=eps,
    )
    batch_size, num_sources, _ = pair_scores.shape
    permutation_table = torch.tensor(
        list(permutations(range(num_sources))),
        dtype=torch.long,
        device=estimates.device,
    )
    target_indices = torch.arange(num_sources, device=estimates.device)
    candidate_scores = torch.stack(
        [
            pair_scores[:, permutation, target_indices].mean(dim=-1)
            for permutation in permutation_table
        ],
        dim=1,
    )
    best_scores, best_indices = candidate_scores.max(dim=1)
    if not return_aligned:
        return best_scores

    best_permutations = permutation_table.index_select(0, best_indices)
    gather_indices = best_permutations.unsqueeze(-1).expand(
        batch_size,
        num_sources,
        estimates.shape[-1],
    )
    aligned = torch.gather(estimates, dim=1, index=gather_indices)
    return best_scores, aligned, best_permutations


def pit_bss_sdr(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    *,
    filter_length: int = DEFAULT_BSS_FILTER_LENGTH,
    zero_mean: bool = False,
    use_cg_iter: int | None = None,
) -> torch.Tensor:
    """Return true PIT BSS-SDR averaged over sources for each batch item.

    ``fast_bss_eval`` performs its own pairwise assignment.  In particular,
    estimates must not first be reordered using the SI-SDR permutation.
    Defaults are written explicitly to document TIGER 0.1.4 parity.
    """

    _validate_source_tensors(estimates, targets)
    try:
        import fast_bss_eval
    except ImportError as exc:  # pragma: no cover - exercised on lean installs
        raise ImportError(
            "BSS-SDR requires fast_bss_eval==0.1.4. Install requirements.txt "
            "or run evaluation with BSS-SDR disabled."
        ) from exc

    losses = fast_bss_eval.sdr_pit_loss(
        estimates,
        targets,
        filter_length=filter_length,
        use_cg_iter=use_cg_iter,
        zero_mean=zero_mean,
    )
    if not torch.is_tensor(losses):
        losses = torch.as_tensor(losses, device=estimates.device)
    return -losses.mean(dim=-1)


def perceptual_metrics(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_rate: int,
    pesq_mode: str = DEFAULT_PESQ_MODE,
    stoi_extended: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return source-averaged PESQ and STOI after TIGER SI-SDR PIT.

    Perceptual metrics do not choose a second, metric-specific permutation.
    The estimates are first aligned with the paper's primary no-zero-mean
    SI-SDR assignment, then PESQ/STOI are averaged over sources for each
    utterance.  The formal M1-StepBound protocol uses 16-kHz wide-band PESQ
    and classical (non-extended) STOI.
    """

    _validate_source_tensors(estimates, targets)
    sample_rate = int(sample_rate)
    pesq_mode = str(pesq_mode).lower()
    if sample_rate not in (8000, 16000):
        raise ValueError("PESQ requires an 8-kHz or 16-kHz sample rate")
    if pesq_mode not in {"nb", "wb"}:
        raise ValueError("PESQ mode must be 'nb' or 'wb'")
    if sample_rate == 8000 and pesq_mode != "nb":
        raise ValueError("8-kHz PESQ supports narrow-band mode only")
    try:
        from pesq import pesq
    except ImportError as exc:  # pragma: no cover - depends on optional wheel
        raise ImportError(
            "PESQ evaluation requires pesq==0.0.4; install requirements.txt"
        ) from exc
    try:
        from pystoi import stoi
    except ImportError as exc:  # pragma: no cover - depends on optional wheel
        raise ImportError(
            "STOI evaluation requires pystoi==0.4.1; install requirements.txt"
        ) from exc

    _, aligned, _ = pit_si_sdr(
        estimates,
        targets,
        zero_mean=False,
        return_aligned=True,
    )
    aligned_np = aligned.detach().float().cpu().numpy()
    targets_np = targets.detach().float().cpu().numpy()
    pesq_rows = []
    stoi_rows = []
    for batch_index in range(aligned_np.shape[0]):
        utterance_pesq = []
        utterance_stoi = []
        for source_index in range(aligned_np.shape[1]):
            reference = targets_np[batch_index, source_index]
            estimate = aligned_np[batch_index, source_index]
            utterance_pesq.append(
                float(pesq(sample_rate, reference, estimate, pesq_mode))
            )
            utterance_stoi.append(
                float(
                    stoi(
                        reference,
                        estimate,
                        sample_rate,
                        extended=bool(stoi_extended),
                    )
                )
            )
        pesq_rows.append(sum(utterance_pesq) / len(utterance_pesq))
        stoi_rows.append(sum(utterance_stoi) / len(utterance_stoi))
    return {
        "pesq": torch.tensor(pesq_rows, device=estimates.device),
        "stoi": torch.tensor(stoi_rows, device=estimates.device),
    }


def separation_metrics(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    *,
    include_bss_sdr: bool = True,
    bss_filter_length: int = DEFAULT_BSS_FILTER_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Evaluate a batch using parity and explicitly named legacy metrics.

    All returned tensors have shape ``(B,)``.  ``tiger_si_*`` uses its own
    no-zero-mean PIT; ``bss_*`` uses the independent PIT inside
    ``fast_bss_eval``.  Historical plain SNR is evaluated after the historical
    zero-mean-SI-SDR alignment, exactly as the old validation loop did.
    """

    _validate_source_tensors(estimates, targets)
    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0)
    if mixture.ndim != 2:
        raise ValueError(f"Mixture must have shape (B, L), got {tuple(mixture.shape)}")
    if mixture.shape[0] != estimates.shape[0] or mixture.shape[1] != estimates.shape[2]:
        raise ValueError(
            "Mixture batch/time dimensions must match estimates, got "
            f"{tuple(mixture.shape)} versus {tuple(estimates.shape)}"
        )

    mixture_sources = mixture.unsqueeze(1).expand_as(targets)
    tiger_si_sdr = pit_si_sdr(estimates, targets, zero_mean=False)
    tiger_si_sdr_input = pit_si_sdr(mixture_sources, targets, zero_mean=False)
    legacy_si_sdr, legacy_aligned, _ = pit_si_sdr(
        estimates,
        targets,
        zero_mean=True,
        return_aligned=True,
    )
    legacy_si_sdr_input = pit_si_sdr(mixture_sources, targets, zero_mean=True)
    legacy_snr = snr(legacy_aligned, targets).mean(dim=1)
    legacy_snr_input = snr(mixture_sources, targets).mean(dim=1)

    metrics = {
        "tiger_si_sdr": tiger_si_sdr,
        "tiger_si_sdri": tiger_si_sdr - tiger_si_sdr_input,
        "legacy_zero_mean_si_sdr": legacy_si_sdr,
        "legacy_zero_mean_si_sdri": legacy_si_sdr - legacy_si_sdr_input,
        "legacy_snr": legacy_snr,
        "legacy_snri": legacy_snr - legacy_snr_input,
    }
    if include_bss_sdr:
        bss_sdr = pit_bss_sdr(
            estimates,
            targets,
            filter_length=bss_filter_length,
            zero_mean=False,
            use_cg_iter=None,
        )
        bss_sdr_input = pit_bss_sdr(
            mixture_sources,
            targets,
            filter_length=bss_filter_length,
            zero_mean=False,
            use_cg_iter=None,
        )
        metrics["bss_sdr"] = bss_sdr
        metrics["bss_sdri"] = bss_sdr - bss_sdr_input
    return metrics
