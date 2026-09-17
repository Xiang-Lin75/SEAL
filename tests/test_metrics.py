"""Focused regressions for TIGER-parity evaluation metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from metrics_ss import (
    pairwise_si_sdr,
    perceptual_metrics,
    pit_bss_sdr,
    pit_si_sdr,
    separation_metrics,
    si_sdr,
)


class TestEvaluationMetrics(unittest.TestCase):
    def test_no_zero_mean_pairwise_equation_matches_tiger_definition(self):
        torch.manual_seed(17)
        estimates = torch.randn(2, 2, 257) + torch.tensor([0.3, -0.4])[None, :, None]
        targets = torch.randn(2, 2, 257) + torch.tensor([0.1, 0.6])[None, :, None]

        estimate_expanded = estimates.unsqueeze(2)
        target_expanded = targets.unsqueeze(1)
        dot = (estimate_expanded * target_expanded).sum(dim=-1, keepdim=True)
        target_energy = target_expanded.square().sum(dim=-1, keepdim=True) + 1e-8
        projection = dot * target_expanded / target_energy
        error = estimate_expanded - projection
        expected = 10.0 * torch.log10(
            projection.square().sum(dim=-1) / (error.square().sum(dim=-1) + 1e-8)
            + 1e-8
        )

        actual = pairwise_si_sdr(estimates, targets, zero_mean=False)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_mean_is_a_metric_convention_not_a_loss_correction(self):
        time = torch.linspace(0.0, 8.0 * torch.pi, 1024)
        target = torch.sin(time)[None]
        estimate = target + 0.75
        zero_mean_score = si_sdr(estimate, target, zero_mean=True)
        tiger_score = si_sdr(estimate, target, zero_mean=False)
        self.assertGreater((zero_mean_score - tiger_score).item(), 40.0)

    def test_si_sdr_pit_recovers_a_source_swap(self):
        torch.manual_seed(23)
        targets = torch.randn(1, 2, 2048)
        estimates = targets.flip(1) + 0.01 * torch.randn_like(targets)
        score, aligned, permutation = pit_si_sdr(
            estimates,
            targets,
            zero_mean=False,
            return_aligned=True,
        )
        self.assertEqual(permutation.tolist(), [[1, 0]])
        self.assertGreater(score.item(), 35.0)
        self.assertLess((aligned - targets).square().mean().item(), 2e-4)

    def test_bss_sdr_uses_fast_bss_eval_independent_pit(self):
        try:
            import fast_bss_eval
        except ImportError:
            self.skipTest("fast_bss_eval is not installed")
        torch.manual_seed(29)
        targets = torch.randn(1, 2, 1536)
        estimates = targets.flip(1) + 0.02 * torch.randn_like(targets)
        actual = pit_bss_sdr(estimates, targets)
        expected = -fast_bss_eval.sdr_pit_loss(
            estimates,
            targets,
            filter_length=512,
            use_cg_iter=None,
            zero_mean=False,
        ).mean(dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_metric_bundle_keeps_legacy_names_and_adds_true_bss_sdr(self):
        torch.manual_seed(31)
        targets = 0.2 * torch.randn(1, 2, 1024)
        mixture = targets.sum(dim=1)
        estimates = targets.flip(1) + 0.01 * torch.randn_like(targets)
        metrics = separation_metrics(estimates, targets, mixture, include_bss_sdr=True)
        self.assertEqual(
            set(metrics),
            {
                "tiger_si_sdr",
                "tiger_si_sdri",
                "bss_sdr",
                "bss_sdri",
                "legacy_zero_mean_si_sdr",
                "legacy_zero_mean_si_sdri",
                "legacy_snr",
                "legacy_snri",
            },
        )
        for value in metrics.values():
            self.assertEqual(tuple(value.shape), (1,))
            self.assertTrue(torch.isfinite(value).all().item())

    def test_perceptual_metrics_use_primary_si_sdr_pit_assignment(self):
        fake_pesq = ModuleType("pesq")
        fake_pystoi = ModuleType("pystoi")

        def exact_match_score(_sample_rate, reference, estimate, _mode):
            return 4.5 if abs(reference - estimate).max() < 1e-7 else 1.0

        def exact_match_stoi(reference, estimate, _sample_rate, *, extended):
            self.assertFalse(extended)
            return 1.0 if abs(reference - estimate).max() < 1e-7 else 0.0

        fake_pesq.pesq = exact_match_score
        fake_pystoi.stoi = exact_match_stoi
        torch.manual_seed(37)
        targets = torch.randn(1, 2, 2048)
        estimates = targets.flip(1)
        with patch.dict(
            sys.modules,
            {"pesq": fake_pesq, "pystoi": fake_pystoi},
        ):
            metrics = perceptual_metrics(
                estimates,
                targets,
                sample_rate=16000,
            )
        self.assertAlmostEqual(float(metrics["pesq"]), 4.5)
        self.assertAlmostEqual(float(metrics["stoi"]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
