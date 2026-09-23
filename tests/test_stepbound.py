"""Architectural invariant tests for the public SEAL model.

Run from the repository root with::

    python -m unittest tests.test_stepbound

The mechanism is a norm clip on an existing embedding, which makes two failure
modes easy to ship unnoticed, so both are gated explicitly:

* the clip could be **vacuous** -- never binding, leaving SEALCore unchanged
  forever. ``test_clip_actually_binds_once_the_embedding_grows`` drives the
  embedding to norms measured on a trained SEALCore checkpoint (0.65 .. 1.17)
  and requires the effective norm to be the budget, not the raw value.
* the clip could be **destructive** -- changing direction, or killing the
  gradient the way a saturating ``tanh`` bound would.
  ``test_clip_preserves_direction`` and ``test_gradient_survives_the_clip``
  cover those.

Bit-exactness with SEALCore at construction still applies here:
SEALCore zero-initializes the step embedding, so the clip starts inert.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


_DEPENDENCY_ERROR = ""
try:
    import numpy  # noqa: F401 - required by the inherited ERB implementation.
    import torch
    from omegaconf import OmegaConf
except ModuleNotFoundError as error:  # pragma: no cover - environment guard.
    torch = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    _DEPENDENCY_ERROR = str(error)

if torch is not None:
    from seal.models import SEAL, NormClippedStepEmbedding, SEALCore

CONFIG = "configs/seal_small_echoset.yaml"
SEED = 20260811
# Measured on a trained SEALCore checkpoint (epoch 247), steps 1..4. The clip must bind against these.
TRAINED_NORMS = (1.1666, 1.1634, 0.6543, 0.8147)


def _build(cls, **kwargs):
    torch.manual_seed(SEED)
    model = cls(**kwargs)
    model.eval()
    return model


@unittest.skipIf(torch is None, f"missing dependency: {_DEPENDENCY_ERROR}")
class TestStepBound(unittest.TestCase):
    def setUp(self) -> None:
        self.waveform = torch.randn(2, 16000, generator=torch.Generator().manual_seed(7))

    @staticmethod
    def _router(model):
        return model.separator.cell.temporal_readout.router

    def _load_trained_norms(self, model) -> None:
        """Give the embedding the measured trained magnitudes."""

        weight = self._router(model).step_embedding.weight
        with torch.no_grad():
            direction = torch.randn(
                weight.shape, generator=torch.Generator().manual_seed(11)
            )
            direction = direction / direction.norm(dim=-1, keepdim=True)
            target = torch.tensor(TRAINED_NORMS[: weight.shape[0]])
            weight.copy_(direction * target[:, None])

    def test_matches_core_bit_exactly_at_init(self) -> None:
        core = _build(SEALCore)
        bounded = _build(SEAL)
        self.assertEqual(
            sum(p.numel() for p in core.parameters()),
            sum(p.numel() for p in bounded.parameters()),
            "the clip must cost no parameters",
        )
        lengths = torch.tensor([16000, 9000])
        with torch.no_grad():
            self.assertTrue(torch.equal(core(self.waveform), bounded(self.waveform)))
            self.assertTrue(
                torch.equal(
                    core(self.waveform, lengths=lengths),
                    bounded(self.waveform, lengths=lengths),
                )
            )

    def test_clip_actually_binds_once_the_embedding_grows(self) -> None:
        """The gate against a vacuous bound.

        A clip that never binds would pass every other test in this file while
        leaving the model identical to SEALCore forever.
        """

        budget = 0.15
        model = _build(SEAL, step_embedding_max_norm=budget)
        self._load_trained_norms(model)
        report = model.step_bound_report()
        entry = next(iter(report["step_embedding_norms"].values()))
        self.assertEqual(report["rows_clipped"], len(TRAINED_NORMS))
        self.assertAlmostEqual(report["clip_active_fraction"], 1.0)
        self.assertTrue(report["bound_is_active"])
        for raw, effective in zip(entry["raw"], entry["effective"]):
            self.assertGreater(raw, budget, "the probe did not exceed the budget")
            self.assertAlmostEqual(effective, budget, places=5)

    def test_below_budget_rows_pass_through_untouched(self) -> None:
        budget = 0.15
        model = _build(SEAL, step_embedding_max_norm=budget)
        weight = self._router(model).step_embedding.weight
        with torch.no_grad():
            small = torch.randn(
                weight.shape, generator=torch.Generator().manual_seed(5)
            )
            small = 0.01 * small / small.norm(dim=-1, keepdim=True)
            weight.copy_(small)
            clipped = self._router(model).step_embedding(
                torch.arange(weight.shape[0])
            )
        self.assertTrue(torch.allclose(clipped, small, atol=0, rtol=0))

    def test_clip_preserves_direction(self) -> None:
        """Only the magnitude may change; the routed direction must survive."""

        model = _build(SEAL, step_embedding_max_norm=0.15)
        self._load_trained_norms(model)
        weight = self._router(model).step_embedding.weight
        with torch.no_grad():
            clipped = self._router(model).step_embedding(
                torch.arange(weight.shape[0])
            )
            cosine = torch.nn.functional.cosine_similarity(clipped, weight, dim=-1)
        self.assertLess(float((cosine - 1.0).abs().max()), 1e-5)

    def test_gradient_survives_the_clip(self) -> None:
        """A bound must not kill its own gradient.

        A saturating ``tanh`` can freeze the bounded parameter permanently. A
        norm clip rescales, so the gradient stays finite however far past the
        budget the row sits.
        """

        torch.backends.cudnn.enabled = False
        model = _build(SEAL, step_embedding_max_norm=0.15)
        self._load_trained_norms(model)
        model.train()
        model(self.waveform).square().mean().backward()
        grad = self._router(model).step_embedding.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().max()), 0.0, "the clip froze the branch")
        self.assertTrue(torch.isfinite(grad).all())

    def test_zero_cap_reproduces_core_exactly(self) -> None:
        """``0.0`` disables the cap and must leave the router untouched."""

        core = _build(SEALCore)
        control = _build(SEAL, step_embedding_max_norm=0.0)
        self.assertEqual(control.bounded_routers, [])
        self.assertIsInstance(
            self._router(control).step_embedding, torch.nn.Embedding
        )
        for model in (core, control):
            self._load_trained_norms(model)
        with torch.no_grad():
            self.assertTrue(torch.equal(core(self.waveform), control(self.waveform)))

    def test_bounding_changes_the_output_once_the_embedding_is_trained(self) -> None:
        """With the measured trained norms loaded, the bound must matter."""

        control = _build(SEAL, step_embedding_max_norm=0.0)
        bounded = _build(SEAL, step_embedding_max_norm=0.15)
        for model in (control, bounded):
            self._load_trained_norms(model)
        with torch.no_grad():
            self.assertFalse(
                torch.equal(control(self.waveform), bounded(self.waveform))
            )

    def test_fusion_step_embedding_is_left_alone(self) -> None:
        """Only the expert router is bounded; SAFR's own embedding is separate."""

        model = _build(SEAL)
        self.assertIsInstance(
            model.separator.fusion.step_embedding, torch.nn.Embedding
        )
        self.assertNotIsInstance(
            model.separator.fusion.step_embedding, NormClippedStepEmbedding
        )
        self.assertEqual(len(model.bounded_routers), 1)

    def test_state_dict_round_trip(self) -> None:
        model = _build(SEAL)
        self._load_trained_norms(model)
        with torch.no_grad():
            expected = model(self.waveform)
        clone = _build(SEAL)
        clone.load_state_dict(model.state_dict(), strict=True)
        clone.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, clone(self.waveform)))

    def test_state_dict_is_interchangeable_with_core(self) -> None:
        """The wrapper must not rename anything, so SEALCore checkpoints still load."""

        core = _build(SEALCore)
        bounded = _build(SEAL)
        self.assertEqual(set(core.state_dict()), set(bounded.state_dict()))
        bounded.load_state_dict(core.state_dict(), strict=True)

    def test_rejects_unknown_architecture_version(self) -> None:
        with self.assertRaises(ValueError):
            SEAL(architecture_version="not_a_version")

    def test_production_config_builds(self) -> None:
        path = REPOSITORY_ROOT / CONFIG
        if not path.exists():
            self.skipTest(f"{CONFIG} not present")
        config = OmegaConf.load(path)
        network = OmegaConf.to_container(config.network_config, resolve=True)
        torch.manual_seed(SEED)
        model = SEAL(**network)
        model.eval()
        with torch.no_grad():
            output = model(self.waveform)
        self.assertEqual(output.shape[1], network.get("num_sources", 2))
        self.assertEqual(model.architecture_version, "seal_v1")
        report = model.step_bound_report()
        print(
            f"\nconfig build: max_norm={report['max_norm']}, "
            f"bounded {report['bounded_routers']}, "
            f"total {sum(p.numel() for p in model.parameters())} parameters"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
