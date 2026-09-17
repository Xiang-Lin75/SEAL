"""Architectural invariant tests for the lean M1-Core separator.

Run from the repository root with::

    python tests/test_m1_model.py

Tiny CPU models cover numerical/gradient invariants.  One final smoke test
builds the production EchoSet configuration and uses the same pinned ptflops
call as the TIGER complexity report.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


_DEPENDENCY_ERROR = ""
try:
    import numpy  # noqa: F401 - required by the inherited ERB implementation.
    import torch
    from omegaconf import OmegaConf
except ModuleNotFoundError as error:
    torch = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    _DEPENDENCY_ERROR = str(error)


if torch is not None:
    from models.gtcrn_ss_noncausal_M1_core import (
        BaselineCompatibleSAFR,
        ConservedLatentAdditiveResidualHead,
        ConservedSpectralResidualHead,
        FullBandObservationAdapter,
        GTCRN_SS_NonCausal_M1_Core,
        RefinementAwareDynamicRouter,
    )
    from models.gtcrn_ss_noncausal_M0_shared_recursive_moe_latent import (
        ConservationStructuredLatentAtomHead,
        GTCRN_SS_NonCausal_M0_SharedRecursive_MoE_Latent,
    )


def _iter_tensors(value: Any) -> Iterator["torch.Tensor"]:
    if torch is not None and torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_tensors(child)


def _has_nonzero_finite_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all().item()
        and parameter.grad.abs().sum().item() > 0.0
        for parameter in parameters
    )


@unittest.skipIf(
    torch is None,
    f"M1 runtime dependencies are not installed: {_DEPENDENCY_ERROR}",
)
class TestM1Model(unittest.TestCase):
    """Fast tests that lock M1's implemented graph, not future candidates."""

    INPUT_SAMPLES = 256
    MODEL_KWARGS = {
        "n_fft": 128,
        "hop_len": 64,
        "win_len": 128,
        "num_sources": 2,
        "hidden_channels": 8,
        "freq_downsample_layers": 1,
        "stft_center": True,
        "noise_sink": True,
        "mask_head_channels": 8,
        "moe_enabled": True,
        "num_experts": 2,
        "moe_top_k": 1,
        "moe_expert_width": 8,
        "moe_dropout": 0.0,
        "router_dim": 8,
        "router_temperature_init": 0.7,
    }

    @classmethod
    def _make_model(cls, **overrides):
        kwargs = dict(cls.MODEL_KWARGS)
        kwargs.update(overrides)
        return GTCRN_SS_NonCausal_M1_Core(**kwargs)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cls.model = cls._make_model(num_refinement_steps=4).eval()
        cls.waveform = 0.05 * torch.randn(1, cls.INPUT_SAMPLES)
        cls.router_calls = []
        cls.cell_inputs = []
        cls.fusion_calls = 0

        def capture_router(_module, inputs, output) -> None:
            previous = inputs[2]
            prepared = inputs[3]
            cls.router_calls.append(
                {
                    "shared": inputs[1].detach().clone(),
                    "previous": (
                        None if previous is None else previous.detach().clone()
                    ),
                    "valid": prepared["valid_time_mask"].detach().clone(),
                    "delta": output[4].detach().clone(),
                }
            )

        def capture_cell(_module, inputs, kwargs) -> None:
            cls.cell_inputs.append(
                (
                    inputs[0].detach().clone(),
                    kwargs["fixed_anchor"].detach().clone(),
                )
            )

        def count_fusion(_module, _inputs, _output) -> None:
            cls.fusion_calls += 1

        handles = (
            cls.model.separator.cell.temporal_readout.router.register_forward_hook(
                capture_router
            ),
            cls.model.separator.cell.register_forward_pre_hook(
                capture_cell,
                with_kwargs=True,
            ),
            cls.model.separator.fusion.register_forward_hook(count_fusion),
        )
        try:
            with torch.inference_mode():
                cls.output = cls.model(cls.waveform)
        finally:
            for handle in handles:
                handle.remove()
        cls.aux = cls.model._last_aux
        if cls.aux is None:
            raise AssertionError("M1 forward must retain model._last_aux")

    def test_01_refinement_is_shared_and_deferred_candidates_are_absent(self) -> None:
        one_step = self._make_model(num_refinement_steps=1)
        four_steps = self._make_model(num_refinement_steps=4)
        one_signature = [
            (name, tuple(parameter.shape))
            for name, parameter in one_step.named_parameters()
        ]
        four_signature = [
            (name, tuple(parameter.shape))
            for name, parameter in four_steps.named_parameters()
        ]
        self.assertEqual(one_signature, four_signature)
        self.assertEqual(
            sum(parameter.numel() for parameter in one_step.parameters()),
            sum(parameter.numel() for parameter in four_steps.parameters()),
        )
        diagnostics = self.model.get_m1_diagnostics()
        self.assertEqual(diagnostics["shared_cell_count"], 1)
        self.assertFalse(diagnostics["recurrent_source_memory_present"])
        self.assertFalse(diagnostics["frequency_attention_present"])
        self.assertFalse(diagnostics["dax_msa_present"])
        deferred_names = (
            "route_transition",
            "source_memory",
            "frequency_attention",
            "dax_msa",
        )
        module_names = [name.lower() for name, _ in self.model.named_modules()]
        for deferred_name in deferred_names:
            self.assertFalse(
                any(deferred_name in name for name in module_names),
                f"Deferred module {deferred_name!r} unexpectedly entered M1-Core",
            )

    def test_02_forward_shape_finiteness_and_identity_strength_separation(self) -> None:
        self.assertEqual(tuple(self.output.shape), (1, 2, self.INPUT_SAMPLES))
        self.assertTrue(torch.isfinite(self.output).all().item())
        for tensor in _iter_tensors(self.aux):
            self.assertTrue(torch.isfinite(tensor).all().item())

        readout = self.model.separator.cell.temporal_readout
        router_parameter_ids = {id(parameter) for parameter in readout.router.parameters()}
        strength_parameter_ids = {
            id(parameter) for parameter in readout.strength_mlp.parameters()
        }
        self.assertTrue(router_parameter_ids)
        self.assertTrue(strength_parameter_ids)
        self.assertTrue(router_parameter_ids.isdisjoint(strength_parameter_ids))
        for step in self.aux["moe_steps"]:
            self.assertLessEqual(step["straight_through_forward_error"].item(), 1e-6)
            self.assertGreater(step["mean_correction_strength"].item(), 0.0)
            self.assertLessEqual(
                step["max_correction_strength"].item(),
                readout.strength_max,
            )

    def test_03_grouped_spectra_have_exact_analytic_mixture_closure(self) -> None:
        grouped_specs = self.aux["grouped_specs"]
        responsibilities = self.aux["responsibilities"]
        self.assertEqual(grouped_specs.shape[1:3], (3, 2))
        self.assertEqual(responsibilities.shape[1], 3)
        self.assertTrue((responsibilities >= 0.0).all().item())
        torch.testing.assert_close(
            responsibilities.sum(dim=1),
            torch.ones_like(responsibilities[:, 0]),
            rtol=1e-6,
            atol=1e-6,
        )

        window = torch.hann_window(
            self.MODEL_KWARGS["win_len"],
            dtype=self.waveform.dtype,
        )
        mixture_complex = torch.stft(
            self.waveform,
            n_fft=self.MODEL_KWARGS["n_fft"],
            hop_length=self.MODEL_KWARGS["hop_len"],
            win_length=self.MODEL_KWARGS["win_len"],
            window=window,
            onesided=True,
            center=True,
            return_complex=True,
        )
        mixture_ri = torch.view_as_real(mixture_complex).permute(0, 3, 2, 1)
        torch.testing.assert_close(
            grouped_specs.sum(dim=1),
            mixture_ri,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertLessEqual(self.aux["mixture_consistency_max_error"].item(), 2e-6)
        self.assertIn(
            "peak_group_residual_to_local_reference_ratio",
            self.aux,
        )
        self.assertLessEqual(
            self.aux[
                "peak_group_residual_to_local_reference_ratio"
            ].item(),
            self.aux["latent_additive_residual_scale"].item() + 1e-5,
        )

    def test_04_csr_speech_queries_are_opposites_and_swap_equivariant(self) -> None:
        torch.manual_seed(13)
        head = ConservedSpectralResidualHead(
            feature_channels=8,
            query_dim=8,
        ).eval()
        # Make the residual branch non-zero so equivariance covers both the
        # responsibility base and the shared complex edge, not only init zeros.
        with torch.no_grad():
            head.edge_output.weight.normal_(mean=0.0, std=0.1)
            head.edge_output.bias.normal_(mean=0.0, std=0.1)
        queries = head._queries().detach()
        torch.testing.assert_close(queries[0], -queries[1], rtol=0.0, atol=1e-7)
        torch.testing.assert_close(
            queries.norm(dim=-1),
            torch.ones(3),
            rtol=1e-6,
            atol=1e-6,
        )

        feature = torch.randn(2, 8, 5, 9)
        mixture_spec = torch.randn(2, 2, 5, 9)
        valid = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        with torch.inference_mode():
            _, original = head(feature, mixture_spec, valid_time_mask=valid)
            head.speech_seed.mul_(-1.0)
            _, swapped = head(feature, mixture_spec, valid_time_mask=valid)

        torch.testing.assert_close(
            swapped["responsibilities"][:, 0],
            original["responsibilities"][:, 1],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            swapped["responsibilities"][:, 1],
            original["responsibilities"][:, 0],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            swapped["grouped_specs"][:, 0],
            original["grouped_specs"][:, 1],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            swapped["grouped_specs"][:, 1],
            original["grouped_specs"][:, 0],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            swapped["grouped_specs"][:, 2],
            original["grouped_specs"][:, 2],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            swapped["grouped_specs"].sum(dim=1),
            mixture_spec,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_05_variable_length_batch_is_padding_invariant(self) -> None:
        torch.manual_seed(17)
        model = self._make_model(num_refinement_steps=2).eval()
        valid_samples = 192
        short = 0.05 * torch.randn(1, valid_samples)
        malicious_tail = 0.7 * torch.randn(1, self.INPUT_SAMPLES - valid_samples)
        padded_short = torch.cat([short, malicious_tail], dim=-1)
        full = 0.05 * torch.randn(1, self.INPUT_SAMPLES)
        batch = torch.cat([padded_short, full], dim=0)
        with torch.inference_mode():
            direct = model(short)
            padded = model(
                batch,
                lengths=torch.tensor([valid_samples, self.INPUT_SAMPLES]),
            )
        torch.testing.assert_close(
            padded[:1, :, :valid_samples],
            direct,
            rtol=2e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            padded[0, :, valid_samples:],
            torch.zeros_like(padded[0, :, valid_samples:]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(torch.isfinite(padded).all().item())

    def test_06_radr_uses_zero_first_delta_and_same_stage_later_deltas(self) -> None:
        self.assertEqual(len(self.router_calls), 4)
        first = self.router_calls[0]
        self.assertIsNone(first["previous"])
        self.assertEqual(torch.count_nonzero(first["delta"]).item(), 0)
        self.assertEqual(self.aux["moe_steps"][0]["refinement_delta_rms"].item(), 0.0)

        for step_index, call in enumerate(self.router_calls[1:], start=1):
            self.assertIsNotNone(call["previous"])
            expected = call["shared"] - call["previous"]
            expected = expected * call["valid"][:, None, :, None]
            torch.testing.assert_close(call["delta"], expected, rtol=0.0, atol=0.0)
            self.assertGreater(torch.count_nonzero(call["delta"]).item(), 0)
            self.assertGreater(
                self.aux["moe_steps"][step_index]["refinement_delta_rms"].item(),
                0.0,
            )

    def test_07_safr_is_skipped_on_step_one_and_called_only_afterward(self) -> None:
        self.assertEqual(self.fusion_calls, 3)
        self.assertEqual(len(self.cell_inputs), 4)
        first_input, first_anchor = self.cell_inputs[0]
        torch.testing.assert_close(first_input, first_anchor, rtol=0.0, atol=0.0)
        steps = self.aux["moe_steps"]
        self.assertEqual(steps[0]["safr_applied"].item(), 0.0)
        self.assertEqual(steps[0]["safr_blend"].item(), 0.0)
        self.assertEqual(steps[0]["safr_candidate_rms"].item(), 0.0)
        for step in steps[1:]:
            self.assertEqual(step["safr_applied"].item(), 1.0)
            self.assertGreater(step["safr_blend"].item(), 0.0)
            self.assertLess(step["safr_blend"].item(), 1.0)

    def test_08_fullband_observation_has_no_unconditional_bypass(self) -> None:
        torch.manual_seed(19)
        adapter = FullBandObservationAdapter(feature_channels=8).eval()
        decoder_zero = torch.zeros(2, 8, 5, 9)
        first_spec = torch.randn(2, 2, 5, 9)
        second_spec = torch.randn(2, 2, 5, 9)
        valid = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        with torch.inference_mode():
            first, first_aux = adapter(decoder_zero, first_spec, valid)
            second, second_aux = adapter(decoder_zero, second_spec, valid)
        # The raw observation path is active, but a zero decoder feature gives
        # an exactly zero bias-free gate. Therefore mixture STFT cannot bypass
        # the separator unconditionally.
        self.assertGreater(first_aux["observation_feature_rms"].item(), 0.0)
        self.assertGreater(second_aux["observation_feature_rms"].item(), 0.0)
        torch.testing.assert_close(first, decoder_zero, rtol=0.0, atol=0.0)
        torch.testing.assert_close(second, decoder_zero, rtol=0.0, atol=0.0)
        self.assertEqual(first_aux["observation_gate_abs_max"].item(), 0.0)
        self.assertEqual(first_aux["observation_injection_rms"].item(), 0.0)

    def test_09_backward_reaches_all_m1_roles_without_unused_parameters(self) -> None:
        torch.manual_seed(23)
        model = self._make_model(num_refinement_steps=2).train()
        waveform = (0.05 * torch.randn(1, self.INPUT_SAMPLES)).requires_grad_(True)
        output = model(waveform)
        loss = output.square().mean()
        loss = loss + 0.01 * model._get_balance_loss()
        loss = loss + 0.01 * model._get_routing_aux_loss()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all().item())

        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [], f"M1 left trainable parameters unused: {missing}")
        role_prefixes = (
            "separator.cell.",
            "separator.fusion.",
            "observation.",
            "mask.occupancy_out.",
            "mask.ownership_out.",
            "mask.residual_out.",
            "mask.additive_output.",
        )
        for prefix in role_prefixes:
            parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(prefix)
            ]
            self.assertTrue(parameters, f"No parameters found for role {prefix}")
            self.assertTrue(
                _has_nonzero_finite_gradient(parameters),
                f"Role {prefix} received no non-zero finite gradient",
            )

    def test_10_empty_experts_and_r1_safr_are_ddp_graph_safe(self) -> None:
        torch.manual_seed(29)
        model = self._make_model(
            num_refinement_steps=1,
            max_refinement_steps=1,
            num_experts=4,
        ).train()
        with torch.no_grad():
            model.separator.cell.temporal_readout.router.prototypes.fill_(1.0)
        output = model(0.05 * torch.randn(1, self.INPUT_SAMPLES))
        counts = model._last_aux["moe_steps"][0]["expert_counts"]
        self.assertGreater(counts[0].item(), 0.0)
        self.assertTrue((counts[1:] == 0.0).all().item())
        (output.square().mean() + 0.01 * model._get_balance_loss()).backward()

        for expert_index, expert in enumerate(
            model.separator.cell.temporal_readout.experts
        ):
            for name, parameter in expert.named_parameters():
                self.assertIsNotNone(
                    parameter.grad,
                    f"empty expert {expert_index}.{name} was absent from autograd",
                )
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for name, parameter in model.separator.fusion.named_parameters():
            self.assertIsNotNone(
                parameter.grad,
                f"R=1 SAFR parameter {name} lacked its zero graph anchor",
            )

    def test_11_diagnostic_alias_and_clear_contract(self) -> None:
        m1 = self.model.get_m1_diagnostics()
        m0_alias = self.model.get_m0_diagnostics()
        self.assertEqual(m1.keys(), m0_alias.keys())
        self.assertEqual(
            m1["architecture_version"],
            "m1_core_radr_latent_additive_v1",
        )
        self.assertEqual(m1["mask_head_type"], "latent_additive")
        self.assertTrue(m1["latent_atoms_present"])
        self.assertEqual(m1["num_latent_atoms"], 6)
        model = self._make_model(num_refinement_steps=1).eval()
        with torch.inference_mode():
            model(self.waveform)
        self.assertIsNotNone(model._last_aux)
        model.clear_m1_aux()
        self.assertIsNone(model._last_aux)
        with torch.inference_mode():
            model(self.waveform)
        model.clear_m0_aux()
        self.assertIsNone(model._last_aux)

    def test_12_production_parameter_and_tiger_mac_budget_smoke(self) -> None:
        config_path = (
            REPOSITORY_ROOT
            / "configs"
            / "m1_core_echoset.yaml"
        )
        config = OmegaConf.load(config_path)
        model = GTCRN_SS_NonCausal_M1_Core(**config["network_config"]).cpu().eval()
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self.assertLess(total, 820_000)
        self.assertEqual(total - trainable, 24_576)

        # Exercise the same public CLI documented for TIGER-compatible
        # complexity measurement, in a clean interpreter.
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "measure_complexity.py"
                ),
                "--config",
                str(config_path),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        parameter_match = re.search(r"parameters:\s*([\d,]+)", completed.stdout)
        mac_match = re.search(r"MAC/s:\s*([\d.]+)\s*G", completed.stdout)
        self.assertIsNotNone(parameter_match)
        self.assertIsNotNone(mac_match)
        measured_parameters = int(parameter_match.group(1).replace(",", ""))
        measured_macs = float(mac_match.group(1)) * 1_000_000_000
        self.assertEqual(measured_parameters, trainable)
        # This is the raw TIGER/ptflops headline. Functional routing and
        # elementwise complex work still require the separate analytic audit.
        self.assertLessEqual(measured_macs, 3_200_000_000)

    def test_13_safr_candidate_is_forward_closed_but_scale_can_learn(self) -> None:
        """Prevent recurrence of SAFR's bilinear zero-times-zero deadlock."""

        torch.manual_seed(37)
        saf_r = BaselineCompatibleSAFR(
            channels=8,
            fusion_dim=8,
            max_refinement_steps=4,
            candidate_scale_init=0.0,
        ).train()
        candidate_weights = saf_r.output_projection.weight[8:]
        self.assertGreater(candidate_weights.detach().abs().sum().item(), 0.0)
        torch.testing.assert_close(
            saf_r.output_projection.weight[:8],
            torch.zeros_like(saf_r.output_projection.weight[:8]),
            rtol=0.0,
            atol=0.0,
        )

        anchor = torch.randn(2, 8, 4, 5)
        previous_state = torch.randn_like(anchor)
        valid = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        altered_candidate = copy.deepcopy(saf_r)
        with torch.no_grad():
            altered_candidate.output_projection.weight[8:].normal_(
                mean=0.0,
                std=1.0,
            )
            altered_candidate.output_projection.bias[8:].normal_(
                mean=0.0,
                std=1.0,
            )

        output, aux = saf_r(
            anchor,
            previous_state,
            step_index=2,
            valid_time_mask=valid,
        )
        with torch.no_grad():
            altered_output, altered_aux = altered_candidate(
                anchor,
                previous_state,
                step_index=2,
                valid_time_mask=valid,
            )
        self.assertEqual(aux["safr_candidate_scale"].item(), 0.0)
        self.assertEqual(altered_aux["safr_candidate_scale"].item(), 0.0)
        self.assertGreater(aux["safr_candidate_rms"].item(), 0.0)
        # Candidate values can differ arbitrarily, but their contribution is
        # analytically closed while tanh(raw_candidate_scale) is exactly zero.
        torch.testing.assert_close(output, altered_output, rtol=0.0, atol=0.0)

        probe = torch.randn_like(output)
        (output * probe).sum().backward()
        scale_gradient = saf_r.raw_candidate_scale.grad
        self.assertIsNotNone(scale_gradient)
        self.assertTrue(torch.isfinite(scale_gradient).all().item())
        # Cell refinement step 2 is SAFR transition slot 1; no permanently
        # unused slot is reserved for the first, fusion-free refinement.
        self.assertEqual(saf_r.num_transition_steps, 3)
        self.assertEqual(saf_r.step_embedding.num_embeddings, 3)
        self.assertGreater(scale_gradient[1].abs().item(), 0.0)
        self.assertEqual(scale_gradient[:1].abs().sum().item(), 0.0)
        self.assertEqual(scale_gradient[2:].abs().sum().item(), 0.0)
        # The candidate projection itself stays closed until its scale moves;
        # only raw_candidate_scale receives the bootstrap gradient initially.
        candidate_gradient = saf_r.output_projection.weight.grad[8:]
        torch.testing.assert_close(
            candidate_gradient,
            torch.zeros_like(candidate_gradient),
            rtol=0.0,
            atol=0.0,
        )

    def test_14_r1_zero_delta_bypasses_affine_normalization(self) -> None:
        """Keep absent R1 progress out of the singular normalized branch."""

        torch.manual_seed(41)
        router = RefinementAwareDynamicRouter(
            channels=8,
            num_experts=2,
            router_dim=8,
            max_refinement_steps=4,
        ).train()
        anchor = torch.randn(2, 5, 4, 8)
        raw_states = torch.randn(2, 5, 4, 16)
        shared_readout = torch.randn(2, 5, 4, 8)
        valid = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        prepared = router.prepare_anchor_context(anchor, valid)
        calls = {"norm": 0, "projection": 0}

        def count_norm(_module, _inputs, _output) -> None:
            calls["norm"] += 1

        def count_projection(_module, _inputs, _output) -> None:
            calls["projection"] += 1

        handles = [
            router.delta_norm.register_forward_hook(count_norm),
            router.delta_projection.register_forward_hook(count_projection),
        ]
        try:
            first = router(
                raw_states,
                shared_readout,
                previous_shared_readout=None,
                prepared_context=prepared,
                step_index=0,
            )
            first_query = first[3]
            first_delta = first[4]
            first_aux = first[5]
            self.assertEqual(calls, {"norm": 0, "projection": 0})
            self.assertEqual(torch.count_nonzero(first_delta).item(), 0)
            self.assertEqual(first_aux["delta_evidence_rms"].item(), 0.0)
            self.assertEqual(first_aux["delta_evidence_max"].item(), 0.0)
            self.assertEqual(
                first_aux["evidence_weights"][router.DELTA_BRANCH_INDEX].item(),
                0.0,
            )
            torch.testing.assert_close(
                first_aux["evidence_weights"].sum(),
                torch.ones((), dtype=first_aux["evidence_weights"].dtype),
                rtol=0.0,
                atol=1e-7,
            )

            probe = torch.randn_like(first_query)
            (first_query * probe).sum().backward()
            for parameter in (
                router.delta_norm.weight,
                router.delta_norm.bias,
                router.delta_projection.weight,
            ):
                self.assertIsNone(
                    parameter.grad,
                    "R1 must not create a gradient through absent delta evidence",
                )

            router.zero_grad(set_to_none=True)
            later = router(
                raw_states,
                shared_readout,
                previous_shared_readout=torch.randn_like(shared_readout),
                prepared_context=prepared,
                step_index=1,
            )
            later_aux = later[5]
            self.assertEqual(calls, {"norm": 1, "projection": 1})
            self.assertGreater(later_aux["delta_evidence_rms"].item(), 0.0)
            self.assertGreater(
                later_aux["evidence_weights"][router.DELTA_BRANCH_INDEX].item(),
                0.0,
            )
            later_aux["delta_evidence_rms"].backward()
            self.assertTrue(
                _has_nonzero_finite_gradient(
                    [
                        router.delta_norm.weight,
                        router.delta_norm.bias,
                        router.delta_projection.weight,
                    ]
                )
            )
        finally:
            for handle in handles:
                handle.remove()

    def test_15_paired_initialization_copies_only_compatible_m0_roles(self) -> None:
        """Same-seed M0/M1 runs must share semantically identical operators."""

        m0_kwargs = dict(self.MODEL_KWARGS)
        m0_kwargs.update(
            {
                "num_refinement_steps": 4,
                "architecture_version": "m0_trr_shar_v1",
                "num_latent_atoms": 6,
                "atom_residual_scale": 0.1,
            }
        )
        torch.manual_seed(47)
        m0 = GTCRN_SS_NonCausal_M0_SharedRecursive_MoE_Latent(**m0_kwargs)
        torch.manual_seed(47)
        m1 = self._make_model(
            num_refinement_steps=4,
            max_refinement_steps=4,
            paired_m0_initialization=True,
        )

        def assert_same_state(left, right, role: str) -> None:
            left_state = left.state_dict()
            right_state = right.state_dict()
            self.assertEqual(left_state.keys(), right_state.keys(), role)
            for key in left_state:
                torch.testing.assert_close(
                    left_state[key],
                    right_state[key],
                    rtol=0.0,
                    atol=0.0,
                    msg=lambda message, key=key: f"{role}.{key}: {message}",
                )

        assert_same_state(m0.encoder, m1.encoder, "encoder")
        assert_same_state(m0.decoder, m1.decoder, "decoder")
        assert_same_state(
            m0.separator.anchor_fuse,
            m1.separator.fusion.baseline_fuse,
            "baseline_fuse",
        )
        for role in (
            "intra_rnn",
            "intra_fc",
            "intra_ln",
            "inter_rnn",
            "inter_fc",
            "inter_ln",
            "attn",
        ):
            assert_same_state(
                getattr(m0.separator.cell, role),
                getattr(m1.separator.cell, role),
                role,
            )
        assert_same_state(
            m0.separator.cell.temporal_readout.experts,
            m1.separator.cell.temporal_readout.experts,
            "experts",
        )
        m0_router = m0.separator.cell.temporal_readout.router
        m1_router = m1.separator.cell.temporal_readout.router
        for role in (
            "local_norm",
            "local_projection",
            "anchor_norm",
            "anchor_projection",
            "trajectory_norm",
            "trajectory_projection",
            "global_norm",
            "global_projection",
        ):
            assert_same_state(
                getattr(m0_router, role),
                getattr(m1_router, role),
                f"router.{role}",
            )
        torch.testing.assert_close(
            m0_router.prototypes,
            m1_router.prototypes,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            m0_router.raw_temperature,
            m1_router.raw_temperature,
            rtol=0.0,
            atol=0.0,
        )
        for role in ("pre", "occupancy_out", "ownership_out", "residual_out"):
            assert_same_state(
                getattr(m0.mask, role),
                getattr(m1.mask, role),
                f"mask.{role}",
            )

    def test_16_m0_latent_start_and_additive_cancellation_escape(self) -> None:
        """Start exactly from M0 speech masks, then learn an additive escape."""

        torch.manual_seed(53)
        m0_head = ConservationStructuredLatentAtomHead(
            feature_channels=8,
            num_sources=2,
            num_latent_atoms=6,
            noise_sink=True,
            atom_residual_scale=0.1,
        ).eval()
        torch.manual_seed(59)
        head = ConservedLatentAdditiveResidualHead(
            feature_channels=8,
            num_sources=2,
            num_latent_atoms=6,
            noise_sink=True,
            atom_residual_scale=0.1,
            additive_residual_scale_init=0.1,
            additive_residual_scale_max=0.5,
        ).train()
        head.copy_m0_mask_initialization(m0_head)
        feature = torch.randn(2, 8, 5, 9, requires_grad=True)
        mixture_spec = torch.randn(2, 2, 5, 9)
        valid = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )

        with torch.no_grad():
            _, m0_aux = m0_head(feature.detach(), mixture_spec, valid)
        speech, initial = head(feature, mixture_spec, valid)
        torch.testing.assert_close(
            speech,
            m0_aux["grouped_specs"][:, :2],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            torch.count_nonzero(initial["additive_complex_residuals"]).item(),
            0,
        )
        torch.testing.assert_close(
            initial["occupancy"].sum(dim=1),
            torch.ones_like(initial["occupancy"][:, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            initial["ownership"].sum(dim=2),
            torch.ones_like(initial["ownership"][:, :, 0]),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            initial["grouped_specs"].sum(dim=1),
            mixture_spec,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertGreaterEqual(initial["effective_atom_count"].item(), 1.0)
        self.assertLessEqual(initial["effective_atom_count"].item(), 6.0 + 1e-5)
        torch.testing.assert_close(
            initial["atom_group_profile"].sum(),
            torch.ones((), dtype=initial["atom_group_profile"].dtype),
            rtol=1e-6,
            atol=1e-6,
        )

        speech.square().mean().backward()
        self.assertTrue(
            _has_nonzero_finite_gradient(
                [head.additive_output.weight, head.additive_output.bias]
            )
        )
        self.assertIsNotNone(head.raw_additive_residual_scale.grad)
        self.assertTrue(
            torch.isfinite(head.raw_additive_residual_scale.grad).all().item()
        )

        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.additive_output.weight.normal_(mean=0.0, std=0.1)
            head.additive_output.bias.normal_(mean=0.0, std=0.1)
            _, corrected = head(feature.detach(), mixture_spec, valid)
        self.assertGreater(
            corrected["additive_complex_residuals"].abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            (
                corrected["grouped_specs"][:, :2]
                - m0_aux["grouped_specs"][:, :2]
            ).abs().sum().item(),
            0.0,
        )
        torch.testing.assert_close(
            corrected["grouped_specs"].sum(dim=1),
            mixture_spec,
            rtol=1e-6,
            atol=1e-6,
        )

        # A softmax may underflow an inactive atom/source prior to exact zero.
        # The sqrt conditioning must keep that boundary's gradient finite.
        zero_base = torch.zeros(2, 3, 5, 9, requires_grad=True)
        probe_feature = torch.randn(2, 8, 5, 9, requires_grad=True)
        probe_grouped, _ = head._additive_group_residuals(
            probe_feature,
            mixture_spec,
            zero_base,
            valid,
        )
        probe_grouped.square().mean().backward()
        self.assertIsNotNone(zero_base.grad)
        self.assertTrue(torch.isfinite(zero_base.grad).all().item())
        self.assertIsNotNone(probe_feature.grad)
        self.assertTrue(torch.isfinite(probe_feature.grad).all().item())

    def test_17_additive_branch_escapes_exact_tf_bin_cancellation(self) -> None:
        """A local additive residual can recover speech where ``M * X`` cannot."""

        torch.manual_seed(61)
        m0_head = ConservationStructuredLatentAtomHead(
            feature_channels=8,
            num_sources=2,
            num_latent_atoms=6,
            noise_sink=True,
            atom_residual_scale=0.1,
        ).eval()
        head = ConservedLatentAdditiveResidualHead(
            feature_channels=8,
            num_sources=2,
            num_latent_atoms=6,
            noise_sink=True,
            atom_residual_scale=0.1,
            additive_residual_scale_init=0.1,
            additive_residual_scale_max=0.5,
        ).train()
        head.copy_m0_mask_initialization(m0_head)

        # The centre mixture TF bin is an exact cancellation (X[t, f] = 0),
        # while a bin inside its 3x3 neighbourhood carries energy.  Every M0
        # output remains M_k * X and must therefore be exactly zero at centre.
        centre = (1, 2)
        mixture_spec = torch.zeros(1, 2, 3, 5)
        mixture_spec[0, 0, 1, 1] = 3.0
        mixture_spec[0, 1, 1, 1] = 4.0
        feature = torch.zeros(1, 8, 3, 5, requires_grad=True)
        valid = torch.ones(1, 3, dtype=torch.bool)

        with torch.no_grad():
            m0_speech, m0_aux = m0_head(feature.detach(), mixture_spec, valid)
            initial_speech, initial_aux = head(
                feature.detach(),
                mixture_spec,
                valid,
            )
        self.assertEqual(tuple(initial_speech.shape), (1, 2, 2, 3, 5))
        self.assertEqual(
            torch.count_nonzero(
                m0_aux["grouped_specs"][0, :, :, centre[0], centre[1]]
            ).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(
                m0_speech[0, :, :, centre[0], centre[1]]
            ).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(
                initial_speech[0, :, :, centre[0], centre[1]]
            ).item(),
            0,
        )
        self.assertGreater(
            initial_aux["local_residual_reference"][
                0,
                centre[0],
                centre[1],
            ].item(),
            0.0,
        )

        # Set a single known real correction logit.  This is deterministic and
        # leaves the second speech correction at zero; the sink is constructed
        # as its exact negative by the production head.
        with torch.no_grad():
            head.additive_output.weight.zero_()
            head.additive_output.bias.zero_()
            head.additive_output.bias[0] = 1.0
        corrected_speech, corrected_aux = head(feature, mixture_spec, valid)
        centre_speech = corrected_speech[
            0,
            :,
            :,
            centre[0],
            centre[1],
        ]
        centre_additive = corrected_aux["additive_grouped_residuals"][
            0,
            :,
            :,
            centre[0],
            centre[1],
        ]
        self.assertGreater(centre_speech[0].abs().sum().item(), 0.0)
        self.assertEqual(torch.count_nonzero(centre_speech[1]).item(), 0)
        torch.testing.assert_close(
            centre_additive.sum(dim=0),
            torch.zeros(2),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            corrected_aux["grouped_specs"][
                0,
                :,
                :,
                centre[0],
                centre[1],
            ].sum(dim=0),
            mixture_spec[0, :, centre[0], centre[1]],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            corrected_aux["grouped_specs"].sum(dim=1),
            mixture_spec,
            rtol=1e-6,
            atol=1e-6,
        )

        # The controlled cancellation-bin estimate remains part of autograd.
        centre_speech[0, 0].backward()
        self.assertIsNotNone(head.additive_output.bias.grad)
        self.assertTrue(torch.isfinite(head.additive_output.bias.grad).all().item())
        self.assertGreater(head.additive_output.bias.grad[0].abs().item(), 0.0)
        self.assertIsNotNone(feature.grad)
        self.assertTrue(torch.isfinite(feature.grad).all().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
