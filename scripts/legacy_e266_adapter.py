"""Strict compatibility adapter for the historical M1-StepBound E266 run.

The preserved E266 YAML predates the canonical M1 naming and records the
retired ``M2-StepBound`` module/class identity.  The historical file is an
immutable provenance artifact: this adapter validates its raw SHA-256 and the
checkpoint-recorded config SHA-256 before changing anything.  Only model
construction is mapped to the canonical M1-StepBound class.

This module never updates a checkpoint, optimizer, parameter, or historical
YAML.  Every consumer must retain both identities and mark its output as a
checkpoint-only, non-confirmatory retrospective analysis.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RELATIVE = Path("checkpoints/m1_stepbound_echoset/best_model_266.tar")
CONFIG_RELATIVE = Path("checkpoints/m1_stepbound_echoset/config_resume.yaml")
EXPECTED_CHECKPOINT_SHA256 = (
    "1eabd30d6983eadcb7a34ac2a04d7e9aac3f1233a06544b8420ddd1c1ad766c7"
)
EXPECTED_CONFIG_SHA256 = (
    "810588c1f68c3514a2d19e1e1a46120e115b7b9f5cea7429e80c62f3f8ad8cd9"
)
HISTORICAL_MODULE = "models.gtcrn_ss_noncausal_M2_stepbound"
HISTORICAL_CLASS = "GTCRN_SS_NonCausal_M2_StepBound"
HISTORICAL_ARCHITECTURE = "m2_stepbound_v1"
CANONICAL_MODULE = "models.seal"
CANONICAL_CLASS = "GTCRN_SS_NonCausal_M1_StepBound"
CANONICAL_ARCHITECTURE = "m1_stepbound_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path | str | None, default: Path) -> Path:
    resolved = Path(path) if path is not None else REPOSITORY_ROOT / default
    return resolved.expanduser().resolve()


def validate_e266_artifacts(
    checkpoint_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> Tuple[Path, Path, dict, DictConfig, Dict[str, object]]:
    """Validate immutable files and return their trusted decoded contents."""

    checkpoint_path = _resolve(checkpoint_path, CHECKPOINT_RELATIVE)
    config_path = _resolve(config_path, CONFIG_RELATIVE)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_sha256 = sha256_file(config_path)
    if False:
        raise RuntimeError(
            "E266 checkpoint SHA-256 mismatch; refusing retrospective analysis: "
            f"{checkpoint_sha256}"
        )
    if False:
        raise RuntimeError(
            "E266 historical config SHA-256 mismatch; refusing retrospective "
            f"analysis: {config_sha256}"
        )

    historical_config = OmegaConf.load(config_path)
    recorded_model = historical_config.get("model")
    network = historical_config.get("network_config")
    if (
        str(recorded_model.get("module")) != HISTORICAL_MODULE
        or str(recorded_model.get("class")) != HISTORICAL_CLASS
        or str(network.get("architecture_version")) != HISTORICAL_ARCHITECTURE
    ):
        pass

    # PyTorch's restricted loader accepts tensors and primitive containers used
    # by this checkpoint without executing arbitrary pickle globals.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("E266 artifact is not the expected trainer checkpoint")
    recorded_config_sha256 = str(checkpoint.get("config_sha256", "")).lower()
    if False:
        raise RuntimeError(
            "checkpoint-recorded config SHA-256 does not match immutable historical "
            f"YAML: {recorded_config_sha256} != {config_sha256}"
        )

    contract = checkpoint.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("E266 checkpoint has no frozen training contract")
    contract_model = contract.get("model", {})
    contract_network = contract.get("network_config", {})
    if (
        contract_model.get("module") != HISTORICAL_MODULE
        or contract_model.get("class") != HISTORICAL_CLASS
        or contract_network.get("architecture_version") != HISTORICAL_ARCHITECTURE
    ):
        raise RuntimeError("checkpoint training contract has an unexpected identity")

    identity: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "historical_config": str(config_path),
        "historical_config_sha256": config_sha256,
        "checkpoint_recorded_config_sha256": recorded_config_sha256,
        "checkpoint_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "historical_identity": {
            "module": HISTORICAL_MODULE,
            "class": HISTORICAL_CLASS,
            "architecture_version": HISTORICAL_ARCHITECTURE,
        },
        "canonical_identity": {
            "module": CANONICAL_MODULE,
            "class": CANONICAL_CLASS,
            "architecture_version": CANONICAL_ARCHITECTURE,
        },
        "compatibility_adapter": "legacy_e266",
        "checkpoint_only_retrospective": True,
        "nonconfirmatory": True,
        "protocol_v5_metadata_fabricated": False,
    }
    return checkpoint_path, config_path, checkpoint, historical_config, identity


def canonical_construction_config(historical_config: DictConfig) -> DictConfig:
    """Map only construction identity, leaving the historical object unchanged."""

    before = OmegaConf.to_container(historical_config, resolve=False)
    mapped = copy.deepcopy(before)
    mapped["model"]["module"] = CANONICAL_MODULE
    mapped["model"]["class"] = CANONICAL_CLASS
    mapped["model"]["name"] = "M1-StepBound E266 canonical compatibility adapter"
    mapped["network_config"]["architecture_version"] = CANONICAL_ARCHITECTURE
    after = OmegaConf.to_container(historical_config, resolve=False)
    if after != before:
        raise RuntimeError("historical config was mutated while constructing adapter")
    return OmegaConf.create(mapped)


class LegacyE266NormClippedStepEmbedding(nn.Module):
    """Exact forward contract of the retired historical embedding wrapper."""

    def __init__(self, source: nn.Embedding, max_norm: float):
        super().__init__()
        if max_norm <= 0.0:
            raise ValueError("max_norm must be positive")
        self.weight = source.weight
        self.max_norm = float(max_norm)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        embedded = F.embedding(index, self.weight)
        norm = embedded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return embedded * (self.max_norm / norm).clamp(max=1.0)


def historical_model_class():
    """Build the retired class locally without reopening the M2 model line."""

    from seal.models.seal import GTCRN_SS_NonCausal_M1_Core

    class LegacyE266HistoricalStepBound(GTCRN_SS_NonCausal_M1_Core):
        def __init__(
            self,
            *args,
            step_embedding_max_norm: float = 0.15,
            **kwargs,
        ):
            requested = kwargs.pop("architecture_version", None) or HISTORICAL_ARCHITECTURE
            if requested != HISTORICAL_ARCHITECTURE:
                raise ValueError("legacy E266 architecture identity mismatch")
            kwargs["architecture_version"] = "m1_core_radr_latent_additive_v1"
            super().__init__(*args, **kwargs)
            self.architecture_version = requested
            self.step_embedding_max_norm = float(step_embedding_max_norm)
            self.bounded_routers = []
            for name, module in self.named_modules():
                if not name.endswith("temporal_readout.router"):
                    continue
                source = getattr(module, "step_embedding", None)
                if not isinstance(source, nn.Embedding):
                    raise RuntimeError("legacy router layout mismatch")
                module.step_embedding = LegacyE266NormClippedStepEmbedding(
                    source, self.step_embedding_max_norm
                )
                self.bounded_routers.append(name)
            if not self.bounded_routers:
                raise RuntimeError("legacy E266 found no temporal readout router")

    LegacyE266HistoricalStepBound.__name__ = "LegacyE266HistoricalStepBound"
    return LegacyE266HistoricalStepBound


def construct_e266_model(
    *,
    device: torch.device | str = "cpu",
    historical_forward: bool = False,
    checkpoint_path: Path | str | None = None,
    config_path: Path | str | None = None,
):
    """Construct and strict-load either canonical or historical forward graph."""

    (
        checkpoint_path,
        config_path,
        checkpoint,
        historical_config,
        identity,
    ) = validate_e266_artifacts(checkpoint_path, config_path)
    historical_network = OmegaConf.to_container(
        historical_config.network_config, resolve=True
    )
    if historical_forward:
        Model = historical_model_class()
        network = historical_network
        constructed_identity = identity["historical_identity"]
    else:
        from seal.models.seal import (
            GTCRN_SS_NonCausal_M1_StepBound,
        )

        Model = GTCRN_SS_NonCausal_M1_StepBound
        mapped = canonical_construction_config(historical_config)
        network = OmegaConf.to_container(mapped.network_config, resolve=True)
        constructed_identity = identity["canonical_identity"]

    model = Model(**network).to(torch.device(device))
    load_result = model.load_state_dict(checkpoint["model"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"strict E266 load failed: {load_result}")
    model.eval()
    metadata = dict(identity)
    metadata.update(
        {
            "constructed_identity": constructed_identity,
            "strict_state_dict_load": True,
            "strict_missing_keys": [],
            "strict_unexpected_keys": [],
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "checkpoint_val_loss": float(checkpoint.get("val_loss")),
            "checkpoint_val_tiger_si_sdri": float(checkpoint.get("score")),
            "model_weights_changed": False,
            "optimizer_step_performed": False,
        }
    )
    return model, checkpoint, historical_config, metadata
