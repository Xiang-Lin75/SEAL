"""Load the released SEAL-small E266 checkpoint.

The checkpoint and its training config (``e266_config_historical.yaml``) were
produced before the public renaming, so they record an internal lab identity
(``GTCRN_SS_NonCausal_M2_StepBound`` / ``m2_stepbound_v1``) and the config key
``paired_m0_initialization``. Both files are pinned by SHA-256 and are never
modified. This adapter verifies both digests, then maps only the construction
identity and that one key onto the public :class:`seal.models.SEAL` class and
strict-loads the unchanged state dictionary.

The trainer checkpoint also stores optimizer and NumPy RNG state, which
PyTorch's ``weights_only=True`` loader rejects. It is therefore unpickled with
``weights_only=False``, but only after its SHA-256 matches the released file.
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
CHECKPOINT_RELATIVE = Path("checkpoints/seal-small-e266.tar")
CONFIG_RELATIVE = Path("checkpoints/e266_config_historical.yaml")
EXPECTED_CHECKPOINT_SHA256 = (
    "1eabd30d6983eadcb7a34ac2a04d7e9aac3f1233a06544b8420ddd1c1ad766c7"
)
EXPECTED_CONFIG_SHA256 = (
    "810588c1f68c3514a2d19e1e1a46120e115b7b9f5cea7429e80c62f3f8ad8cd9"
)
HISTORICAL_MODULE = "models.gtcrn_ss_noncausal_M2_stepbound"
HISTORICAL_CLASS = "GTCRN_SS_NonCausal_M2_StepBound"
HISTORICAL_ARCHITECTURE = "m2_stepbound_v1"
CANONICAL_MODULE = "seal.models.seal"
CANONICAL_CLASS = "SEAL"
CANONICAL_ARCHITECTURE = "seal_v1"
# Config keys renamed for the public release: historical -> public.
RENAMED_NETWORK_KEYS = {"paired_m0_initialization": "paired_initialization"}


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
    """Verify the pinned files and return their decoded contents."""

    checkpoint_path = _resolve(checkpoint_path, CHECKPOINT_RELATIVE)
    config_path = _resolve(config_path, CONFIG_RELATIVE)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_sha256 = sha256_file(config_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "E266 checkpoint SHA-256 mismatch; download seal-small-e266.tar "
            f"from the GitHub release again (got {checkpoint_sha256})"
        )
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "E266 config SHA-256 mismatch; use the unmodified "
            f"checkpoints/e266_config_historical.yaml (got {config_sha256})"
        )

    historical_config = OmegaConf.load(config_path)
    recorded_model = historical_config.get("model")
    network = historical_config.get("network_config")
    if (
        str(recorded_model.get("module")) != HISTORICAL_MODULE
        or str(recorded_model.get("class")) != HISTORICAL_CLASS
        or str(network.get("architecture_version")) != HISTORICAL_ARCHITECTURE
    ):
        raise RuntimeError("historical E266 identity fields do not match the contract")

    # Safe only because the digest above pins the exact released file.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("E266 artifact is not the expected trainer checkpoint")
    recorded_config_sha256 = str(checkpoint.get("config_sha256", "")).lower()
    if recorded_config_sha256 != config_sha256:
        raise RuntimeError(
            "checkpoint-recorded config SHA-256 does not match the historical "
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
    }
    return checkpoint_path, config_path, checkpoint, historical_config, identity


def _public_network_config(network: Dict[str, object]) -> Dict[str, object]:
    renamed = {RENAMED_NETWORK_KEYS.get(key, key): value for key, value in network.items()}
    renamed.pop("architecture_version", None)
    return renamed


def canonical_construction_config(historical_config: DictConfig) -> DictConfig:
    """Map construction identity and renamed keys, leaving the input unchanged."""

    before = OmegaConf.to_container(historical_config, resolve=False)
    mapped = copy.deepcopy(before)
    mapped["model"]["module"] = CANONICAL_MODULE
    mapped["model"]["class"] = CANONICAL_CLASS
    mapped["model"]["name"] = "SEAL-small E266 (public class)"
    mapped["network_config"] = _public_network_config(mapped["network_config"])
    mapped["network_config"]["architecture_version"] = CANONICAL_ARCHITECTURE
    after = OmegaConf.to_container(historical_config, resolve=False)
    if after != before:
        raise RuntimeError("historical config was mutated while constructing adapter")
    return OmegaConf.create(mapped)


class LegacyE266NormClippedStepEmbedding(nn.Module):
    """Exact forward contract of the historical step-embedding wrapper."""

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
    """Rebuild the historical graph locally, as an independent cross-check."""

    from seal.models.seal import SEALCore

    class LegacyE266HistoricalStepBound(SEALCore):
        def __init__(
            self,
            *args,
            step_embedding_max_norm: float = 0.15,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.architecture_version = HISTORICAL_ARCHITECTURE
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

    return LegacyE266HistoricalStepBound


def construct_e266_model(
    *,
    device: torch.device | str = "cpu",
    historical_forward: bool = False,
    checkpoint_path: Path | str | None = None,
    config_path: Path | str | None = None,
):
    """Construct and strict-load the public (or historical) E266 graph."""

    (
        checkpoint_path,
        config_path,
        checkpoint,
        historical_config,
        identity,
    ) = validate_e266_artifacts(checkpoint_path, config_path)
    if historical_forward:
        Model = historical_model_class()
        network = _public_network_config(
            OmegaConf.to_container(historical_config.network_config, resolve=True)
        )
        constructed_identity = identity["historical_identity"]
    else:
        from seal.models.seal import SEAL

        Model = SEAL
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
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "checkpoint_val_loss": float(checkpoint.get("val_loss")),
            "checkpoint_val_tiger_si_sdri": float(checkpoint.get("score")),
        }
    )
    return model, checkpoint, historical_config, metadata
