"""Separate a mono waveform with SEAL.

Examples:
    python inference.py --config configs/seal_small_echoset.yaml \
        --checkpoint checkpoints/seal-small-e266.tar --legacy-e266 \
        --audio mixture.wav --output-dir outputs/example
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device, legacy_e266: bool):
    if legacy_e266:
        from scripts.legacy_e266_adapter import construct_e266_model

        model, _, _, metadata = construct_e266_model(
            device=device,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
        return model, metadata

    config = OmegaConf.load(config_path)
    module = importlib.import_module(str(config.model.module))
    model_class = getattr(module, str(config.model["class"]))
    kwargs = OmegaConf.to_container(config.network_config, resolve=True)
    model = model_class(**kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dictionary")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    model.eval()
    return model, {"strict_state_dict_load": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-speaker separation with SEAL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--legacy-e266", action="store_true")
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"SEAL expects 16 kHz audio, received {sample_rate} Hz")
    if waveform.shape[1] != 1:
        waveform = waveform.mean(axis=1, keepdims=True)

    device = torch.device(args.device)
    model, metadata = load_model(args.config, args.checkpoint, device, args.legacy_e266)
    mixture = torch.from_numpy(np.ascontiguousarray(waveform[:, 0])).unsqueeze(0).to(device)
    with torch.inference_mode():
        estimates = model(mixture)
    if estimates.ndim != 3 or estimates.shape[0] != 1:
        raise RuntimeError(f"unexpected model output shape: {tuple(estimates.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(estimates[0].detach().cpu().numpy(), start=1):
        sf.write(args.output_dir / f"source{index}.wav", source, sample_rate)
    print(f"Wrote {estimates.shape[1]} sources to {args.output_dir}")
    if metadata.get("checkpoint_only_retrospective"):
        print("Loaded the verified historical E266 checkpoint.")


if __name__ == "__main__":
    main()
