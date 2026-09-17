"""Measure SEAL parameters and one-second MACs with TIGER's ptflops protocol."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from omegaconf import OmegaConf
from ptflops import get_model_complexity_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/seal_small_echoset.yaml"))
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    module = importlib.import_module(str(config.model.module))
    cls = getattr(module, str(config.model["class"]))
    model = cls(**OmegaConf.to_container(config.network_config, resolve=True)).eval()
    macs, parameters = get_model_complexity_info(
        model,
        (16000,),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"parameters: {int(parameters):,}")
    print(f"MAC/s: {float(macs) / 1e9:.9f} G")


if __name__ == "__main__":
    main()
