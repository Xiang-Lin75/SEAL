"""Resolve and launch one public SEAL ablation configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    arm_path = args.arm.resolve()
    arm = OmegaConf.load(arm_path)
    base_path = (arm_path.parent / str(arm.base_config)).resolve()
    base = OmegaConf.load(base_path)
    resolved = OmegaConf.merge(base, arm.get("overrides", {}))
    resolved.ablation_metadata = {
        "id": str(arm.id),
        "claim": str(arm.get("claim", "")),
        "control": str(arm.get("control", "")),
        "source_arm": arm_path.as_posix(),
        "source_base": base_path.as_posix(),
    }

    output = args.output or Path("outputs") / "resolved" / f"{arm.id}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(resolved, output)
    print(f"Resolved {arm.id} -> {output}")
    if not args.dry_run:
        subprocess.run(
            [sys.executable, "train.py", "-C", str(output), "-D", args.device],
            check=True,
        )


if __name__ == "__main__":
    main()
