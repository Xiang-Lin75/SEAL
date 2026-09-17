# Checkpoints

The paper-supporting epoch-266 checkpoint is distributed as a GitHub Release
asset rather than committed to Git history.

- File: `seal-small-e266.tar`
- SHA-256: `1eabd30d6983eadcb7a34ac2a04d7e9aac3f1233a06544b8420ddd1c1ad766c7`
- Status: historical single-run checkpoint used for the reported SEAL-small
  operating point; it is not a multi-seed confirmatory model.
- Format: PyTorch trainer checkpoint. The included loader verifies the digest
  and uses `weights_only=True`.

`e266_config_historical.yaml` intentionally retains the original laboratory
paths and retired M2-era identity because its exact SHA-256 is part of the
checkpoint contract. Those paths are provenance strings, not runtime defaults;
pass `--data-dir` to evaluation. New training uses the portable configs under
`configs/`.

Download the asset into this directory and use `--legacy-e266` with the public
inference or evaluation scripts.
