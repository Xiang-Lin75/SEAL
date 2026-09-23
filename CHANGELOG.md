# Changelog

## 0.2.0 - unreleased

Model outputs, initialization, parameter count, and MACs are bit-identical to
0.1.2; the E266 checkpoint loads unchanged.

- Split the 5,194-line `seal/models/seal.py` into `seal.py`, `separator.py`,
  `routing.py`, `heads.py`, and `blocks.py`, and removed six unused classes
  from the pre-SEAL single-stage model.
- Public class names: `SEAL`, `SEALCore`, `SEALBaseline` replace the internal
  `GTCRN_SS_NonCausal_*` names; `get_diagnostics()` / `clear_aux()` replace
  the `*_m0_*` / `*_m1_*` methods. Configs use `class: SEAL`,
  `architecture_version: seal_v1`, and `paired_initialization`.
- Removed the CSR mask-head ablation option and legacy constructor aliases.
- Fixed `train.py` crashing at startup: its code snapshot still copied the
  removed `models/` and `ABLATION/` directories.
- `inference.py` / `evaluate.py` can now load self-trained checkpoints via
  `--trust-checkpoint`; previously the restricted loader always refused them.
- The E266 loader verifies the checkpoint and config SHA-256 again before
  unpickling; documentation now describes what it actually does.
- `train.py -D cpu` runs the trainer without a GPU.
- New `tests/test_train_smoke.py` runs the documented workflow on every CI
  push: train one epoch on CPU, resume to a second epoch, then separate with
  `inference.py`. It fails if the trainer cannot start from a fresh clone.

## 0.1.2 - 2026-09-23

- Restructured the code as the `seal` package (`seal/data`, `seal/models`,
  `seal/losses`, `seal/metrics`, `seal/utils`) and consolidated the model into
  `seal/models/seal.py`.
- Fixed `seal/data/dataloader_echoset.py` missing from the repository: the
  `data/` ignore rule also matched the package directory, which broke
  `train.py` and `evaluate.py` on a fresh clone. Ignore rules for local data
  now apply to the repository root only.
- The EchoSet test suite now fails, instead of skipping, when a SEAL module
  cannot be imported.
- Removed the ablation configurations and the unused Libri2Mix/WHAM loaders
  from the public release.
- Declared PESQ/STOI as the optional `metrics` extra.
- Rounded the README results table; exact values remain in the model card.

## 0.1.1 - 2026-09-17

- Fixed the portable M1 configuration path used by the public test suite.
- Changed checkpoint instructions to track the latest GitHub release.

## 0.1.0 - 2026-09-17

- Initial public SEAL-small implementation.
- Portable EchoSet small and R8 scaling configurations.
- Training, inference, ordered test-set evaluation, and complexity tools.
- Seven paper-ablation configurations.
- Architecture, data-loader, and metric regression tests with GitHub Actions.
- E266 historical checkpoint published as a verified release asset.
