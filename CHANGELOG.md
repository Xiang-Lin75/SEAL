# Changelog

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
