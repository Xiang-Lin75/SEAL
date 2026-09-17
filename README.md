# SEAL

**Sparse Expert readout with Additive Latent reconstruction** for compact,
non-causal, single-channel speech separation.

SEAL combines three design ideas:

1. a shared-weight DPGRNN cell that iteratively updates a feature state;
2. refinement-aware Top-1 routing over stateless residual experts, with a
   norm-capped step cue;
3. conservation-structured six-atom latent reconstruction with a bounded
   additive complex correction and a residual sink that preserves mixture
   closure.

![Architecture](assets/architecture.pdf)

## Release contents

This repository provides more than the minimum code-only release:

- the SEAL-small model and its GTCRN-derived backbone;
- EchoSet and Libri2Mix data loaders;
- training losses, TIGER-compatible SI-SDRi evaluation, and a trainer;
- inference and evaluation commands;
- the seven configurations in the compact ICASSP ablation table;
- model invariant, dataloader, and metric tests;
- a model card, reproducibility guide, citation metadata, CI, security notes,
  and third-party attribution;
- a verified E266 checkpoint as a GitHub Release asset rather than Git history.

Datasets, experiment workspaces, TensorBoard logs, private machine paths,
third-party papers, and superseded M2 experiments are intentionally excluded.

## Installation

Python 3.10+ and PyTorch 2.6+ are recommended.

```bash
git clone git@github.com:Xiang-Lin75/SEAL.git
cd SEAL
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For the exact CUDA 12.4 reference environment, install the appropriate PyTorch
wheel first and then run `pip install -e .`.

## Pretrained checkpoint and inference

Download `seal-small-e266.tar` from the `v0.1.0` GitHub release into
`checkpoints/`. Verify the SHA-256 shown in
[`checkpoints/README.md`](checkpoints/README.md), then run:

```bash
python inference.py \
  --config checkpoints/e266_config_historical.yaml \
  --checkpoint checkpoints/seal-small-e266.tar \
  --legacy-e266 \
  --audio mixture.wav \
  --output-dir outputs/example
```

Input must be mono or downmixable 16 kHz audio. SEAL-small outputs two sources.
The model is non-causal and is intended for offline processing.

## Train on EchoSet

Download and extract [EchoSet](https://huggingface.co/datasets/JusperLee/EchoSet),
then point `ECHOSET_ROOT` at the directory containing `train`, `val`, and `test`.

```bash
export ECHOSET_ROOT=/path/to/EchoSet
python train.py -C configs/seal_small_echoset.yaml -D 0
```

For multiple GPUs, pass comma-separated device indices such as `-D 0,1`.
Absolute laboratory paths are not embedded in the public configuration.

## Evaluate

```bash
python evaluate.py \
  --config checkpoints/e266_config_historical.yaml \
  --checkpoint checkpoints/seal-small-e266.tar \
  --legacy-e266 \
  --data-dir /path/to/EchoSet/test \
  --output-csv outputs/e266_test.csv
```

The public headline metric is utterance-level PIT SI-SDRi with
`zero_mean=False`, matching TIGER's public convention. Evaluation writes one
row per utterance before printing the mean; this enables paired comparisons.

## Reported SEAL-small operating point

| Dataset | SI-SDRi | BSS-SDRi | Parameters | MAC/s |
|---|---:|---:|---:|---:|
| EchoSet test | 12.8906 dB | 13.6188 dB | 590,463 | 2.636825 G |

This is one historical validation-selected E266 run, not a multi-seed estimate.
See [`MODEL_CARD.md`](MODEL_CARD.md) for metric definitions and limitations.

## Ablations

[`ablations/README.md`](ablations/README.md) maps every paper row to one causal
question. The core table tests additive reconstruction, atom factorization,
expert capacity, learned routing, progress evidence, and the step cue. It does
not claim that the chosen Top-K, number of experts, number of atoms, or bound is
globally optimal.

```bash
python -m ablations.run \
  --arm ablations/configs/p0_07_multiplicative_only.yaml \
  --device 0
```

## Repository layout

```text
models/                 SEAL and inherited backbone modules
configs/                portable training configuration
ablations/              paper-control implementations and configs
entrypoints/train.py    trainer implementation
train.py                root training launcher
inference.py            waveform separation
evaluate.py             per-utterance SI-SDRi evaluation
scripts/                complexity and legacy-checkpoint utilities
tests/                  architecture, data, and metric regressions
assets/                 paper architecture diagrams
docs/                   detailed reproducibility instructions
```

## Relationship to TIGER

[TIGER](https://github.com/JusperLee/TIGER) publishes model/training code,
EchoSet preprocessing support, inference scripts, two EchoSet training configs,
demo assets, a static demo site, and three pretrained speech-separation models
on Hugging Face. Its current README command still names a missing
`configs/tiger.yml`; the repository actually contains `tiger-small.yml` and
`tiger-large.yml`. Its test script is also fixed to one sample index rather
than a full benchmark loop. SEAL therefore exposes a portable config and a
full ordered-dataset evaluator directly.

TIGER's GitHub `LICENSE` is MIT, while its README badge and Hugging Face cards
say Apache-2.0. SEAL uses one explicit MIT license and records all inherited
notices in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). The manuscript citation
will be added after publication metadata is available.

SEAL code is MIT licensed. Dataset, third-party code, paper, and model-weight
terms remain separate; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
