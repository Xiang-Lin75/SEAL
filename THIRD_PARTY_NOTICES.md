# Third-party notices

SEAL is released under MIT. Portions of the implementation build on or adapt
ideas and code from the following MIT-licensed projects. Their copyright and
license notices must be retained in redistributed substantial portions.

## GTCRN

- Source: https://github.com/Xiaobin-Rong/gtcrn
- Copyright (c) 2024 Rong Xiaobin
- License: MIT
- Full notice: [`LICENSES/GTCRN-MIT.txt`](LICENSES/GTCRN-MIT.txt)

The ERB frontend and several convolution/recurrent building blocks descend from
GTCRN and were extended for non-causal separation, shared refinement, routing,
and multi-source reconstruction.

## TIGER

- Source: https://github.com/JusperLee/TIGER
- Copyright (c) 2026 Kai Li
- Repository license file: MIT
- Full notice: [`LICENSES/TIGER-MIT.txt`](LICENSES/TIGER-MIT.txt)

The training/evaluation parity code adapts TIGER's public loss and metric
conventions. TIGER's README badge and its Hugging Face model cards currently
say Apache-2.0 while the GitHub source repository contains an MIT `LICENSE`;
this release records the source repository license actually distributed with
the referenced code.

## Datasets and weights

EchoSet, LibriMix/Libri2Mix, LibriSpeech, and any other datasets are not bundled.
Their own licenses and terms apply. The EchoSet Hugging Face card currently
declares Apache-2.0. SEAL's MIT license does not relicense datasets, audio, model
weights from third parties, or papers.
