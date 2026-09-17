# SEAL-small model card

SEAL is a two-speaker, single-channel, non-causal speech separator. The public
model combines shared iterative feature-state refinement, refinement-aware Top-1
routing with a bounded step cue, and conservation-structured six-atom latent
reconstruction with an additive complex correction.

## Intended use

- Research on offline two-speaker speech separation at 16 kHz.
- Reproduction and analysis on EchoSet-compatible mixtures.
- Not intended for safety-critical, forensic, surveillance, or real-time causal
  use. The architecture uses future context.

## Reported operating point

| Model | Dataset | SI-SDRi | BSS-SDRi | Parameters | MAC/s |
|---|---|---:|---:|---:|---:|
| SEAL-small E266 | EchoSet test | 12.8906 dB | 13.6188 dB | 590,463 | 2.636825 G |

The scores are from one historical validation-selected run. SI-SDRi uses
utterance-level PIT with `zero_mean=False`; BSS-SDRi uses a separate
`fast_bss_eval` assignment with a 512-tap filter. These are not multi-seed
confidence estimates. Cross-corpus and independently retrained baseline results
are not established by this release.

## Checkpoint

The release asset is `seal-small-e266.tar`, SHA-256
`1eabd30d6983eadcb7a34ac2a04d7e9aac3f1233a06544b8420ddd1c1ad766c7`.
It preserves the historical trainer state and naming. The compatibility adapter
maps only the construction identity to the public SEAL class and strict-loads
the unchanged state dictionary.

## Limitations

EchoSet is a simulated acoustic corpus. Performance can degrade under unseen
languages, microphones, room responses, speaker counts, clipping, or sampling
rates. Separation errors may alter words or speaker identity cues. Always audit
outputs in the target domain.
