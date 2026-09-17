# Paper ablations

This directory contains the seven configurations behind the compact paper
table: Full plus six direct controls.

| Configuration | Question |
|---|---|
| `p0_00_full_m1_stepbound` | reference SEAL-small operating point |
| `p0_07_multiplicative_only` | is the additive complex correction useful? |
| `p0_09_single_latent_atom` | does the six-atom factorization help over one atom? |
| `p1_18_dense_temporal_readout` | does the sparse expert bank help over one dense readout? |
| `p2_24_fixed_hashed_routing` | does learned assignment help over a fixed rule? |
| `p1_20_no_progress_router_evidence` | do delta/trajectory cues help routing? |
| `p0_02_no_step_embedding` | does the additive refinement-step cue help? |

The table is an architectural ablation, not a hyperparameter sweep. StepBound
`rho=0.15`, Top-1 routing, six experts, four refinements, and six latent atoms
are fixed operating choices. Do not claim global optimality from this matrix.

Each run should preserve its resolved configuration, source commit, checkpoint
SHA-256, ordered test keys, and per-utterance metric CSV. A rounded score without
those artifacts is not release-grade evidence.

Resolve and launch one arm with:

```bash
export ECHOSET_ROOT=/path/to/EchoSet
python -m ablations.run \
  --arm ablations/configs/p0_07_multiplicative_only.yaml \
  --device 0
```

Add `--dry-run` to inspect the merged configuration without training.
