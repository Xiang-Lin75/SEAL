# Reproducibility guide

## Environment

The recorded E266 environment used Python 3.12.11, PyTorch 2.6.0+cu124, CUDA
12.4, and cuDNN 9.1 on an RTX 4060 Ti. Other supported PyTorch 2.6+ builds can
train or run inference, but exact runtime numbers are hardware-specific.

## EchoSet layout

Set `ECHOSET_ROOT` to the extracted dataset root:

```text
EchoSet/
  train/**/mix.wav, spk1_reverb.wav, spk2_reverb.wav
  val/**/mix.wav, spk1_reverb.wav, spk2_reverb.wav
  test/**/mix.wav, spk1_reverb.wav, spk2_reverb.wav
```

## Train

```bash
export ECHOSET_ROOT=/path/to/EchoSet
python train.py -C configs/seal_small_echoset.yaml -D 0
```

## Evaluate

```bash
python evaluate.py \
  --config checkpoints/e266_config_historical.yaml \
  --checkpoint checkpoints/seal-small-e266.tar \
  --legacy-e266 \
  --data-dir "$ECHOSET_ROOT/test" \
  --output-csv outputs/e266_test.csv
```

Formal comparisons must use identical ordered test keys and paired statistics.
Do not compare validation loss to test SI-SDRi or silently switch zero-mean
conventions.
