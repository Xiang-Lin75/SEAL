"""Evaluate a SEAL checkpoint with the TIGER-compatible SI-SDRi convention."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from seal.data.dataloader_echoset import EchoSetDataset
from inference import load_model
from seal.metrics.metrics_ss import pit_si_sdr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--legacy-e266", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = EchoSetDataset(
        args.data_dir,
        sample_rate=16000,
        segment=None,
        num_sources=2,
        normalize=False,
        random_start=False,
        return_key=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn_with_lengths,
    )
    model, _ = load_model(args.config, args.checkpoint, device, args.legacy_e266)
    rows = []
    with torch.inference_mode():
        for mixtures, targets, lengths, keys in loader:
            length = int(lengths[0])
            mixture = mixtures[:, :length].to(device)
            target = targets[:, :, :length].to(device)
            estimate = model(mixture)[..., :length]
            output_score = pit_si_sdr(estimate, target, zero_mean=False)
            input_sources = mixture[:, None, :].expand_as(target)
            input_score = pit_si_sdr(input_sources, target, zero_mean=False)
            rows.append(
                {
                    "key": keys[0],
                    "tiger_si_sdr": float(output_score[0].cpu()),
                    "tiger_si_sdri": float((output_score - input_score)[0].cpu()),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = sum(row["tiger_si_sdri"] for row in rows) / len(rows)
    print(f"EchoSet utterances: {len(rows)}")
    print(f"Mean TIGER-compatible SI-SDRi: {mean:.6f} dB")
    print(f"Per-utterance output: {args.output_csv}")


if __name__ == "__main__":
    main()
