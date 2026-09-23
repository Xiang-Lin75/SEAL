"""EchoSet loader for two-speaker reverberant speech separation."""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def normalize_audio(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return zero-mean, unit-variance audio."""

    return (wav - wav.mean(-1, keepdim=True)) / (wav.std(-1, keepdim=True) + eps)


class EchoSetDataset(Dataset):
    """Load EchoSet mixtures and reverberant source targets.

    ``data_dir`` may point directly to an extracted split::

        train/.../<pair>/mix.wav
        train/.../<pair>/spk1_reverb.wav
        train/.../<pair>/spk2_reverb.wav

    It may instead point to the EchoSet root when ``split`` is supplied, or to
    a legacy index directory containing ``mix.json``, ``s1.json`` and
    ``s2.json``. Fixed-length examples are cropped at a random offset when
    ``random_start`` is true. Full utterances are returned when ``segment`` is
    null, and the length-aware collate function pads them per batch while
    preserving their original sample counts.
    """

    def __init__(
        self,
        data_dir: str,
        split: Optional[str] = None,
        sample_rate: int = 16000,
        segment: Optional[float] = 3.0,
        num_sources: int = 2,
        normalize: bool = False,
        random_start: bool = False,
        return_key: bool = False,
        check_sample_rate: bool = True,
        crop_seed: Optional[int] = None,
    ):
        super().__init__()
        if num_sources not in (1, 2):
            raise ValueError(f"num_sources must be 1 or 2, got {num_sources}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if segment is not None and segment <= 0:
            raise ValueError(f"segment must be positive or null, got {segment}")

        self.data_dir = str(data_dir)
        self.root_dir = Path(data_dir).expanduser().resolve()
        self.split = split
        self.sample_rate = int(sample_rate)
        self.num_sources = int(num_sources)
        self.normalize = bool(normalize)
        self.random_start = bool(random_start)
        self.return_key = bool(return_key)
        self.check_sample_rate = bool(check_sample_rate)
        self.crop_seed: Optional[int] = None
        self.crop_epoch = 0
        self.set_crop_seed(crop_seed)
        self.eps = 1e-8
        self.seg_len = (
            int(float(segment) * self.sample_rate) if segment is not None else None
        )

        self.samples = []
        if (self.root_dir / "mix.json").exists():
            self._init_json_index(self.root_dir)
        else:
            self._init_extracted_tree()

        if self.seg_len is not None:
            self._filter_short_samples()
        if not self.samples:
            raise RuntimeError(f"No usable EchoSet samples found under {self.scan_dir}")

        print(f"[EchoSetDataset] Loaded {len(self.samples)} samples from {self.scan_dir}")
        if self.seg_len is not None:
            print(
                f"[EchoSetDataset] Segment: {self.seg_len / self.sample_rate:.1f}s "
                f"= {self.seg_len} samples"
            )

    def _init_json_index(self, index_dir: Path) -> None:
        self.scan_dir = index_dir
        with open(index_dir / "mix.json", "r", encoding="utf-8") as handle:
            mix_infos = json.load(handle)

        source_infos = []
        for source_index in range(1, self.num_sources + 1):
            index_path = index_dir / f"s{source_index}.json"
            if not index_path.exists():
                raise FileNotFoundError(f"EchoSet source index not found: {index_path}")
            with open(index_path, "r", encoding="utf-8") as handle:
                current_infos = json.load(handle)
            if len(current_infos) != len(mix_infos):
                raise ValueError(
                    f"EchoSet index length mismatch: {index_path} has "
                    f"{len(current_infos)} entries, mix.json has {len(mix_infos)}"
                )
            source_infos.append(current_infos)

        for sample_index, mix_info in enumerate(mix_infos):
            mix_path = self._resolve_path(str(mix_info[0]))
            sources = [
                self._resolve_path(str(infos[sample_index][0]))
                for infos in source_infos
            ]
            self.samples.append(
                {
                    "mix": mix_path,
                    "sources": sources,
                    "length": int(mix_info[1]),
                    "key": Path(mix_path).stem,
                    "crop_index": sample_index,
                }
            )

    def _init_extracted_tree(self) -> None:
        self.scan_dir = self._resolve_scan_dir()
        if not self.scan_dir.is_dir():
            raise FileNotFoundError(f"EchoSet directory not found: {self.scan_dir}")

        mix_paths = sorted(self.scan_dir.rglob("mix.wav"))
        if not mix_paths:
            raise RuntimeError(f"No mix.wav files found under {self.scan_dir}")

        missing_sources = []
        for sample_index, mix_path in enumerate(mix_paths):
            sample_dir = mix_path.parent
            source_paths = [
                sample_dir / f"spk{source_index}_reverb.wav"
                for source_index in range(1, self.num_sources + 1)
            ]
            missing = [str(path) for path in source_paths if not path.is_file()]
            if missing:
                missing_sources.extend(missing)
                continue

            info = sf.info(str(mix_path))
            self._validate_audio_info(mix_path, info)
            self.samples.append(
                {
                    "mix": str(mix_path),
                    "sources": [str(path) for path in source_paths],
                    "length": int(info.frames),
                    "key": sample_dir.relative_to(self.scan_dir).as_posix(),
                    "crop_index": sample_index,
                }
            )

        if missing_sources:
            preview = "\n".join(missing_sources[:10])
            raise FileNotFoundError(
                f"Missing EchoSet source files for {len(missing_sources)} paths. "
                f"First missing files:\n{preview}"
            )

    def _resolve_scan_dir(self) -> Path:
        if self.split:
            return self.root_dir / self.split
        if self.root_dir.name in {"train", "val", "valid", "validation", "test"}:
            return self.root_dir

        available = [
            name for name in ("train", "val", "test")
            if (self.root_dir / name).is_dir()
        ]
        if available:
            raise ValueError(
                "data_dir points to an EchoSet root. Pass split=... or point "
                f"to one split directory directly. Available: {available}"
            )
        return self.root_dir

    def _filter_short_samples(self) -> None:
        original_count = len(self.samples)
        self.samples = [
            sample for sample in self.samples if sample["length"] >= self.seg_len
        ]
        dropped = original_count - len(self.samples)
        if dropped:
            print(
                f"[EchoSetDataset] Dropped {dropped}/{original_count} samples "
                f"shorter than {self.seg_len / self.sample_rate:.1f}s"
            )

    def _resolve_path(self, raw_path: str) -> str:
        if os.path.exists(raw_path):
            return raw_path

        path = Path(raw_path)
        if not path.is_absolute():
            local_path = self.root_dir / path
            if local_path.exists():
                return str(local_path)

        if "EchoSet" in path.parts:
            suffix = Path(*path.parts[path.parts.index("EchoSet") + 1 :])
            local_path = self.root_dir / suffix
            if local_path.exists():
                return str(local_path)
        return raw_path

    def _validate_audio_info(self, path: Path, info) -> None:
        if self.check_sample_rate and info.samplerate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {path}: "
                f"{info.samplerate} != {self.sample_rate}"
            )
        if info.channels != 1:
            raise ValueError(f"EchoSet audio must be mono: {path} has {info.channels} channels")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _validate_integer(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        return int(value)

    def set_crop_seed(self, crop_seed: Optional[int]) -> None:
        """Enable deterministic random crops, or restore legacy randomness.

        A value of ``None`` preserves the original behavior: every random crop
        draws from NumPy's process-local random state. An integer makes the crop
        a pure function of ``(crop_seed, crop_epoch, sample key)``.
        """

        if crop_seed is None:
            self.crop_seed = None
            return
        self.crop_seed = self._validate_integer("crop_seed", crop_seed)

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used by deterministic random cropping."""

        epoch = self._validate_integer("epoch", epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.crop_epoch = epoch

    def _deterministic_crop_start(
        self,
        sample: dict,
        num_starts: int,
        *,
        epoch: Optional[int] = None,
    ) -> int:
        if self.crop_seed is None:
            raise RuntimeError(
                "Deterministic crop positions require crop_seed to be set"
            )
        selected_epoch = self.crop_epoch if epoch is None else self._validate_integer(
            "epoch", epoch
        )
        if selected_epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {selected_epoch}")

        # Do not use Python's salted hash() or an RNG implementation here. This
        # byte-level definition is stable across workers, ranks, and processes.
        crop_identity = {
            "schema": "echoset-crop-v1",
            "crop_seed": self.crop_seed,
            "epoch": selected_epoch,
            "sample_index": int(sample["crop_index"]),
            "sample_key": str(sample["key"]),
            "audio_length": int(sample["length"]),
            "segment_samples": self.seg_len,
        }
        encoded_identity = json.dumps(
            crop_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded_identity).digest()
        return int.from_bytes(digest[:8], byteorder="big") % num_starts

    def _crop_bounds(
        self,
        sample: dict,
        *,
        epoch: Optional[int] = None,
        require_deterministic: bool = False,
    ) -> Tuple[int, Optional[int]]:
        audio_len = int(sample["length"])
        if self.seg_len is None:
            return 0, None
        if audio_len == self.seg_len:
            return 0, None
        if not self.random_start:
            return 0, self.seg_len

        num_starts = audio_len - self.seg_len + 1
        if self.crop_seed is not None:
            start = self._deterministic_crop_start(
                sample,
                num_starts,
                epoch=epoch,
            )
        elif require_deterministic:
            raise RuntimeError(
                "Cannot create a stable random-crop manifest while crop_seed is "
                "None; set crop_seed or disable random_start"
            )
        else:
            start = int(np.random.randint(0, num_starts))
        return start, start + self.seg_len

    def crop_manifest(self, epoch: Optional[int] = None) -> dict:
        """Return the canonical, stable record of all crop decisions.

        Records are sorted by discovery-time sample index, so the result is
        independent of
        sampler, worker, rank, and in-memory dataset order. For stochastic
        ``random_start`` datasets, ``crop_seed`` must be configured.
        """

        selected_epoch = self.crop_epoch if epoch is None else self._validate_integer(
            "epoch", epoch
        )
        if selected_epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {selected_epoch}")

        records = []
        for sample in self.samples:
            start, stop = self._crop_bounds(
                sample,
                epoch=selected_epoch,
                require_deterministic=True,
            )
            records.append(
                {
                    "index": int(sample["crop_index"]),
                    "key": str(sample["key"]),
                    "length": int(sample["length"]),
                    "start": start,
                    "stop": stop,
                }
            )
        records.sort(
            key=lambda record: (
                record["index"],
                record["key"],
                record["length"],
                record["start"],
                -1 if record["stop"] is None else record["stop"],
            )
        )
        return {
            "schema": "echoset-crop-manifest-v1",
            "crop_seed": self.crop_seed,
            "epoch": selected_epoch,
            "random_start": self.random_start,
            "segment_samples": self.seg_len,
            "samples": records,
        }

    def crop_manifest_sha256(self, epoch: Optional[int] = None) -> str:
        """Return a stable SHA-256 digest of ``crop_manifest(epoch)``."""

        manifest = self.crop_manifest(epoch=epoch)
        encoded_manifest = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded_manifest).hexdigest()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[index]
        start, stop = self._crop_bounds(sample)

        mix, mix_sr = sf.read(
            sample["mix"], start=start, stop=stop, dtype="float32"
        )
        self._validate_loaded_audio(sample["mix"], mix, mix_sr)

        source_waves = []
        for source_path in sample["sources"]:
            source, source_sr = sf.read(
                source_path, start=start, stop=stop, dtype="float32"
            )
            self._validate_loaded_audio(source_path, source, source_sr)
            source_waves.append(source)

        common_len = min(len(wave) for wave in [mix, *source_waves])
        mix_tensor = torch.from_numpy(mix[:common_len])
        source_tensor = torch.from_numpy(
            np.stack([wave[:common_len] for wave in source_waves], axis=0)
        )

        if self.normalize:
            mix_std = mix_tensor.std(keepdim=True)
            mix_tensor = normalize_audio(mix_tensor, eps=self.eps)
            source_tensor = (
                source_tensor - source_tensor.mean(dim=-1, keepdim=True)
            ) / (mix_std + self.eps)

        if self.return_key:
            return mix_tensor, source_tensor, sample["key"]
        return mix_tensor, source_tensor

    def _validate_loaded_audio(self, path: str, wave: np.ndarray, sample_rate: int) -> None:
        if self.check_sample_rate and sample_rate != self.sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {path}: {sample_rate} != {self.sample_rate}"
            )
        if wave.ndim != 1:
            raise ValueError(f"EchoSet audio must be mono: {path} has shape {wave.shape}")

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        """Pad variable-length examples and retain optional utterance keys."""

        if len(batch[0]) == 3:
            mixes, sources, keys = zip(*batch)
        else:
            mixes, sources = zip(*batch)
            keys = None

        max_len = max(mix.shape[0] for mix in mixes)
        padded_mixes = []
        padded_sources = []
        for mix, source in zip(mixes, sources):
            pad_len = max_len - mix.shape[0]
            if pad_len:
                mix = torch.nn.functional.pad(mix, (0, pad_len))
                source = torch.nn.functional.pad(source, (0, pad_len))
            padded_mixes.append(mix)
            padded_sources.append(source)

        collated = (torch.stack(padded_mixes), torch.stack(padded_sources))
        if keys is not None:
            return (*collated, list(keys))
        return collated

    @staticmethod
    def collate_fn_with_lengths(
        batch: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, ...]:
        """Pad a batch and include each example's true sample count."""

        lengths = torch.tensor(
            [item[0].shape[0] for item in batch], dtype=torch.long
        )
        collated = EchoSetDataset.collate_fn(batch)
        if len(collated) == 3:
            mixes, sources, keys = collated
            return mixes, sources, lengths, keys
        mixes, sources = collated
        return mixes, sources, lengths


if __name__ == "__main__":
    print("EchoSetDataset module loaded successfully.")
