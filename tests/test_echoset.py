"""Regression tests for EchoSet discovery, cropping, and length-aware collation."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import soundfile as sf
    import torch

    from dataloader_echoset import EchoSetDataset
except ImportError as exc:  # pragma: no cover - exercised in minimal environments
    sf = None
    torch = None
    EchoSetDataset = None
    _DEPENDENCY_ERROR = exc
else:
    _DEPENDENCY_ERROR = None


@unittest.skipIf(
    EchoSetDataset is None,
    f"EchoSet runtime dependencies are not installed: {_DEPENDENCY_ERROR}",
)
class TestEchoSetDataset(unittest.TestCase):
    SAMPLE_RATE = 8000

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "EchoSet"
        self.train_dir = self.root / "train"
        self._write_example("scene_a/room_a/pair_a", 160)
        self._write_example("scene_b/room_b/pair_b", 240)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_example(self, relative_dir: str, length: int) -> None:
        sample_dir = self.train_dir / relative_dir
        sample_dir.mkdir(parents=True)
        time = np.arange(length, dtype=np.float32) / self.SAMPLE_RATE
        source_one = 0.1 * np.sin(2 * np.pi * 200 * time).astype(np.float32)
        source_two = 0.1 * np.sin(2 * np.pi * 350 * time).astype(np.float32)
        sf.write(sample_dir / "spk1_reverb.wav", source_one, self.SAMPLE_RATE)
        sf.write(sample_dir / "spk2_reverb.wav", source_two, self.SAMPLE_RATE)
        sf.write(sample_dir / "mix.wav", source_one + source_two, self.SAMPLE_RATE)

    def test_split_root_and_fixed_crop(self) -> None:
        dataset = EchoSetDataset(
            self.root,
            split="train",
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=False,
            return_key=True,
        )
        self.assertEqual(len(dataset), 2)
        mix, sources, key = dataset[0]
        self.assertEqual(tuple(mix.shape), (80,))
        self.assertEqual(tuple(sources.shape), (2, 80))
        self.assertEqual(key, "scene_a/room_a/pair_a")

    def test_full_utterance_collate_preserves_lengths(self) -> None:
        dataset = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=None,
            return_key=True,
        )
        mixes, sources, lengths, keys = dataset.collate_fn_with_lengths(
            [dataset[0], dataset[1]]
        )
        self.assertEqual(tuple(mixes.shape), (2, 240))
        self.assertEqual(tuple(sources.shape), (2, 2, 240))
        torch.testing.assert_close(lengths, torch.tensor([160, 240]))
        self.assertEqual(
            keys,
            ["scene_a/room_a/pair_a", "scene_b/room_b/pair_b"],
        )
        self.assertTrue(torch.equal(mixes[0, 160:], torch.zeros(80)))

    def test_seeded_random_crop_is_sample_key_and_epoch_deterministic(self) -> None:
        first = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=True,
            crop_seed=31415,
            return_key=True,
        )
        second = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=True,
            crop_seed=31415,
            return_key=True,
        )

        first.set_epoch(7)
        second.set_epoch(7)
        expected_by_key = {
            first[index][2]: first[index][0].clone()
            for index in range(len(first))
        }

        # Simulate a different sampler order and unrelated process RNG state.
        second.samples.reverse()
        np.random.seed(999)
        actual_by_key = {
            second[index][2]: second[index][0].clone()
            for index in range(len(second))
        }
        self.assertEqual(expected_by_key.keys(), actual_by_key.keys())
        for key in expected_by_key:
            torch.testing.assert_close(expected_by_key[key], actual_by_key[key])

        starts_at_epoch_7 = {
            sample["key"]: first._crop_bounds(sample)[0]
            for sample in first.samples
        }
        first.set_epoch(8)
        starts_at_epoch_8 = {
            sample["key"]: first._crop_bounds(sample)[0]
            for sample in first.samples
        }
        self.assertTrue(
            any(
                starts_at_epoch_7[key] != starts_at_epoch_8[key]
                for key in starts_at_epoch_7
            )
        )

    def test_crop_manifest_is_stable_across_dataset_order(self) -> None:
        first = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=True,
            crop_seed=43,
        )
        second = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=True,
            crop_seed=43,
        )
        second.samples.reverse()

        first_digest = first.crop_manifest_sha256(epoch=3)
        second_digest = second.crop_manifest_sha256(epoch=3)
        self.assertEqual(first_digest, second_digest)
        self.assertEqual(len(first_digest), 64)
        manifest = first.crop_manifest(epoch=3)
        self.assertEqual(manifest["epoch"], 3)
        self.assertEqual(
            [record["index"] for record in manifest["samples"]],
            [0, 1],
        )
        self.assertTrue(
            all(
                {"index", "key", "length", "start", "stop"}
                == set(record)
                for record in manifest["samples"]
            )
        )
        self.assertNotEqual(
            first_digest,
            first.crop_manifest_sha256(epoch=4),
        )

    def test_unseeded_random_crop_preserves_legacy_numpy_behavior(self) -> None:
        dataset = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=True,
        )
        with mock.patch("dataloader_echoset.np.random.randint", return_value=7) as draw:
            dataset[0]
        draw.assert_called_once_with(0, 81)
        with self.assertRaisesRegex(RuntimeError, "crop_seed"):
            dataset.crop_manifest_sha256()

    def test_nonrandom_crop_ignores_crop_seed_and_epoch(self) -> None:
        dataset = EchoSetDataset(
            self.train_dir,
            sample_rate=self.SAMPLE_RATE,
            segment=0.01,
            random_start=False,
            crop_seed=1,
        )
        epoch_zero = dataset[0][0].clone()
        dataset.set_epoch(99)
        torch.testing.assert_close(epoch_zero, dataset[0][0])


if __name__ == "__main__":
    unittest.main()
