"""End-to-end smoke test of the documented training workflow on CPU.

Runs ``train.py`` for one epoch on a tiny synthetic EchoSet tree, resumes the
same run for a second epoch, then separates a mixture with ``inference.py``
from the trained checkpoint. Unit tests exercise the model in isolation; this
catches breakage in the trainer itself, such as a snapshot step that copies a
directory which no longer exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import torch  # noqa: F401 - the trainer subprocess needs it
    from omegaconf import OmegaConf
except ImportError as exc:  # pragma: no cover - exercised in minimal environments
    OmegaConf = None
    _DEPENDENCY_ERROR = exc
else:
    _DEPENDENCY_ERROR = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


def _write_echoset(root: Path) -> None:
    rng = np.random.default_rng(0)
    for split, count in (("train", 4), ("val", 2), ("test", 1)):
        for index in range(count):
            folder = root / split / "scene" / f"utt{index}"
            folder.mkdir(parents=True)
            speakers = 0.1 * rng.standard_normal((2, int(1.5 * SAMPLE_RATE)))
            speakers = speakers.astype("float32")
            sf.write(folder / "spk1_reverb.wav", speakers[0], SAMPLE_RATE)
            sf.write(folder / "spk2_reverb.wav", speakers[1], SAMPLE_RATE)
            sf.write(folder / "mix.wav", speakers.sum(axis=0), SAMPLE_RATE)


@unittest.skipIf(
    OmegaConf is None,
    f"training runtime dependencies are not installed: {_DEPENDENCY_ERROR}",
)
class TestTrainingWorkflow(unittest.TestCase):
    def _run(self, *args: str, env: dict) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            [sys.executable, *args],
            cwd=REPOSITORY_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            self.fail(
                f"{' '.join(args[:1])} exited with {completed.returncode}\n"
                f"--- stdout (tail) ---\n{completed.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{completed.stderr[-3000:]}"
            )
        return completed

    def test_train_resume_and_infer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _write_echoset(work / "EchoSet")
            env = dict(os.environ, ECHOSET_ROOT=str(work / "EchoSet"))

            config = OmegaConf.load(REPOSITORY_ROOT / "configs/seal_small_echoset.yaml")
            config.trainer.epochs = 1
            config.trainer.exp_path = str(work / "exp" / "smoke")
            config.train_dataset.segment = 1.0
            config.train_dataloader.batch_size = 2
            config.train_dataloader.pin_memory = False
            config.validation_dataloader.pin_memory = False
            first = work / "first.yaml"
            OmegaConf.save(config, first)

            self._run("train.py", "-C", str(first), "-D", "cpu", env=env)
            runs = sorted((work / "exp").glob("smoke_*"))
            self.assertEqual(len(runs), 1, runs)
            checkpoints = runs[0] / "checkpoints"
            self.assertTrue((checkpoints / "best_model.tar").is_file())

            config.trainer.epochs = 2
            config.trainer.resume = True
            config.trainer.resume_datetime = runs[0].name[len("smoke_") :]
            second = work / "second.yaml"
            OmegaConf.save(config, second)
            self._run("train.py", "-C", str(second), "-D", "cpu", env=env)
            self.assertTrue((checkpoints / "model_002.tar").is_file())

            mixture = next((work / "EchoSet" / "test").rglob("mix.wav"))
            output = work / "separated"
            self._run(
                "inference.py",
                "--config", str(first),
                "--checkpoint", str(checkpoints / "best_model.tar"),
                "--trust-checkpoint",
                "--device", "cpu",
                "--audio", str(mixture),
                "--output-dir", str(output),
                env=env,
            )
            for name in ("source1.wav", "source2.wav"):
                separated, rate = sf.read(output / name)
                self.assertEqual(rate, SAMPLE_RATE)
                self.assertEqual(separated.shape, (int(1.5 * SAMPLE_RATE),))
                self.assertTrue(np.isfinite(separated).all())


if __name__ == "__main__":
    unittest.main()
