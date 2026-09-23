import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# Keep imports independent of the checkout directory name.
while str(REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from seal.data.dataloader_echoset import EchoSetDataset


from seal.losses.loss_ss import (
    PITAuxBalanceWrapper,
    PITHybridLoss,
    PITSISNRLoss,
    PITSNRLoss,
    PITTigerLoss,
)
from seal.metrics.metrics_ss import pit_si_sdr, separation_metrics, si_sdr, snr

try:
    from seal.utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR
except ImportError:
    WarmupLR = None

try:
    from seal.utils.distributed_utils import reduce_value
except ImportError:
    def reduce_value(value):
        return value


SEED = 43
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def state_dict_sha256(
    module: torch.nn.Module,
    *,
    canonicalize_adapter_sources: bool = False,
) -> str:
    """Hash a module's complete tensor state with names, dtypes and shapes."""

    digest = hashlib.sha256()
    entries = []
    seen_names = set()
    for original_name, tensor in module.state_dict().items():
        name = original_name
        if canonicalize_adapter_sources:
            while ".source." in name:
                name = name.replace(".source.", ".")
        if name in seen_names:
            raise RuntimeError(f"canonical state name collision: {name!r}")
        seen_names.add(name)
        entries.append((name, tensor))
    for name, tensor in sorted(entries):
        if not torch.is_tensor(tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        name_bytes = name.encode("utf-8")
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        value = tensor.detach().contiguous().cpu()
        value_bytes = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for payload in (name_bytes, metadata, value_bytes):
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
    return digest.hexdigest()


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Partition evaluation indices across ranks without padding duplicates."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be within [0, num_replicas)")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas


class EpochRandomSampler(torch.utils.data.Sampler):
    """Single-rank shuffle whose order is a pure function of seed and epoch."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_dataloader_worker(_worker_id):
    """Seed Python/NumPy from the worker seed assigned by DataLoader."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_model_class(config):
    model_cfg = config["model"]
    module_name = model_cfg["module"]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if module_name.startswith("models."):
            local_model = REPOSITORY_ROOT / (module_name.split(".", 1)[1].replace(".", "/") + ".py")
            if local_model.exists():
                spec = importlib.util.spec_from_file_location(module_name, local_model)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                assert spec.loader is not None
                spec.loader.exec_module(module)
            else:
                raise
        else:
            raise
    return getattr(module, model_cfg["class"]), model_cfg.get("name", model_cfg["class"])


def compute_si_snr(estimate, target, eps=1e-8):
    """Historical zero-mean SI-SDR helper retained for old run parity."""

    return si_sdr(estimate, target, zero_mean=True, eps=eps)


def compute_snr(estimate, target, eps=1e-8):
    """Historical plain SNR helper (this is not BSS-SDR)."""

    return snr(estimate, target, eps=eps)


def unwrap_loss_output(loss_output):
    """
    Normalize different loss return styles into a single scalar tensor.

    Some PIT losses return:
    - loss
    - (loss, best_perm)
    - (loss, aux1, aux2, ...)
    """
    if torch.is_tensor(loss_output):
        return loss_output

    if isinstance(loss_output, (tuple, list)):
        for item in loss_output:
            if torch.is_tensor(item):
                return item
        raise TypeError("Loss output tuple/list does not contain a tensor loss.")

    raise TypeError(f"Unsupported loss output type: {type(loss_output)!r}")


def pit_align_estimates(estimates, targets, *, zero_mean=True):
    """
    Align estimates using the historical zero-mean SI-SDR PIT assignment.

    New TIGER-parity metrics do not use this alignment: no-zero-mean SI-SDR
    and BSS-SDR each solve their own PIT assignment in ``metrics_ss``.
    """
    _, aligned, _ = pit_si_sdr(
        estimates,
        targets,
        zero_mean=zero_mean,
        return_aligned=True,
    )
    return aligned


def _run_worker(rank, config, args):
    if args.world_size > 1:
        torch.cuda.set_device(rank)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        if args.master_port is None:
            os.environ.setdefault("MASTER_PORT", "12354")
        else:
            os.environ["MASTER_PORT"] = str(args.master_port)
        backend = args.ddp_backend
        if backend == "auto":
            backend = "nccl" if dist.is_nccl_available() else "gloo"
        if backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError(
                "NCCL was requested but this PyTorch build has no NCCL support"
            )
        if backend == "gloo" and not dist.is_gloo_available():
            raise RuntimeError(
                "Gloo was requested but this PyTorch build has no Gloo support"
            )
        dist.init_process_group(backend, rank=rank, world_size=args.world_size)
        dist.barrier()

    args.rank = rank
    args.device = torch.device("cpu") if args.use_cpu else torch.device("cuda", rank)
    reproducibility_enabled = "reproducibility" in config
    reproducibility_config = config.get("reproducibility", {}) or {}
    init_seed = int(reproducibility_config.get("init_seed", SEED))
    sampler_seed = int(reproducibility_config.get("sampler_seed", SEED))
    crop_seed = int(reproducibility_config.get("crop_seed", SEED))
    worker_seed = int(reproducibility_config.get("worker_seed", SEED))
    runtime_seed = int(reproducibility_config.get("runtime_seed", SEED))
    if reproducibility_enabled:
        torch.backends.cudnn.deterministic = bool(
            reproducibility_config.get("cudnn_deterministic", True)
        )
        torch.backends.cudnn.benchmark = bool(
            reproducibility_config.get("cudnn_benchmark", False)
        )
    rank_seed = runtime_seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    shuffle = False if args.world_size > 1 else True

    dataset_type = str(config.get("dataset_type", "libri2mix")).lower()
    dataset_classes = {
        "echoset": EchoSetDataset,

    }
    try:
        DatasetClass = dataset_classes[dataset_type]
    except KeyError as exc:
        supported = ", ".join(sorted(dataset_classes))
        raise ValueError(
            f"Unsupported dataset_type {dataset_type!r}. Supported: {supported}"
        ) from exc

    train_dataset = DatasetClass(**config["train_dataset"])
    if reproducibility_enabled and hasattr(train_dataset, "set_crop_seed"):
        train_dataset.set_crop_seed(crop_seed)
    if args.world_size > 1:
        sampler_kwargs = {}
        if reproducibility_enabled:
            sampler_kwargs["seed"] = sampler_seed
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            **sampler_kwargs,
        )
    elif reproducibility_enabled:
        train_sampler = EpochRandomSampler(train_dataset, seed=sampler_seed)
    else:
        train_sampler = None

    train_loader_kwargs = dict(config["train_dataloader"])
    if (
        reproducibility_enabled
        and bool(train_loader_kwargs.get("persistent_workers", False))
        and hasattr(train_dataset, "set_epoch")
    ):
        raise ValueError(
            "Deterministic epoch-addressed crops require "
            "train_dataloader.persistent_workers=false"
        )
    if reproducibility_enabled:
        train_generator = torch.Generator()
        train_generator.manual_seed(worker_seed + rank)
        train_loader_kwargs.setdefault("generator", train_generator)
        train_loader_kwargs.setdefault("worker_init_fn", seed_dataloader_worker)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        **train_loader_kwargs,
        shuffle=shuffle if train_sampler is None else False,
        collate_fn=DatasetClass.collate_fn_with_lengths,
    )

    validation_dataset = DatasetClass(**config["validation_dataset"])
    if reproducibility_enabled and hasattr(validation_dataset, "set_crop_seed"):
        validation_dataset.set_crop_seed(crop_seed)
    if args.world_size > 1 and len(validation_dataset) < args.world_size:
        raise ValueError(
            "Validation dataset must contain at least one sample per DDP rank"
        )
    validation_sampler = (
        DistributedEvalSampler(
            validation_dataset,
            num_replicas=args.world_size,
            rank=rank,
        )
        if args.world_size > 1
        else None
    )
    validation_loader_kwargs = dict(config["validation_dataloader"])
    if reproducibility_enabled:
        validation_generator = torch.Generator()
        validation_generator.manual_seed(worker_seed + 10_000_000 + rank)
        validation_loader_kwargs.setdefault("generator", validation_generator)
        validation_loader_kwargs.setdefault("worker_init_fn", seed_dataloader_worker)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        **validation_loader_kwargs,
        shuffle=False,
        collate_fn=DatasetClass.collate_fn_with_lengths,
    )

    Model, model_name = load_model_class(config)
    if reproducibility_enabled:
        # All DDP ranks must start from identical parameters. Runtime randomness
        # is rank-specific and is restored immediately after construction.
        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
    model = Model(**config["network_config"]).to(args.device)
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if reproducibility_enabled:
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)

    optimizer = torch.optim.Adam(params=model.parameters(), **config["optimizer"])
    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "plateau")
    if scheduler_type == "plateau":
        monitor_mode = str(
            config.get("trainer", {}).get("monitor_mode", "min")
        ).lower()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            threshold=scheduler_config.get("threshold", 1e-4),
            threshold_mode=scheduler_config.get("threshold_mode", "abs"),
            cooldown=scheduler_config.get("cooldown", 0),
            min_lr=scheduler_config.get("min_lr", 0.0),
            eps=scheduler_config.get("eps", 1e-8),
        )
    elif scheduler_type == "cosine" and WarmupLR is not None:
        scheduler = WarmupLR(optimizer, **config["scheduler"]["kwargs"])
    else:
        scheduler = None

    loss_type = config.get("loss", {}).get("type", "sisnr")
    num_sources = config["network_config"].get("num_sources", 2)
    if loss_type == "sisnr":
        loss_func = PITSISNRLoss(num_sources=num_sources)
    elif loss_type == "snr":
        loss_func = PITSNRLoss(num_sources=num_sources)
    elif loss_type == "tiger":
        loss_kwargs = config.get("loss", {}).get("kwargs", {})
        loss_func = PITTigerLoss(
            num_sources=num_sources,
            use_snr=loss_kwargs.get("use_snr", True),
            lambda_snr=loss_kwargs.get("lambda_snr", 1.0),
            lambda_stft=loss_kwargs.get("lambda_stft", 0.0),
            use_freq_wav_loss=loss_kwargs.get("use_freq_wav_loss", True),
            lambda_freq_wav=loss_kwargs.get("lambda_freq_wav", 1.0),
            fft_sizes=loss_kwargs.get("fft_sizes", None),
            hop_sizes=loss_kwargs.get("hop_sizes", None),
            win_sizes=loss_kwargs.get("win_sizes", None),
        )
    else:
        loss_func = PITHybridLoss(num_sources=num_sources, **config.get("loss", {}).get("kwargs", {}))

    # Keep the selected PIT objective unchanged and optionally add model-side
    # auxiliary terms. `_get_balance_loss()` must return an unweighted scalar:
    # PITAuxBalanceWrapper applies `balance_loss_weight` exactly once. By
    # contrast, `_get_routing_aux_loss()` is treated as an already-weighted
    # additive term (for example latent-atom regularizers).
    wrapper_cfg = config.get("loss", {}).get("wrapper", {})
    if wrapper_cfg and wrapper_cfg.get("enabled", True):
        wrapper_type = str(wrapper_cfg.get("type", "pit_aux_balance")).lower()
        if wrapper_type not in {"pit_aux_balance", "pitauxbalancewrapper"}:
            raise ValueError(f"Unsupported loss wrapper: {wrapper_type!r}")
        loss_func = PITAuxBalanceWrapper(
            loss_func,
            balance_loss_weight=float(wrapper_cfg.get("balance_loss_weight", 0.01)),
        )
        # Attach the object used for forward(). The wrapper resolves `.module`
        # itself when this is DistributedDataParallel, and also works directly
        # with the non-DDP model.
        loss_func.attach_model(model)

    trainer = Trainer(
        config=config,
        model=model,
        model_name=model_name,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        train_sampler=train_sampler,
        args=args,
    )
    trainer.train()

def run(rank, config, args):
    """Run one training rank and always release its process group."""

    try:
        return _run_worker(rank, config, args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        config,
        model,
        model_name,
        optimizer,
        scheduler,
        loss_func,
        train_dataloader,
        validation_dataloader,
        train_sampler,
        args,
    ):
        self.config = config
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size
        self.num_sources = config["network_config"].get("num_sources", 2)

        self.trainer_config = config["trainer"]
        self.epochs = self.trainer_config["epochs"]
        self.save_checkpoint_interval = self.trainer_config["save_checkpoint_interval"]
        self.clip_grad_norm_value = self.trainer_config["clip_grad_norm_value"]
        self.resume = self.trainer_config["resume"]
        self.early_stop_patience = self.trainer_config.get("early_stop_patience", 40)
        self.monitor_name = str(
            self.trainer_config.get("monitor", "val_loss")
        ).lower()
        self.monitor_mode = str(
            self.trainer_config.get("monitor_mode", "min")
        ).lower()
        self.monitor_min_delta = float(
            self.trainer_config.get("min_delta", 0.0)
        )
        supported_monitors = {
            "val_loss",
            "val_base_pit",
            "val_tiger_si_sdri",
        }
        if self.monitor_name not in supported_monitors:
            raise ValueError(
                f"Unsupported trainer.monitor={self.monitor_name!r}; "
                f"choose one of {sorted(supported_monitors)}"
            )
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("trainer.monitor_mode must be 'min' or 'max'")
        if self.monitor_min_delta < 0.0:
            raise ValueError("trainer.min_delta must be non-negative")

        self.reproducibility_enabled = "reproducibility" in config
        self.reproducibility_config = config.get("reproducibility", {}) or {}
        self.sampler_seed = int(
            self.reproducibility_config.get("sampler_seed", SEED)
        )
        self.worker_seed = int(
            self.reproducibility_config.get("worker_seed", SEED)
        )
        self.write_crop_manifest_hash = bool(
            self.reproducibility_config.get("write_crop_manifest_hash", False)
        )
        self.write_crop_manifest = bool(
            self.reproducibility_config.get("write_crop_manifest", False)
        )
        self.latest_crop_manifest_sha256 = None
        self.no_improve_count = 0
        self.global_step = 0
        self.run_id = getattr(
            args,
            "run_id",
            (
                datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss-%f")
                + "-"
                + uuid.uuid4().hex[:8]
            ),
        )
        self.config_yaml = OmegaConf.to_yaml(OmegaConf.create(config), resolve=True)
        self.config_sha256 = hashlib.sha256(self.config_yaml.encode("utf-8")).hexdigest()
        target_model = self.model.module if self.world_size > 1 else self.model
        self.constructed_initialization_sha256 = state_dict_sha256(target_model)
        self.constructed_paired_initialization_sha256 = state_dict_sha256(
            target_model,
            canonicalize_adapter_sources=True,
        )
        self.initialization_sha256 = (
            None if self.resume else self.constructed_initialization_sha256
        )
        self.paired_initialization_sha256 = (
            None
            if self.resume
            else self.constructed_paired_initialization_sha256
        )

        if not self.resume:
            self.exp_path = self.trainer_config["exp_path"] + "_" + self.run_id
        else:
            resume_dt = self.trainer_config.get("resume_datetime")
            if (resume_dt is None) or (str(resume_dt).strip() == ""):
                raise ValueError("trainer.resume is True but 'trainer.resume_datetime' is not set.")
            self.exp_path = self.trainer_config["exp_path"] + "_" + resume_dt

        self.log_path = os.path.join(self.exp_path, "logs")
        self.checkpoint_path = os.path.join(self.exp_path, "checkpoints")
        self.sample_path = os.path.join(self.exp_path, "val_samples")
        self.code_path = os.path.join(self.exp_path, "codes")
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        if self.rank == 0:
            data = OmegaConf.create(config)
            if self.resume:
                snapshot_root = Path(self.code_path) / f"resume_{self.run_id}"
                config_path = Path(self.exp_path) / f"config_resume_{self.run_id}.yaml"
                trainer_copy = Path(self.exp_path) / f"train_resume_{self.run_id}.py"
            else:
                snapshot_root = Path(self.code_path)
                config_path = Path(self.exp_path) / "config.yaml"
                trainer_copy = Path(self.exp_path) / "train.py"

            snapshot_root.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(data, config_path)
            shutil.copy2(__file__, trainer_copy)
            # Snapshot the package and configs so a checkpoint never outlives
            # the exact code that produced it.
            for package_dir in ("seal", "configs"):
                shutil.copytree(
                    REPOSITORY_ROOT / package_dir,
                    snapshot_root / package_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for dependency_name in (
                "requirements.txt",
                "pyproject.toml",
            ):
                dependency_path = REPOSITORY_ROOT / dependency_name
                if dependency_path.exists():
                    shutil.copy2(dependency_path, snapshot_root / dependency_name)

            def git_output(*git_args):
                try:
                    result = subprocess.run(
                        ["git", *git_args],
                        cwd=REPOSITORY_ROOT,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=10,
                        check=False,
                    )
                    return result.stdout.strip() if result.returncode == 0 else None
                except (OSError, subprocess.SubprocessError):
                    return None

            git_diff = git_output("diff", "--binary")
            if git_diff:
                (snapshot_root / "git_diff.patch").write_text(git_diff, encoding="utf-8")
            runtime_manifest = {
                "run_id": self.run_id,
                "resume": bool(self.resume),
                "command": [sys.executable, *sys.argv],
                "config_sha256": self.config_sha256,
                "constructed_initialization_sha256": (
                    self.constructed_initialization_sha256
                ),
                "constructed_paired_initialization_sha256": (
                    self.constructed_paired_initialization_sha256
                ),
                "initialization_sha256": self.initialization_sha256,
                "paired_initialization_sha256": (
                    self.paired_initialization_sha256
                ),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "ddp_backend": (
                    dist.get_backend()
                    if dist.is_available() and dist.is_initialized()
                    else None
                ),
                "git_commit": git_output("rev-parse", "HEAD"),
                "git_status": git_output("status", "--short"),
            }
            (snapshot_root / "runtime_manifest.json").write_text(
                json.dumps(runtime_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.writer = SummaryWriter(self.log_path)
            self.writer.add_text("Config/Network", str(config.get("network_config", "")))

            net_cfg = config.get("network_config", {})
            target_model = self.model.module if self.world_size > 1 else self.model
            total_params = sum(p.numel() for p in target_model.parameters())
            trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
            sr = config.get("sample_rate", 16000)
            try:
                from ptflops import get_model_complexity_info

                def input_constructor(input_res):
                    return {"x": torch.randn(1, *input_res, device=self.device)}

                macs, _ = get_model_complexity_info(
                    target_model,
                    (sr,),
                    as_strings=False,
                    input_constructor=input_constructor,
                    print_per_layer_stat=False,
                    verbose=False,
                )
                macs_str = f"{macs / 1e6:.2f} M/s"
            except Exception as e:
                macs_str = f"N/A ({e})"

            print("\n" + "=" * 72)
            print(f" {self.model_name} ")
            print("=" * 72)
            for k, v in net_cfg.items():
                print(f"  - {k:20s}: {v}")
            print("-" * 72)
            print(f"  - Params              : {total_params:,} ({total_params / 1e6:.4f} M)")
            print(f"  - Trainable Params    : {trainable_params:,} ({trainable_params / 1e6:.4f} M)")
            print(f"  - MACs                : {macs_str}")
            print("=" * 72 + "\n")

        self.start_epoch = 1
        self.best_score = -float("inf")
        self.best_val_loss = float("inf")
        self.best_monitor_value = (
            float("inf") if self.monitor_mode == "min" else -float("inf")
        )
        self.latest_validation_stats = {}
        self.latest_peak_training_memory_gib = None
        self.state_dict_best = None
        self._gate_diagnostic_layers = None
        # A legacy opt-in model owns hard-load statistics as model buffers. This
        # trainer-side token prevents an accidental second controller update
        # for the same successful optimizer step (and remains future-proof if
        # gradient accumulation is introduced around ``optimizer.step``).
        self._last_m2_load_bias_update_step = None

    @staticmethod
    def _resolve_gate_diagnostics(module):
        """Return the nested object that owns the latest router diagnostics."""
        queue = [module]
        seen = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if (
                hasattr(candidate, "latest_expert_strength")
                or hasattr(candidate, "latest_distribution_stats")
                or callable(getattr(candidate, "get_diagnostics", None))
            ):
                return candidate
            for attr in ("temporal_moe", "temporal_ffn", "moe", "gate", "router"):
                nested = getattr(candidate, attr, None)
                if nested is not None:
                    queue.append(nested)
        return None

    def _iter_gate_layers(self):
        if self._gate_diagnostic_layers is not None:
            return self._gate_diagnostic_layers

        target_model = self.model.module if self.world_size > 1 else self.model
        candidates = []

        # Backward-compatible containers used by earlier DPGRNN variants.
        for attr in ("dp_blocks", "dpgrnn_layers", "blocks"):
            layers = getattr(target_model, attr, None)
            if layers is not None:
                candidates.extend(list(layers))

        # SEAL owns one shared recursive separator/cell. Its temporal MoE is
        # therefore visited only once here even though the cell is unrolled R
        # times during forward().
        separator = getattr(target_model, "separator", None)
        if separator is not None:
            candidates.append(separator)
            for attr in (
                "cell",
                "shared_cell",
                "block",
                "dpgrnn",
                "dpgrnn_block",
                "temporal_moe",
                "temporal_ffn",
            ):
                nested = getattr(separator, attr, None)
                if nested is not None:
                    candidates.append(nested)

        # Attribute names can evolve without silently losing diagnostics.
        candidates.extend(module for _, module in target_model.named_modules())

        diagnostic_layers = []
        seen = set()
        for candidate in candidates:
            diagnostics = self._resolve_gate_diagnostics(candidate)
            if diagnostics is None or id(diagnostics) in seen:
                continue
            seen.add(id(diagnostics))
            diagnostic_layers.append(diagnostics)
        self._gate_diagnostic_layers = diagnostic_layers
        return self._gate_diagnostic_layers

    def _collect_router_strengths(self, *, reduce_distributed=True):
        strengths = []
        for layer in self._iter_gate_layers():
            strength = getattr(layer, "latest_expert_strength", None)
            if strength is None:
                get_diagnostics = getattr(layer, "get_diagnostics", None)
                if callable(get_diagnostics):
                    strength = get_diagnostics().get("expert_load")
            if strength is None:
                strengths.append(None)
                continue
            # Clone before DDP all-reduce so diagnostics never mutate tensors
            # retained by the model's forward graph.
            strength = strength.detach().clone()
            if self.world_size > 1 and reduce_distributed:
                strength = torch.stack([reduce_value(v) for v in strength])
            strengths.append(strength)
        return strengths

    def _collect_gate_distribution_stats(self, *, reduce_distributed=True):
        stats_all = []
        for layer in self._iter_gate_layers():
            stats = getattr(layer, "latest_distribution_stats", None)
            if stats is None:
                get_diagnostics = getattr(layer, "get_diagnostics", None)
                if callable(get_diagnostics):
                    stats = get_diagnostics()
            if stats is None:
                stats_all.append(None)
                continue

            # TensorBoard's add_scalars expects scalar values. Preserve scalar
            # diagnostics and expand expert/atom vectors into stable keys.
            detached = {}
            pending = [(str(key), value) for key, value in stats.items()]
            while pending:
                key, value = pending.pop(0)
                if value is None:
                    continue
                if isinstance(value, dict):
                    pending.extend(
                        (f"{key}/{nested_key}", nested_value)
                        for nested_key, nested_value in value.items()
                    )
                    continue
                if not torch.is_tensor(value):
                    try:
                        value = torch.as_tensor(value, device=self.device)
                    except (TypeError, ValueError):
                        continue
                # Clone before DDP all-reduce so diagnostics never mutate
                # model-side auxiliary tensors needed by backward().
                value = value.detach().clone()
                # DDP's reduce_value averages in-place, so metadata such as
                # bool/int must be promoted before collective reduction.
                if not value.is_floating_point() and not value.is_complex():
                    value = value.float()
                if value.numel() == 1:
                    detached[str(key)] = value.reshape(())
                elif value.numel() <= 64:
                    for value_idx, scalar in enumerate(value.reshape(-1)):
                        detached[f"{key}_{value_idx}"] = scalar
                else:
                    # Never emit a TensorBoard scalar per TF-bin if a model
                    # accidentally exposes a full activation tensor.
                    value_float = value.float()
                    detached[f"{key}/mean"] = value_float.mean()
                    detached[f"{key}/std"] = value_float.std(unbiased=False)
                    detached[f"{key}/min"] = value_float.min()
                    detached[f"{key}/max"] = value_float.max()
            if not detached:
                stats_all.append(None)
                continue
            if self.world_size > 1 and reduce_distributed:
                detached = {k: reduce_value(v) for k, v in detached.items()}
            stats_all.append(detached)
        return stats_all

    def _reduce_epoch_diagnostics(self, router_strengths, gate_stats):
        """Average all accumulated diagnostic tensors with one collective.

        Diagnostics are logging-only values.  Packing them avoids issuing one
        all-reduce per scalar on every training step, which is especially
        expensive for SEAL's per-refinement router statistics.
        """

        if self.world_size <= 1:
            return
        tensors = []
        if router_strengths is not None:
            tensors.extend(value for value in router_strengths if value is not None)
        if gate_stats is not None:
            for stats in gate_stats:
                if stats is not None:
                    tensors.extend(stats.values())
        if not tensors:
            return

        packed = torch.cat([value.detach().float().reshape(-1) for value in tensors])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(self.world_size)

        offset = 0
        with torch.no_grad():
            for value in tensors:
                count = value.numel()
                reduced = packed[offset : offset + count].reshape(value.shape)
                value.copy_(reduced.to(dtype=value.dtype))
                offset += count

    @staticmethod
    def _capture_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def _gather_rng_states(self):
        local_state = self._capture_rng_state()
        if self.world_size <= 1:
            return [local_state]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_state)
        return gathered

    def _restore_rng_state(self, rng_states):
        if not rng_states:
            return
        if len(rng_states) != self.world_size:
            raise ValueError(
                f"Checkpoint has {len(rng_states)} RNG rank states, "
                f"but this run uses world_size={self.world_size}"
            )
        state = rng_states[self.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda"):
            if len(state["torch_cuda"]) != torch.cuda.device_count():
                raise ValueError(
                    "Checkpoint CUDA RNG topology does not match visible CUDA devices"
                )
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    @staticmethod
    def _atomic_torch_save(state, destination):
        destination = Path(destination)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _checkpoint_contract(self):
        resolved = OmegaConf.to_container(
            OmegaConf.create(self.config),
            resolve=True,
        )
        contract_keys = (
            "sample_rate",
            "dataset_type",
            "model",
            "network_config",
            "loss",
            "train_dataset",
            "train_dataloader",
            "validation_dataset",
            "validation_dataloader",
            "optimizer",
            "scheduler",
            "reproducibility",
        )
        contract = {key: resolved.get(key) for key in contract_keys}
        trainer = resolved.get("trainer") or {}
        # Epoch count is intentionally excluded so a completed run can be
        # extended.  Resume location and checkpoint cadence do not alter the
        # optimization trajectory, while clipping and early stopping do.
        contract["trainer_optimization"] = {
            "clip_grad_norm_value": trainer.get("clip_grad_norm_value"),
            "early_stop_patience": trainer.get("early_stop_patience", 40),
            "early_stop_monitor": trainer.get("monitor", "val_loss"),
            "monitor_mode": trainer.get("monitor_mode", "min"),
            "min_delta": trainer.get("min_delta", 0.0),
        }
        return contract

    @staticmethod
    def _canonicalize_checkpoint_contract(contract):
        """Add defaults introduced after the earliest trainer checkpoints."""

        canonical = dict(contract)
        canonical.setdefault("reproducibility", None)
        trainer_optimization = dict(
            canonical.get("trainer_optimization") or {}
        )
        trainer_optimization.setdefault("early_stop_monitor", "val_loss")
        trainer_optimization.setdefault("monitor_mode", "min")
        trainer_optimization.setdefault("min_delta", 0.0)
        canonical["trainer_optimization"] = trainer_optimization
        return canonical

    def _current_step_bound_report(self):
        target_model = self.model.module if self.world_size > 1 else self.model
        reporter = getattr(target_model, "step_bound_report", None)
        return reporter() if callable(reporter) else None

    def _record_step_bound_epoch(self, epoch):
        """Append an auditable clipping record after each validation epoch."""

        if self.rank != 0:
            return
        report = self._current_step_bound_report()
        if report is None:
            return
        record = {"epoch": int(epoch), **report}
        destination = Path(self.exp_path) / "step_bound_history.jsonl"
        records = {}
        if destination.is_file():
            for line in destination.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    previous = json.loads(line)
                    records[int(previous["epoch"])] = previous
        records[int(epoch)] = record
        rendered = "".join(
            json.dumps(records[key], ensure_ascii=False) + "\n"
            for key in sorted(records)
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        # ``Path.write_text(..., newline=...)`` is unavailable on Python 3.9,
        # which is still used by the laboratory training environment.
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        os.replace(temporary, destination)

    def _save_checkpoint(
        self,
        epoch,
        val_loss,
        score,
        *,
        monitor_value=None,
        is_best=False,
        save_regular=True,
    ):
        if monitor_value is None:
            monitor_value = self._resolve_monitor_value(val_loss, score)
        monitor_value = float(monitor_value)
        if is_best:
            self.best_monitor_value = monitor_value
        best_epoch = epoch if is_best else (
            self.state_dict_best["epoch"] if self.state_dict_best is not None else None
        )
        state = {
            "model": self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "score": score,
            "best_score": self.best_score,
            "best_val_loss": self.best_val_loss,
            "monitor_name": self.monitor_name,
            "monitor_mode": self.monitor_mode,
            "monitor_value": monitor_value,
            "best_monitor_value": self.best_monitor_value,
            "best_epoch": best_epoch,
            "no_improve_count": self.no_improve_count,
            "global_step": self.global_step,
            "latest_crop_manifest_sha256": self.latest_crop_manifest_sha256,
            "initialization_sha256": self.initialization_sha256,
            "paired_initialization_sha256": self.paired_initialization_sha256,
            "rng_states": self._gather_rng_states(),
            "contract": self._checkpoint_contract(),
            "config_sha256": self.config_sha256,
            "world_size": self.world_size,
            "cuda_device_count": torch.cuda.device_count(),
            "step_bound_report": self._current_step_bound_report(),
            "peak_training_memory_gib": self.latest_peak_training_memory_gib,
        }

        if self.rank == 0 and save_regular:
            self._atomic_torch_save(
                state,
                Path(self.checkpoint_path) / f"model_{str(epoch).zfill(3)}.tar",
            )

        if is_best:
            self.state_dict_best = {"epoch": epoch}
            if self.rank == 0:
                self._atomic_torch_save(
                    state,
                    Path(self.checkpoint_path) / "best_model.tar",
                )
                self._atomic_torch_save(
                    state,
                    Path(self.checkpoint_path) / f"best_model_{str(epoch).zfill(3)}.tar",
                )
                print(
                    f"New best model at epoch {epoch}: "
                    f"{self.monitor_name} = {monitor_value:.6f}, "
                    f"validation loss = {val_loss:.6f}, "
                    f"SI-SDRi = {score:.4f} dB"
                )

    def _resume_checkpoint(self):
        checkpoint_root = Path(self.checkpoint_path)
        ckpts = list(checkpoint_root.glob("model_*.tar"))
        ckpts.extend(checkpoint_root.glob("best_model_[0-9]*.tar"))
        if not ckpts:
            best_alias = checkpoint_root / "best_model.tar"
            if best_alias.exists():
                ckpts.append(best_alias)
        if not ckpts:
            raise FileNotFoundError(
                f"trainer.resume=True but no checkpoint was found in {checkpoint_root}"
            )

        def checkpoint_epoch(path):
            matches = re.findall(r"(\d+)", path.stem)
            return int(matches[-1]) if matches else -1

        state = None
        latest = None
        load_errors = []
        for candidate in sorted(ckpts, key=checkpoint_epoch, reverse=True):
            try:
                state = torch.load(
                    candidate,
                    map_location="cpu",
                    weights_only=False,
                )
                latest = candidate
                break
            except Exception as error:
                load_errors.append(f"{candidate.name}: {error}")
        if state is None or latest is None:
            raise RuntimeError(
                "No readable checkpoint could be loaded. " + " | ".join(load_errors)
            )

        saved_contract = state.get("contract")
        current_contract = self._checkpoint_contract()
        if saved_contract is not None:
            saved_contract = self._canonicalize_checkpoint_contract(
                saved_contract
            )
            current_contract = self._canonicalize_checkpoint_contract(
                current_contract
            )
        if saved_contract is not None and saved_contract != current_contract:
            raise ValueError(
                "Resume config changes the immutable training contract. "
                "Use the original contract or start a new experiment."
            )
        if saved_contract is None and self.rank == 0:
            warnings.warn(
                "Checkpoint predates contract metadata; architecture compatibility "
                "can only be checked by load_state_dict.",
                RuntimeWarning,
            )
        saved_world_size = state.get("world_size")
        if saved_world_size is not None and int(saved_world_size) != self.world_size:
            raise ValueError(
                f"Checkpoint world_size={saved_world_size}, current world_size={self.world_size}"
            )
        saved_cuda_devices = state.get("cuda_device_count")
        if (
            saved_cuda_devices is not None
            and int(saved_cuda_devices) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Checkpoint visible CUDA device count does not match the current run"
            )

        if self.world_size > 1:
            self.model.module.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
            scheduler_config = self.config.get("scheduler", {})
            if scheduler_config.get("type", "plateau") == "plateau":
                # Loading a checkpoint restores the old threshold policy too.
                # Explicit config values must win so a deliberately migrated
                # checkpoint cannot silently revert to relative comparisons,
                # which are unsafe for this run's negative validation loss.
                if "threshold" in scheduler_config:
                    self.scheduler.threshold = float(scheduler_config["threshold"])
                if "threshold_mode" in scheduler_config:
                    self.scheduler.threshold_mode = str(
                        scheduler_config["threshold_mode"]
                    )
        self.start_epoch = state["epoch"] + 1
        self.best_score = state.get("best_score", -float("inf"))
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if (
            "best_monitor_value" not in state
            and self.monitor_name != "val_loss"
            and state.get("best_epoch") is not None
        ):
            raise ValueError(
                "Checkpoint predates custom-monitor state and cannot safely "
                f"resume trainer.monitor={self.monitor_name!r}"
            )
        fallback_best_monitor = (
            self.best_val_loss
            if self.monitor_name == "val_loss"
            else (
                float("inf")
                if self.monitor_mode == "min"
                else -float("inf")
            )
        )
        self.best_monitor_value = state.get(
            "best_monitor_value",
            fallback_best_monitor,
        )
        self.no_improve_count = state.get("no_improve_count", 0)
        self.global_step = state.get("global_step", 0)
        self.latest_crop_manifest_sha256 = state.get(
            "latest_crop_manifest_sha256"
        )
        saved_initialization_sha256 = state.get("initialization_sha256")
        if saved_initialization_sha256 is None:
            self.initialization_sha256 = None
            if self.rank == 0:
                warnings.warn(
                    "Checkpoint predates initialization hashing; resumed "
                    "checkpoints cannot be formal protocol evidence.",
                    RuntimeWarning,
                )
        else:
            rendered_initialization = str(saved_initialization_sha256).lower()
            if len(rendered_initialization) != 64 or any(
                character not in "0123456789abcdef"
                for character in rendered_initialization
            ):
                raise RuntimeError(
                    "checkpoint has an invalid initialization SHA-256"
                )
            self.initialization_sha256 = rendered_initialization
        saved_paired_initialization_sha256 = state.get(
            "paired_initialization_sha256"
        )
        if saved_paired_initialization_sha256 is None:
            self.paired_initialization_sha256 = None
            if self.rank == 0:
                warnings.warn(
                    "Checkpoint predates paired-initialization hashing; "
                    "resumed checkpoints cannot be formal protocol evidence.",
                    RuntimeWarning,
                )
        else:
            rendered_paired_initialization = str(
                saved_paired_initialization_sha256
            ).lower()
            if len(rendered_paired_initialization) != 64 or any(
                character not in "0123456789abcdef"
                for character in rendered_paired_initialization
            ):
                raise RuntimeError(
                    "checkpoint has an invalid paired-initialization SHA-256"
                )
            self.paired_initialization_sha256 = (
                rendered_paired_initialization
            )
        best_epoch = state.get("best_epoch")
        if best_epoch is not None:
            self.state_dict_best = {"epoch": best_epoch}
        self._restore_rng_state(state.get("rng_states"))
        if self.rank == 0:
            resume_origin = {
                "resume_checkpoint": str(latest),
                "resume_checkpoint_sha256": hashlib.sha256(
                    latest.read_bytes()
                ).hexdigest(),
                "initialization_sha256": self.initialization_sha256,
                "paired_initialization_sha256": (
                    self.paired_initialization_sha256
                ),
                "latest_crop_manifest_sha256": (
                    self.latest_crop_manifest_sha256
                ),
            }
            (
                Path(self.exp_path) / f"resume_origin_{self.run_id}.json"
            ).write_text(
                json.dumps(resume_origin, indent=2),
                encoding="utf-8",
            )
            print(
                f"Resumed {latest.name} at epoch {state['epoch']} "
                f"(next epoch {self.start_epoch}, global_step {self.global_step})."
            )

    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _clear_model_aux(self):
        target_model = self.model.module if self.world_size > 1 else self.model
        clear_aux = getattr(target_model, "clear_aux", None)
        if callable(clear_aux):
            clear_aux()

    def _set_model_training_step(self):
        """Expose the checkpointed optimizer-update count to opt-in models."""

        target_model = self.model.module if self.world_size > 1 else self.model
        set_training_progress = getattr(
            target_model,
            "set_training_progress",
            None,
        )
        if callable(set_training_progress):
            total_training_steps = max(
                1,
                int(self.epochs) * len(self.train_dataloader),
            )
            set_training_progress(
                self.global_step,
                total_training_steps,
            )
            return
        set_training_step = getattr(target_model, "set_training_step", None)
        if callable(set_training_step):
            set_training_step(self.global_step)

    def _reset_m2_load_stats(self):
        """Discard stale optional hard-load counts before a new training epoch.

        An opt-in legacy controller accumulates hard assignments from training forwards in
        model buffers.  A normal controller update clears those buffers after
        every optimizer step; this epoch-boundary reset additionally protects
        resumed runs and guarantees that validation traffic can never leak
        into the next training update.
        """

        target_model = self.model.module if self.world_size > 1 else self.model
        reset_load_stats = getattr(target_model, "reset_m2_load_stats", None)
        if callable(reset_load_stats):
            reset_load_stats()

    def _update_m2_load_bias_after_optimizer_step(self):
        """Run an opt-in hard-load controller once per optimizer update.

        The model method owns count aggregation, DDP ``SUM`` reduction, the
        centered/clipped loss-free bias update, and clearing pending counts.
        Keeping this hook immediately after ``optimizer.step`` means future
        micro-batch accumulation contributes to one controller update instead
        of updating the bias once per forward/backward micro-batch.
        """

        target_model = self.model.module if self.world_size > 1 else self.model
        update_load_bias = getattr(target_model, "update_m2_load_bias", None)
        if not callable(update_load_bias):
            return None
        if not target_model.training:
            # Validation and inference must freeze the selection bias.
            return None
        if self._last_m2_load_bias_update_step == self.global_step:
            return None

        sync_ddp = bool(
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        update_stats = update_load_bias(sync_ddp=sync_ddp)
        self._last_m2_load_bias_update_step = self.global_step
        if update_stats is None:
            return {}
        if not isinstance(update_stats, dict):
            raise TypeError("update_m2_load_bias() must return a dict or None")

        detached = {}
        for key, value in update_stats.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                detached[str(key)] = value.detach().clone()
                continue
            try:
                detached[str(key)] = torch.as_tensor(value, device=self.device)
            except (TypeError, ValueError):
                continue
        return detached

    @staticmethod
    def _m2_load_stat(stats, *names):
        for name in names:
            value = stats.get(name)
            if torch.is_tensor(value):
                return value
        return None

    def _accumulate_m2_load_controller_stats(self, epoch_stats, update_stats):
        """Accumulate already-DDP-synchronized load-controller diagnostics."""

        if update_stats is None:
            return
        updated = self._m2_load_stat(update_stats, "updated", "bias_updated")
        if updated is not None and not bool(updated.reshape(()).item()):
            return
        epoch_stats["updates"] += 1

        counts = self._m2_load_stat(
            update_stats,
            "global_expert_counts",
            "expert_counts",
            "hard_counts",
        )
        if counts is not None:
            counts = counts.detach().float().reshape(-1)
            if epoch_stats["counts"] is None:
                epoch_stats["counts"] = torch.zeros_like(counts)
            if epoch_stats["counts"].shape != counts.shape:
                raise RuntimeError(
                    "load-controller expert-count shape changed within one epoch"
                )
            epoch_stats["counts"] += counts

        bias = self._m2_load_stat(
            update_stats,
            "load_bias",
            "hard_load_bias",
        )
        if bias is not None:
            epoch_stats["last_bias"] = bias.detach().float().reshape(-1)

    def _write_m2_load_controller_stats(self, epoch, epoch_stats):
        """Write exact aggregate hard load and the post-update bias."""

        if self.rank != 0 or epoch_stats["updates"] == 0:
            return
        counts = epoch_stats["counts"]
        if counts is not None and counts.numel() > 0:
            total = counts.sum()
            hard_fraction = counts / total.clamp_min(1.0)
            entropy = -(
                hard_fraction.clamp_min(1e-12)
                * hard_fraction.clamp_min(1e-12).log()
            ).sum()
            effective_experts = entropy.exp()
            load_mean = hard_fraction.mean()
            load_cv = hard_fraction.std(unbiased=False) / load_mean.clamp_min(
                1e-12
            )
            self.writer.add_scalars(
                "load_controller/hard_fraction",
                {
                    f"expert_{index}": value.item()
                    for index, value in enumerate(hard_fraction)
                },
                epoch,
            )
            self.writer.add_scalars(
                "load_controller/hard_counts",
                {
                    f"expert_{index}": value.item()
                    for index, value in enumerate(counts)
                },
                epoch,
            )
            self.writer.add_scalars(
                "load_controller/summary",
                {
                    "optimizer_updates": float(epoch_stats["updates"]),
                    "hard_tokens": total.item(),
                    "effective_experts": effective_experts.item(),
                    "load_cv": load_cv.item(),
                    "minimum_fraction": hard_fraction.min().item(),
                    "maximum_fraction": hard_fraction.max().item(),
                },
                epoch,
            )

        bias = epoch_stats["last_bias"]
        if bias is not None:
            self.writer.add_scalars(
                "load_controller/load_bias",
                {
                    f"expert_{index}": value.item()
                    for index, value in enumerate(bias)
                },
                epoch,
            )

    def _set_data_epoch(self, epoch):
        if self.train_sampler is not None:
            set_sampler_epoch = getattr(self.train_sampler, "set_epoch", None)
            if callable(set_sampler_epoch):
                set_sampler_epoch(epoch)

        train_dataset = getattr(self.train_dataloader, "dataset", None)
        set_dataset_epoch = getattr(train_dataset, "set_epoch", None)
        if callable(set_dataset_epoch):
            set_dataset_epoch(epoch)

        validation_dataset = getattr(self.validation_dataloader, "dataset", None)
        set_validation_epoch = getattr(validation_dataset, "set_epoch", None)
        if callable(set_validation_epoch):
            set_validation_epoch(epoch)

        if self.reproducibility_enabled:
            train_generator = getattr(self.train_dataloader, "generator", None)
            if train_generator is not None:
                train_generator.manual_seed(
                    self.worker_seed + epoch + self.rank * 1_000_003
                )
            validation_generator = getattr(
                self.validation_dataloader,
                "generator",
                None,
            )
            if validation_generator is not None:
                validation_generator.manual_seed(
                    self.worker_seed
                    + 10_000_000
                    + epoch
                    + self.rank * 1_000_003
                )

        self._record_crop_manifest(epoch)

    def _record_crop_manifest(self, epoch):
        if self.rank != 0 or not (
            self.write_crop_manifest_hash or self.write_crop_manifest
        ):
            return
        dataset = getattr(self.train_dataloader, "dataset", None)
        manifest_digest = getattr(dataset, "crop_manifest_sha256", None)
        if not callable(manifest_digest):
            raise RuntimeError(
                "write_crop_manifest_hash=True but the training dataset does "
                "not implement crop_manifest_sha256()"
            )
        digest = manifest_digest(epoch=epoch)
        self.latest_crop_manifest_sha256 = digest
        manifest_dir = Path(self.exp_path) / "crop_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "training-crop-manifest-v1",
            "epoch": int(epoch),
            "sha256": digest,
            "sampler_seed": self.sampler_seed,
            "worker_seed": self.worker_seed,
        }
        destination = manifest_dir / f"epoch_{epoch:04d}.json"
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        try:
            temporary.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        if self.write_crop_manifest:
            build_manifest = getattr(dataset, "crop_manifest", None)
            if not callable(build_manifest):
                raise RuntimeError(
                    "write_crop_manifest=True but the training dataset does "
                    "not implement crop_manifest()"
                )
            manifest = build_manifest(epoch=epoch)
            manifest_payload = {
                "manifest_sha256": digest,
                **manifest,
            }
            manifest_destination = (
                manifest_dir / f"epoch_{epoch:04d}.json.gz"
            )
            manifest_temporary = manifest_destination.with_name(
                f".{manifest_destination.name}.tmp-{os.getpid()}"
            )
            try:
                with gzip.open(
                    manifest_temporary,
                    mode="wt",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        manifest_payload,
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                os.replace(manifest_temporary, manifest_destination)
            finally:
                if manifest_temporary.exists():
                    manifest_temporary.unlink()

    def _resolve_monitor_value(self, val_loss, score):
        if self.monitor_name == "val_loss":
            return float(val_loss)
        if self.monitor_name == "val_tiger_si_sdri":
            return float(score)
        if self.monitor_name == "val_base_pit":
            if "val_base_pit" not in self.latest_validation_stats:
                raise RuntimeError(
                    "trainer.monitor='val_base_pit' requires a loss wrapper "
                    "that reports latest_components['base_pit']"
                )
            return float(self.latest_validation_stats["val_base_pit"])
        raise AssertionError(f"Unhandled monitor {self.monitor_name!r}")

    def _monitor_improved(self, monitor_value):
        if self.monitor_mode == "min":
            return (
                monitor_value
                < self.best_monitor_value - self.monitor_min_delta
            )
        return (
            monitor_value
            > self.best_monitor_value + self.monitor_min_delta
        )

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError(
                "Dataloader must return (mix, sources, lengths[, keys])"
            )
        mix, sources, lengths = batch[:3]
        if not torch.is_tensor(lengths) or lengths.ndim != 1:
            raise ValueError("Batch lengths must be a one-dimensional tensor")
        return mix, sources, lengths

    def _prepare_lengths(self, lengths, mix):
        lengths = lengths.to(self.device, dtype=torch.long)
        if lengths.numel() != mix.shape[0]:
            raise ValueError("Batch length count does not match batch size")
        if bool((lengths == mix.shape[-1]).all()):
            return None
        return lengths

    def _loss_for_lengths(self, estimates, sources, lengths):
        if lengths is None:
            return unwrap_loss_output(self.loss_func(estimates, sources))
        sample_losses = []
        component_sums = None
        for batch_index, sample_length in enumerate(lengths.tolist()):
            sample_losses.append(
                unwrap_loss_output(
                    self.loss_func(
                        estimates[batch_index : batch_index + 1, :, :sample_length],
                        sources[batch_index : batch_index + 1, :, :sample_length],
                    )
                )
            )
            latest_components = getattr(self.loss_func, "latest_components", None)
            if latest_components:
                if component_sums is None:
                    component_sums = {
                        key: value.detach() * 0.0
                        for key, value in latest_components.items()
                    }
                for key, value in latest_components.items():
                    component_sums[key] += value.detach()
        if component_sums is not None:
            self.loss_func.latest_components = {
                key: value / len(sample_losses)
                for key, value in component_sums.items()
            }
        return torch.stack(sample_losses).mean()

    def _model_loss(self, estimates, sources, lengths):
        if isinstance(estimates, tuple) and len(estimates) == 2:
            est_s1, est_s2 = estimates
            loss_s1 = self._loss_for_lengths(est_s1, sources, lengths)
            loss_s2 = self._loss_for_lengths(est_s2, sources, lengths)
            return 0.3 * loss_s1 + loss_s2
        return self._loss_for_lengths(estimates, sources, lengths)

    @staticmethod
    def _pit_align_with_lengths(estimates, sources, lengths, *, zero_mean=True):
        if lengths is None:
            return pit_align_estimates(estimates, sources, zero_mean=zero_mean)
        aligned = torch.zeros_like(estimates)
        for batch_index, sample_length in enumerate(lengths.tolist()):
            aligned[
                batch_index : batch_index + 1,
                :,
                :sample_length,
            ] = pit_align_estimates(
                estimates[batch_index : batch_index + 1, :, :sample_length],
                sources[batch_index : batch_index + 1, :, :sample_length],
                zero_mean=zero_mean,
            )
        return aligned

    def _train_epoch(self, epoch):
        self._reset_m2_load_stats()
        self.train_bar = tqdm(self.train_dataloader, disable=self.rank != 0)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        total_loss = 0
        epoch_router_strengths = None
        epoch_gate_stats = None
        epoch_loss_components = None
        epoch_m2_load_stats = {
            "updates": 0,
            "counts": None,
            "last_bias": None,
        }
        step = 0
        for step, batch in enumerate(self.train_bar, start=1):
            mix, sources, lengths = self._unpack_batch(batch)
            mix = mix.to(self.device)
            sources = sources.to(self.device)
            lengths = self._prepare_lengths(lengths, mix)
            self.optimizer.zero_grad()
            self._set_model_training_step()
            estimates = self.model(mix, lengths=lengths)
            loss = self._model_loss(estimates, sources, lengths)
            latest_loss_components = getattr(
                self.loss_func,
                "latest_components",
                None,
            )
            if latest_loss_components:
                if epoch_loss_components is None:
                    epoch_loss_components = {
                        key: torch.zeros_like(value.detach())
                        for key, value in latest_loss_components.items()
                    }
                for key, value in latest_loss_components.items():
                    epoch_loss_components[key] += value.detach()
            current_strengths = self._collect_router_strengths(
                reduce_distributed=False
            )
            current_gate_stats = self._collect_gate_distribution_stats(
                reduce_distributed=False
            )

            if epoch_router_strengths is None:
                epoch_router_strengths = [
                    torch.zeros_like(strength) if strength is not None else None for strength in current_strengths
                ]
            for layer_idx, strength in enumerate(current_strengths):
                if strength is not None:
                    epoch_router_strengths[layer_idx] += strength

            if epoch_gate_stats is None:
                epoch_gate_stats = []
                for stats in current_gate_stats:
                    if stats is None:
                        epoch_gate_stats.append(None)
                    else:
                        epoch_gate_stats.append({k: torch.zeros_like(v) for k, v in stats.items()})
            for layer_idx, stats in enumerate(current_gate_stats):
                if stats is None:
                    continue
                for key, value in stats.items():
                    epoch_gate_stats[layer_idx][key] += value

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()
            self.global_step += 1
            current_m2_load_update = (
                self._update_m2_load_bias_after_optimizer_step()
            )
            self._accumulate_m2_load_controller_stats(
                epoch_m2_load_stats,
                current_m2_load_update,
            )
            total_loss += loss.item()
            self._clear_model_aux()
            self.train_bar.set_description(
                f"train[{epoch}/{self.epochs}][{datetime.now().strftime('%Y-%m-%d-%H:%M')}]"
            )
            postfix = {"loss": f"{total_loss / step:.3f}"}
            if current_strengths and current_strengths[0] is not None:
                postfix["G0"] = "[" + ",".join(f"{v:.2f}" for v in current_strengths[0].tolist()) + "]"
            if (
                current_gate_stats
                and current_gate_stats[0] is not None
                and {"mean", "std", "min", "max"}.issubset(current_gate_stats[0])
            ):
                s0 = current_gate_stats[0]
                postfix["S0"] = (
                    f"{s0['mean'].item():.2f}+/-{s0['std'].item():.2f}"
                    f"[{s0['min'].item():.2f},{s0['max'].item():.2f}]"
                )
            elif current_gate_stats and current_gate_stats[0] is not None:
                s0 = current_gate_stats[0]
                if "router_entropy" in s0 and "mean_top1_probability" in s0:
                    postfix["MoE"] = (
                        f"H={s0['router_entropy'].item():.2f},"
                        f"P1={s0['mean_top1_probability'].item():.2f}"
                    )
            self.train_bar.set_postfix(**postfix)

        if step == 0:
            raise RuntimeError("Training dataloader is empty.")

        average_train_loss = torch.tensor(
            total_loss / step,
            device=self.device,
            dtype=torch.float32,
        )
        if self.world_size > 1:
            average_train_loss = reduce_value(average_train_loss)
        averaged_train_components = {}
        if epoch_loss_components:
            for key, value in epoch_loss_components.items():
                averaged = value / step
                if self.world_size > 1:
                    averaged = reduce_value(averaged)
                averaged_train_components[key] = averaged
        self._reduce_epoch_diagnostics(epoch_router_strengths, epoch_gate_stats)
        peak_training_memory = None
        if self.device.type == "cuda":
            peak_training_memory = torch.tensor(
                torch.cuda.max_memory_allocated(self.device) / 2**30,
                device=self.device,
                dtype=torch.float64,
            )
            if self.world_size > 1:
                dist.all_reduce(peak_training_memory, op=dist.ReduceOp.MAX)
            self.latest_peak_training_memory_gib = float(peak_training_memory)
        if self.rank == 0:
            self.writer.add_scalar("train/loss", average_train_loss.item(), epoch)
            if peak_training_memory is not None:
                self.writer.add_scalar(
                    "system/peak_training_memory_gib",
                    float(peak_training_memory),
                    epoch,
                )
            if averaged_train_components:
                self.writer.add_scalars(
                    "train_loss_components",
                    {
                        key: value.item()
                        for key, value in averaged_train_components.items()
                    },
                    epoch,
                )
            if epoch_router_strengths is not None:
                for layer_idx, strength in enumerate(epoch_router_strengths):
                    if strength is None:
                        continue
                    avg_strength = strength / step
                    self.writer.add_scalars(
                        f"gate_strength/layer_{layer_idx}",
                        {
                            f"expert_{expert_idx}": avg_strength[expert_idx].item()
                            for expert_idx in range(avg_strength.numel())
                        },
                        epoch,
                    )
            if epoch_gate_stats is not None:
                for layer_idx, stats in enumerate(epoch_gate_stats):
                    if stats is None:
                        continue
                    self.writer.add_scalars(
                        f"gate_distribution/layer_{layer_idx}",
                        {key: (value / step).item() for key, value in stats.items()},
                        epoch,
                    )
            self._write_m2_load_controller_stats(epoch, epoch_m2_load_stats)

    def _validation_epoch(self, epoch):
        self.validation_bar = tqdm(self.validation_dataloader, disable=self.rank != 0)
        total_loss = 0.0
        metric_keys = [
            "tiger_si_sdr",
            "tiger_si_sdri",
            "legacy_zero_mean_si_sdr",
            "legacy_zero_mean_si_sdri",
            "legacy_snr",
            "legacy_snri",
        ]
        evaluation_config = self.config.get("evaluation", {}) or {}
        compute_bss_sdr = bool(
            getattr(
                self,
                "compute_validation_bss_sdr",
                evaluation_config.get("validation_bss_sdr", False),
            )
        )
        if compute_bss_sdr:
            metric_keys.extend(["bss_sdr", "bss_sdri"])
        metric_totals = {key: 0.0 for key in metric_keys}
        num_samples = 0
        epoch_gate_stats = None
        epoch_loss_components = None
        validation_model = self.model.module if self.world_size > 1 else self.model
        with torch.no_grad():
            for step, batch in enumerate(self.validation_bar, start=1):
                mix, sources, lengths = self._unpack_batch(batch)
                mix = mix.to(self.device)
                sources = sources.to(self.device)
                lengths = self._prepare_lengths(lengths, mix)
                batch_size = mix.shape[0]
                estimates = validation_model(mix, lengths=lengths)
                if isinstance(estimates, tuple) and len(estimates) == 2:
                    _, estimates = estimates
                loss = self._loss_for_lengths(estimates, sources, lengths)
                total_loss += loss.item() * batch_size
                latest_loss_components = getattr(
                    self.loss_func,
                    "latest_components",
                    None,
                )
                if latest_loss_components:
                    if epoch_loss_components is None:
                        epoch_loss_components = {
                            key: value.detach() * 0.0
                            for key, value in latest_loss_components.items()
                        }
                    for key, value in latest_loss_components.items():
                        epoch_loss_components[key] += value.detach() * batch_size
                current_gate_stats = self._collect_gate_distribution_stats(
                    reduce_distributed=False
                )
                if epoch_gate_stats is None:
                    epoch_gate_stats = []
                    for stats in current_gate_stats:
                        if stats is None:
                            epoch_gate_stats.append(None)
                        else:
                            epoch_gate_stats.append(
                                {key: torch.zeros_like(value) for key, value in stats.items()}
                            )
                for layer_index, stats in enumerate(current_gate_stats):
                    if stats is None:
                        continue
                    for key, value in stats.items():
                        epoch_gate_stats[layer_index][key] += value * batch_size
                metric_estimates = self._pit_align_with_lengths(
                    estimates,
                    sources,
                    lengths,
                    zero_mean=False,
                )

                batch_metrics = {key: 0.0 for key in metric_keys}
                metric_lengths = (
                    [mix.shape[-1]] * batch_size
                    if lengths is None
                    else lengths.tolist()
                )
                for batch_index, sample_length in enumerate(metric_lengths):
                    sample_metrics = separation_metrics(
                        estimates[
                            batch_index : batch_index + 1,
                            :,
                            :sample_length,
                        ],
                        sources[
                            batch_index : batch_index + 1,
                            :,
                            :sample_length,
                        ],
                        mix[
                            batch_index : batch_index + 1,
                            :sample_length,
                        ],
                        include_bss_sdr=compute_bss_sdr,
                    )
                    for key in metric_keys:
                        batch_metrics[key] += sample_metrics[key].item()

                for key in metric_keys:
                    batch_metrics[key] /= batch_size
                    metric_totals[key] += batch_metrics[key] * batch_size
                num_samples += batch_size

                if self.rank == 0 and (epoch == 1 or epoch % 10 == 0) and step <= 3:
                    first_length = (
                        mix.shape[-1] if lengths is None else int(lengths[0].item())
                    )
                    mix_np = mix[0, :first_length].cpu().numpy()
                    sources_np = sources[0, :, :first_length].cpu().numpy()
                    estimates_np = (
                        metric_estimates[0, :, :first_length].detach().cpu().numpy()
                    )
                    sr = self.config.get("sample_rate", 16000)
                    mix_path = os.path.join(self.sample_path, f"sample_{step}_mix.wav")
                    if not os.path.exists(mix_path):
                        sf.write(mix_path, mix_np, samplerate=sr)
                    for k in range(self.num_sources):
                        src_path = os.path.join(self.sample_path, f"sample_{step}_s{k + 1}.wav")
                        if not os.path.exists(src_path):
                            sf.write(src_path, sources_np[k], samplerate=sr)
                        est_path = os.path.join(self.sample_path, f"sample_{step}_est{k + 1}_epoch{str(epoch).zfill(3)}.wav")
                        sf.write(est_path, estimates_np[k], samplerate=sr)

                self.validation_bar.set_description(
                    f"validate[{epoch}/{self.epochs}][{datetime.now().strftime('%Y-%m-%d-%H:%M')}]"
                )
                self.validation_bar.set_postfix(
                    loss=f"{total_loss / num_samples:.3f}",
                    SI_SDRi=(
                        f"{metric_totals['tiger_si_sdri'] / num_samples:.2f}dB"
                    ),
                    **(
                        {
                            "BSS_SDRi": (
                                f"{metric_totals['bss_sdri'] / num_samples:.2f}dB"
                            )
                        }
                        if compute_bss_sdr
                        else {}
                    ),
                )
                self._clear_model_aux()

        if num_samples == 0:
            raise RuntimeError("Validation dataloader is empty.")

        stats = torch.tensor(
            [total_loss, *[metric_totals[key] for key in metric_keys], num_samples],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_count = stats[-1].item()
        avg_loss = stats[0].item() / global_count
        avg_metrics = {
            key: stats[index + 1].item() / global_count
            for index, key in enumerate(metric_keys)
        }
        current_gate_stats = []
        if epoch_gate_stats is not None:
            for gate_stats in epoch_gate_stats:
                if gate_stats is None:
                    current_gate_stats.append(None)
                else:
                    if self.world_size > 1:
                        for value in gate_stats.values():
                            dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    current_gate_stats.append(
                        {
                            key: value / global_count
                            for key, value in gate_stats.items()
                        }
                    )
        averaged_val_components = {}
        if epoch_loss_components:
            for key, value in epoch_loss_components.items():
                if self.world_size > 1:
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                averaged_val_components[key] = value / global_count
        self.latest_validation_stats = {
            "val_loss": float(avg_loss),
            "val_tiger_si_sdri": float(avg_metrics["tiger_si_sdri"]),
        }
        if "base_pit" in averaged_val_components:
            self.latest_validation_stats["val_base_pit"] = float(
                averaged_val_components["base_pit"].item()
            )

        if self.rank == 0:
            self.writer.add_scalars(
                "val_metrics",
                {
                    "val_loss": avg_loss,
                    "SI-SDR": avg_metrics["tiger_si_sdr"],
                    "SI-SDRi": avg_metrics["tiger_si_sdri"],
                    "legacy_zero_mean_SI-SDR": avg_metrics[
                        "legacy_zero_mean_si_sdr"
                    ],
                    "legacy_zero_mean_SI-SDRi": avg_metrics[
                        "legacy_zero_mean_si_sdri"
                    ],
                    "legacy_SNR": avg_metrics["legacy_snr"],
                    "legacy_SNRi": avg_metrics["legacy_snri"],
                    **(
                        {
                            "BSS-SDR": avg_metrics["bss_sdr"],
                            "BSS-SDRi": avg_metrics["bss_sdri"],
                        }
                        if compute_bss_sdr
                        else {}
                    ),
                },
                epoch,
            )
            if averaged_val_components:
                self.writer.add_scalars(
                    "val_loss_components",
                    {
                        key: value.item()
                        for key, value in averaged_val_components.items()
                    },
                    epoch,
                )
            for layer_idx, stats in enumerate(current_gate_stats):
                if stats is None:
                    continue
                self.writer.add_scalars(
                    f"val_gate_distribution/layer_{layer_idx}",
                    {key: value.item() for key, value in stats.items()},
                    epoch,
                )
        return avg_loss, avg_metrics["tiger_si_sdri"]

    def train(self):
        if self.resume:
            self._resume_checkpoint()

        if self.start_epoch > self.epochs:
            if self.rank == 0:
                print(
                    f"Checkpoint epoch {self.start_epoch - 1} already reached "
                    f"the configured ceiling ({self.epochs}); nothing to train."
                )
                self.writer.close()
            return

        for epoch in range(self.start_epoch, self.epochs + 1):
            self._set_data_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)
            self._set_eval_mode()
            self.latest_validation_stats = {}
            val_loss, score = self._validation_epoch(epoch)
            self._record_step_bound_epoch(epoch)
            monitor_value = self._resolve_monitor_value(val_loss, score)
            if not np.isfinite(monitor_value):
                raise FloatingPointError(
                    f"{self.monitor_name} is non-finite at epoch {epoch}: "
                    f"{monitor_value}"
                )

            self.best_score = max(self.best_score, score)
            self.best_val_loss = min(self.best_val_loss, val_loss)
            improved = self._monitor_improved(monitor_value)
            if self.world_size > 1:
                improved_tensor = torch.tensor(
                    int(improved) if self.rank == 0 else 0,
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(improved_tensor, src=0)
                improved = bool(improved_tensor.item())
            if improved:
                self.best_monitor_value = monitor_value
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]["lr"]
                scheduler_type = self.config.get("scheduler", {}).get("type", "plateau")
                if scheduler_type == "plateau":
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]["lr"]
                if self.rank == 0 and new_lr < old_lr:
                    print(
                        f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f} "
                        f"({self.monitor_name} = {monitor_value:.6f})"
                    )

            save_regular = epoch % self.save_checkpoint_interval == 0
            if save_regular or improved:
                self._save_checkpoint(
                    epoch,
                    val_loss,
                    score,
                    monitor_value=monitor_value,
                    is_best=improved,
                    save_regular=save_regular,
                )

            should_stop = self.no_improve_count >= self.early_stop_patience
            if self.world_size > 1:
                stop_tensor = torch.tensor(
                    int(should_stop) if self.rank == 0 else 0,
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                if self.rank == 0:
                    print(
                        f"\nEarly stopping! {self.monitor_name} did not improve "
                        "by the configured minimum delta for "
                        f"{self.early_stop_patience} epochs."
                    )
                break

        if self.rank == 0 and self.state_dict_best is not None:
            best_epoch = self.state_dict_best["epoch"]
            print(f"------------Training for {self.epochs} epochs is done!------------")
            print(
                f"Best {self.monitor_name}: {self.best_monitor_value:.6f} "
                f"(epoch {best_epoch})"
            )
            print(f"Lowest observed validation loss: {self.best_val_loss:.6f}")
            print(f"Best observed SI-SDRi: {self.best_score:.2f} dB")
        if self.rank == 0:
            self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True)
    parser.add_argument(
        "-D",
        "--device",
        default="0",
        help="comma-separated GPU ids (e.g. 0 or 0,1), or 'cpu' for a slow single-process run",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help=(
            "DDP rendezvous port. An explicit CLI value overrides MASTER_PORT; "
            "otherwise the environment or default 12354 is used."
        ),
    )
    parser.add_argument(
        "--ddp-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="Distributed backend. auto prefers NCCL and otherwise uses Gloo.",
    )
    args = parser.parse_args()
    args.run_id = (
        datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss-%f")
        + "-"
        + uuid.uuid4().hex[:8]
    )

    args.use_cpu = str(args.device).strip().lower() == "cpu"
    if args.use_cpu:
        args.world_size = 1
    else:
        gpu_ids = [int(i) for i in str(args.device).split(",")]
        args.world_size = len(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    config = OmegaConf.load(args.config)
    if args.world_size > 1:
        torch.multiprocessing.spawn(run, args=(config, args), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
