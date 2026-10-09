"""Experiment configuration: one flat dataclass read from YAML, plus the list of experiments (block plans)."""
import os
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Turing cards (2080 Ti) have no bf16; pick the compute dtype by capability, never hardcode it
COMPUTE_DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                 else torch.float16)


@dataclass
class Cfg:
    # paths
    replaceme_repo: str = "."                    # folder holding ReplaceMe/; "." = this repo
    runs: str = "/home/tnn/LLM_optimisatoin/runs_reproduce"
    control_run: str = "llama-3-8b-repo-nf4"      # where baseline / SFT results and the SFT adapter live
    profiles_dir: str = "profiles"                # under runs
    # model
    model_path: str = "unsloth/llama-3-8b"
    n_layers: int = 32
    precision: str = "nf4"
    seed: int = 42
    # calibration (profile + lstsq)
    calib_size: int = 128
    calib_max_length: int = 256
    calib_batch_size: int = 2
    profile_batch_size: int = 1
    profile_answer_only_loss: bool = True
    # T estimation (lstsq_joint)
    alpha_act: float = 1.0
    alpha_grad: float = 0.5
    alpha_reg: float = 0.0
    identity_blend_threshold: float = 0.15
    norm_ratio_limit: float = 3.1                 # ||T^T W|| / ||W|| above this: the block did not recover after heal
    # LoRA / training (SFT control and heal share the protocol)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj"])
    train_batch_size: int = 2
    train_grad_accum: int = 4
    train_limit: Optional[int] = None
    train_max_length: int = 384
    train_learning_rate: float = 2e-4
    train_epochs: float = 1.0
    train_warmup_steps: int = 20
    train_gradient_checkpointing: bool = True
    train_optim: str = "paged_adamw_8bit"
    train_lm_head_on_second_gpu: bool = False     # heal of 24L with 4 carriers did not fit on one 11 GiB card
    # evaluation
    eval_dataset: str = "gsm8k"
    eval_batch_size: int = 8
    eval_max_new_tokens: int = 256
    eval_limit: Optional[int] = None
    # block choice
    gap: int = 1                                  # kept layers between blocks (each one carries the next T)

    @property
    def runs_dir(self) -> Path:
        return Path(self.runs)

    @property
    def control_dir(self) -> Path:
        return self.runs_dir / self.control_run

    @property
    def sft_adapter_dir(self) -> Path:
        return self.control_dir / "sft_adapter"

    @property
    def profile_dir(self) -> Path:
        return self.runs_dir / self.profiles_dir

    # kept for the evaluation / training code, which reads these names from the config object
    @property
    def results_dir(self) -> Path:
        return self.control_dir / "results"


@dataclass
class ExperimentSpec:
    """One pruning run: either explicit blocks, or block lengths chosen greedily from the profiles,
    or the top-1 window of a given length. Blocks are 1-based (start, end): 0-based layers start..end-1 go."""
    name: str
    blocks: Optional[List[Tuple[int, int]]] = None
    lengths: Optional[List[int]] = None
    top1_length: Optional[int] = None
    restore_without_T: bool = False
    eval_pruned: bool = False                     # also evaluate "T only" and "pruned + heal"

    def __post_init__(self):
        n = sum(x is not None for x in (self.blocks, self.lengths, self.top1_length))
        assert n == 1, f"{self.name}: give exactly one of blocks / lengths / top1_length"
        if self.blocks is not None:
            self.blocks = [tuple(int(v) for v in b) for b in self.blocks]


def load_config(path) -> Tuple[Cfg, List[ExperimentSpec]]:
    raw = yaml.safe_load(open(path))
    exps = [ExperimentSpec(**e) for e in raw.pop("experiments", [])]
    known = {f.name for f in fields(Cfg)}
    unknown = set(raw) - known
    assert not unknown, f"unknown config keys: {sorted(unknown)}"
    cfg = Cfg(**raw)
    repo_root = Path(__file__).resolve().parents[1]
    cfg.replaceme_repo = str((repo_root / cfg.replaceme_repo).resolve()
                             if not Path(cfg.replaceme_repo).is_absolute() else Path(cfg.replaceme_repo))
    assert Path(cfg.replaceme_repo, "ReplaceMe", "utils.py").is_file(), cfg.replaceme_repo
    Path(cfg.runs).mkdir(parents=True, exist_ok=True)
    (cfg.control_dir / "results").mkdir(parents=True, exist_ok=True)
    cfg.profile_dir.mkdir(parents=True, exist_ok=True)
    return cfg, exps


def vram(tag: str = "") -> float:
    gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"vram {tag}: {gib:.2f} GiB")
    return gib


def free_disk_gb(path) -> float:
    return round(shutil.disk_usage(path).free / 2**30, 1)
