"""Experiment: prune, heal, restore, evaluate; every stage skips what is already on disk."""
import gc
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .config import Cfg, ExperimentSpec, free_disk_gb, vram
from .evaluate import result_path, run_evaluation
from .profile import pick_blocks, profile, top1
from .prune import prune as _prune
from .restore import restore_blocks
from .train import train_lora


def needed_lengths(exps: List[ExperimentSpec]) -> List[int]:
    out = set()
    for e in exps:
        if e.lengths:
            out |= set(e.lengths)
        if e.top1_length:
            out.add(e.top1_length)
    return sorted(out)


def resolve_blocks(cfg: Cfg, spec: ExperimentSpec, profiles: Optional[Dict[int, object]] = None,
                   rm=None) -> List[Tuple[int, int]]:
    run_dir = cfg.runs_dir / spec.name
    plan_file = run_dir / "plan.json"
    if spec.blocks is not None:
        blocks = spec.blocks
    elif plan_file.exists():
        blocks = [tuple(b) for b in json.load(open(plan_file))["blocks_1based"]]
    elif (run_dir / "selected_block.json").exists():                   # earlier single-block runs
        blocks = [tuple(json.load(open(run_dir / "selected_block.json"))["block"])]
    else:
        assert profiles is not None or rm is not None, f"{spec.name}: profiles are needed to choose the blocks"
        if profiles is None:
            profiles = {k: profile(cfg, rm, k) for k in ([spec.top1_length] if spec.top1_length else spec.lengths)}
        blocks = [top1(profiles[spec.top1_length])] if spec.top1_length else pick_blocks(spec.lengths, profiles, cfg.gap)
    deleted = sorted({l for s, e in blocks for l in range(s, e)})
    carriers = [s - 1 for s, _ in blocks]
    assert len(deleted) == sum(e - s for s, e in blocks), ("overlapping blocks", blocks)
    assert not set(carriers) & set(deleted), ("a carrier layer is deleted", blocks)
    assert min(carriers) >= 0 and max(deleted) < cfg.n_layers, blocks
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"blocks_1based": blocks, "deleted_0based": deleted, "carriers_0based": carriers,
               "rule": ("explicit" if spec.blocks else f"top-1 of length {spec.top1_length}" if spec.top1_length
                        else f"greedy by mean rank of dist_act and dist_grad, lengths {spec.lengths}, gap {cfg.gap}")},
              open(plan_file, "w"), indent=2)
    return blocks


class Experiment:
    def __init__(self, cfg: Cfg, spec: ExperimentSpec, blocks: List[Tuple[int, int]]):
        self.cfg, self.spec, self.blocks = cfg, spec, [tuple(b) for b in blocks]
        self.run_dir = cfg.runs_dir / spec.name
        self.results_dir = self.run_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pruned_dir = self.run_dir / "pruned_ReplaceMe_joint_lstsq_1"
        self.heal_adapter_dir = self.run_dir / "heal_adapter"
        meta = self.pruned_dir / "blocks.json"
        if meta.exists():
            m = json.load(open(meta))
            on_disk = m.get("blocks_1based") or m.get("blocks") or [m.get("block_1based")]
            on_disk = sorted(tuple(b) for b in on_disk if b)
            assert on_disk == sorted(self.blocks), f"{spec.name}: pruned model on disk has blocks {on_disk}, " \
                                                   f"this run asks for {self.blocks}; use another name"

    def __repr__(self):
        return f"Experiment({self.spec.name}, blocks={self.blocks})"

    @property
    def deleted(self) -> List[int]:
        return sorted({l for s, e in self.blocks for l in range(s, e)})

    @property
    def carriers(self) -> List[int]:
        return [s - 1 for s, _ in self.blocks]

    def restored_dir(self, remove_T: bool) -> Path:
        return self.run_dir / ("restored_noT" if remove_T else "restored")

    def done(self, stage: str, suffix_tag: str = "") -> bool:
        return result_path(self.results_dir, stage, self.cfg.precision, suffix_tag).exists()

    def _eval(self, stage, model_path, adapter=None, suffix_tag=""):
        if self.done(stage, suffix_tag):
            print(f"{self.spec.name}: {stage}{suffix_tag} on disk")
            return
        gc.collect(); torch.cuda.empty_cache(); vram(f"before eval {stage}{suffix_tag}")
        run_evaluation(self.cfg, stage, str(model_path), adapter and str(adapter), suffix_tag, self.results_dir)
        gc.collect(); torch.cuda.empty_cache()

    def prune(self, rm) -> Path:
        return _prune(self.cfg, rm, self.blocks, self.run_dir)

    def eval_pruned(self):
        self._eval("pruned_T_only", self.pruned_dir)

    def heal(self) -> Path:
        if (self.heal_adapter_dir / "adapter_config.json").exists():
            print(f"{self.spec.name}: heal adapter on disk")
        else:
            assert (self.pruned_dir / "config.json").exists(), "prune first"
            gc.collect(); torch.cuda.empty_cache(); vram("before heal")
            train_lora(self.cfg, str(self.pruned_dir), self.heal_adapter_dir)
            gc.collect(); torch.cuda.empty_cache()
        return self.heal_adapter_dir

    def eval_healed(self):
        self._eval("pruned_healed", self.pruned_dir, self.heal_adapter_dir)

    def restore(self, remove_T: bool = False) -> Path:
        out = self.restored_dir(remove_T)
        if (out / "config.json").exists():
            print(f"{self.spec.name}: {out.name} on disk")
            return out
        assert (self.heal_adapter_dir / "adapter_config.json").exists(), "heal first"
        print("free disk GB:", free_disk_gb(self.cfg.runs_dir))
        return restore_blocks(self.cfg.model_path, self.pruned_dir, self.heal_adapter_dir, out, self.blocks,
                              self.cfg.n_layers, remove_T=remove_T)

    def eval_restored(self, remove_T: bool = False, delete_after: bool = False):
        tag = "_noT" if remove_T else ""
        if self.done("replaceme", tag):
            print(f"{self.spec.name}: replaceme{tag} on disk")
            return
        d = self.restore(remove_T)
        self._eval("replaceme", d, suffix_tag=tag)
        if delete_after:
            shutil.rmtree(d)
            print("deleted", d)

    def run_all(self, rm, delete_restored: bool = True):
        self.prune(rm)
        if self.spec.eval_pruned:
            self.eval_pruned()
        self.heal()
        if self.spec.eval_pruned:
            self.eval_healed()
        self.eval_restored(False, delete_restored)
        if self.spec.restore_without_T:
            self.eval_restored(True, delete_restored)

    def result_files(self) -> Dict[str, Path]:
        return {s: result_path(self.results_dir, "replaceme" if s.startswith("replaceme") else s, self.cfg.precision,
                               "_noT" if s == "replaceme_noT" else "")
                for s in ("pruned_T_only", "pruned_healed", "replaceme", "replaceme_noT")}


def eval_baseline(cfg: Cfg):
    if result_path(cfg.results_dir, "baseline", cfg.precision).exists():
        print("baseline on disk")
        return
    run_evaluation(cfg, "baseline", cfg.model_path)
    gc.collect(); torch.cuda.empty_cache()


def train_sft(cfg: Cfg):
    if (cfg.sft_adapter_dir / "adapter_config.json").exists():
        print("SFT adapter on disk:", cfg.sft_adapter_dir)
        return cfg.sft_adapter_dir
    vram("before SFT")
    train_lora(cfg, cfg.model_path, cfg.sft_adapter_dir)
    gc.collect(); torch.cuda.empty_cache()
    return cfg.sft_adapter_dir


def eval_sft(cfg: Cfg):
    if result_path(cfg.results_dir, "sft", cfg.precision).exists():
        print("sft on disk")
        return
    run_evaluation(cfg, "sft", cfg.model_path, str(cfg.sft_adapter_dir))
    gc.collect(); torch.cuda.empty_cache()
