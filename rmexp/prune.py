"""lstsq_joint call, norm ratio of every carrier, blocks.json."""
import gc
import io
import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors import safe_open

from .config import vram
from .replaceme_repo import md5


def model_dir(path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path), allow_patterns=["*.json", "*.safetensors"]))


def read_tensor(directory, name: str) -> torch.Tensor:
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    files = ([directory / json.load(open(index))["weight_map"][name]] if index.exists()
             else sorted(directory.glob("*.safetensors")))
    for f in files:
        with safe_open(str(f), "pt") as st:
            if name in st.keys():
                return st.get_tensor(name)
    raise KeyError(name)


def norm_ratios(cfg, pruned_dir, blocks: List[Tuple[int, int]]) -> List[float]:
    out, removed = [], 0
    base = model_dir(cfg.model_path)
    for s, e in blocks:
        w_p = read_tensor(pruned_dir, f"model.layers.{s - 1 - removed}.mlp.down_proj.weight").float()
        w_b = read_tensor(base, f"model.layers.{s - 1}.mlp.down_proj.weight").float()
        out.append(round((w_p.norm() / w_b.norm()).item(), 4))
        removed += e - s
    return out


def prune(cfg, rm, blocks: List[Tuple[int, int]], run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    pruned_dir = run_dir / "pruned_ReplaceMe_joint_lstsq_1"
    blocks = [tuple(b) for b in blocks]
    if (pruned_dir / "config.json").exists():
        print("pruned model on disk:", pruned_dir)
    else:
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.WARNING)
        logging.getLogger().addHandler(h)
        try:
            gc.collect(); torch.cuda.empty_cache(); vram("before prune")
            out = rm.lj.lstsq(model_path=cfg.model_path, dataset="openai/gsm8k", dataset_column="text",
                              batch_size=cfg.calib_batch_size, max_length=cfg.calib_max_length,
                              layers_to_skip=max(e - s for s, e in blocks), dataset_size=cfg.calib_size,
                              dataset_subset="train", use_4bit=cfg.precision == "nf4",
                              save_path=str(run_dir / "pruned"), alpha=cfg.alpha_reg, alpha_act=cfg.alpha_act,
                              alpha_grad=cfg.alpha_grad, identity_blend_threshold=cfg.identity_blend_threshold,
                              num_A=1,
                              selected_blocks=blocks, save_transform_only=True)
            assert Path(out).resolve() == pruned_dir.resolve(), (out, pruned_dir)
        finally:
            logging.getLogger().removeHandler(h)
        assert not any(k in buf.getvalue() for k in ("fallback", "skipping", "identity transform")), buf.getvalue()
        gc.collect(); torch.cuda.empty_cache()

    n = json.load(open(pruned_dir / "config.json"))["num_hidden_layers"]
    removed = sum(e - s for s, e in blocks)
    assert n == cfg.n_layers - removed, (n, removed)
    ratios = norm_ratios(cfg, pruned_dir, blocks)
    json.dump({"blocks_1based": blocks, "deleted_0based": sorted({l for s, e in blocks for l in range(s, e)}),
               "carriers_0based": [s - 1 for s, _ in blocks], "norm_ratios": ratios,
               "lstsq_joint": rm.lj_md5, "distance_scored": rm.ds_md5,
               "alpha_act": cfg.alpha_act, "alpha_grad": cfg.alpha_grad,
               "identity_blend_threshold": cfg.identity_blend_threshold},
              open(pruned_dir / "blocks.json", "w"), indent=2)
    flag = "" if max(ratios) < cfg.norm_ratio_limit else f"  <-- above {cfg.norm_ratio_limit}: expect a failed restore"
    print(f"{n} layers | blocks {blocks} | norm ratios {ratios}{flag}")
    return pruned_dir
