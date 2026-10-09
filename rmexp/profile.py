"""Block profiles and block choice: mean rank of dist_act and dist_grad, ties by dist_act; greedy for several blocks."""
import gc
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import vram

SIGNALS = ["dist_act", "dist_grad", "grad_contrib", "grad_contrib_fro", "grad_coher"]
RANK_COLS = ["r_dist_act", "r_dist_grad", "r_score_cos", "r_grad_contrib", "r_grad_contrib_fro", "r_coher"]


def profile_csv(cfg, k: int) -> Path:
    return cfg.profile_dir / f"llama3_8b_skip{k}_gsm8k{cfg.calib_size}_3scores.csv"


def profile(cfg, rm, k: int, force: bool = False) -> pd.DataFrame:
    path = profile_csv(cfg, k)
    if path.exists() and not force:
        print("profile on disk:", path)
    else:
        kw = dict(model_path=cfg.model_path, dataset="openai/gsm8k", dataset_column="text",
                  batch_size=cfg.profile_batch_size, max_length=cfg.calib_max_length, layers_to_skip=k,
                  dataset_size=cfg.calib_size, dataset_subset="train", use_4bit=cfg.precision == "nf4",
                  answer_only_loss=cfg.profile_answer_only_loss, compute_taylor=False, taylor_veto_quantile=None,
                  csv_save_path=str(path), distances_save_path=str(path.with_suffix(".distances.pth")))
        kw = {key: v for key, v in kw.items() if key in rm.ds_sig}
        gc.collect(); torch.cuda.empty_cache()
        alloc = vram(f"before profile skip {k}")
        assert alloc < 0.5, "GPU0 is not empty: restart the kernel before profiling"
        res = rm.ds.profile_distances(**kw)
        gc.collect(); torch.cuda.empty_cache()
        json.dump({"hw": torch.cuda.get_device_name(0), "host": platform.node(),
                   "extra": {"mode": cfg.precision, "layers_to_skip": k, "dataset_size": cfg.calib_size,
                             "max_length": cfg.calib_max_length, "batch_size": cfg.profile_batch_size,
                             "answer_only_loss": cfg.profile_answer_only_loss, "compute_taylor": False,
                             "distance_scored_md5": rm.ds_md5, "selected_block_cos": res["selected_block"]}},
                  open(path.with_suffix(".meta.json"), "w"), indent=2)
    p = pd.read_csv(path)
    assert len(p) == cfg.n_layers - k, (k, len(p))
    assert p["dist_grad"].notna().all(), "no dist_grad: the gradients did not reach the hidden states"
    return p


def ranked(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["rank_act"] = p["dist_act"].rank(method="first") - 1
    p["rank_grad"] = p["dist_grad"].rank(method="first") - 1
    p["score_ras"] = (p["rank_act"] + p["rank_grad"]) / 2
    return p.sort_values(["score_ras", "dist_act"]).reset_index(drop=True)


def top1(p: pd.DataFrame) -> Tuple[int, int]:
    r = ranked(p)
    return int(r.loc[0, "block_start"]), int(r.loc[0, "block_end"])


def pick_blocks(lengths: List[int], profiles: Dict[int, pd.DataFrame], gap: int = 1,
                order_col: str = "score_ras") -> List[Tuple[int, int]]:
    used, chosen = set(), []
    for L in sorted(lengths, reverse=True):
        r = ranked(profiles[L]) if order_col == "score_ras" else profiles[L].sort_values([order_col, "dist_act"])
        for s, e in zip(r["block_start"].astype(int), r["block_end"].astype(int)):
            if set(range(s - gap, e + gap)) & used:
                continue
            chosen.append((s, e))
            used |= set(range(s, e))
            break
        else:
            raise RuntimeError(f"no free block of length {L} for {lengths}")
    return sorted(chosen)


def rank_table(p: pd.DataFrame) -> pd.DataFrame:
    """Rank of every window under every signal, 0 = best."""
    t = p[["block_start", "block_end"]].astype(int).copy()
    rank = lambda x: pd.Series(x).rank(method="first").astype(int).values - 1
    t["r_dist_act"] = rank(p["dist_act"])
    t["r_dist_grad"] = rank(p["dist_grad"])
    cos = (t["r_dist_act"] + t["r_dist_grad"]) / 2
    order = np.lexsort((p["dist_act"].values, cos.values))
    t["r_score_cos"] = 0
    t.loc[order, "r_score_cos"] = np.arange(len(t))
    for col in ("grad_contrib", "grad_contrib_fro"):
        t["r_" + col] = rank(p[col]) if col in p else -1
    t["r_coher"] = rank(1 - p["grad_coher"]) if "grad_coher" in p else -1
    return t


def greedy_by_signal(lengths: List[int], tables: Dict[int, pd.DataFrame], col: str, gap: int = 1):
    used, chosen = set(), []
    for L in sorted(lengths, reverse=True):
        t = tables[L].sort_values([col, "r_dist_act"])
        for s, e in zip(t.block_start, t.block_end):
            if set(range(s - gap, e + gap)) & used:
                continue
            chosen.append((int(s), int(e)))
            used |= set(range(s, e))
            break
        else:
            return None
    return sorted(chosen)
