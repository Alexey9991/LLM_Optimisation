"""Distance profiling module for analyzing transformer model layer distances.

v2: multi-signal block selection for depth pruning.

Signals per candidate block (start, start + layers_to_skip):
  dist_act  - masked angular distance between h_in and h_out, ALL non-padded
              tokens (v1 used only the last token -> high variance).
  lin_res   - relative residual of a ridge linear fit  h_in @ W ~= h_out.
              Directly measures "linear replaceability", i.e. exactly what
              lstsq will do to this block. Accumulated in the same forward
              pass, solved on CPU afterwards.  (recommended second signal)
  taylor    - |sum g_out * (h_in - h_out)| per token: first-order estimate of
              the loss increase if the block is ablated. Needs one backward
              per batch (enabled via alpha_taylor > 0).

Combined score = weighted sum of min-max normalized signals (lower = better).
Defaults (alpha_act=1, others 0) reproduce v1 ranking behaviour, but computed
over all tokens.

Output:
  layer_distances.csv - all raw signals + combined score per candidate
  distances.pth       - combined score list, same format/convention as v1
                        (index i -> block (i+1, i+1+layers_to_skip), 1-based)
"""

import argparse
import csv
import gc
import logging
from typing import List, Optional

import numpy as np
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import get_calib_dataloader, seed_all

init(autoreset=True)
logging.basicConfig(
    format=(f"{Fore.CYAN}%(asctime)s "
            f"{Fore.YELLOW}[%(levelname)s] "
            f"{Fore.RESET}%(message)s"),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
seed_all()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _masked_angular_distance(h_in: torch.Tensor,
                             h_out: torch.Tensor,
                             mask: torch.Tensor,
                             eps: float = 1e-7) -> float:
    """Mean angular distance over non-padded tokens.

    h_in, h_out: (B, S, D); mask: (B, S) with 1 for real tokens.
    """
    m = mask.bool().view(-1)
    a = h_in.view(-1, h_in.shape[-1])[m].float()
    b = h_out.view(-1, h_out.shape[-1])[m].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    ang = torch.arccos(cos.clamp(-1 + eps, 1 - eps)) / torch.pi
    return ang.mean().item()


def _answer_only_label_mask(texts: List[str],
                            input_ids: torch.Tensor,
                            tokenizer,
                            marker: str = "Answer:") -> torch.Tensor:
    """Labels mask (B, S): True for tokens belonging to the answer part.

    Falls back to the full sequence for texts without the marker.
    """
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    for b, text in enumerate(texts):
        pos = text.find(marker)
        if pos == -1:
            continue
        prefix_len = len(tokenizer(text[:pos + len(marker)],
                                   add_special_tokens=True)["input_ids"])
        keep[b, :min(prefix_len, input_ids.shape[1])] = False
    return keep


def _ce_loss(logits: torch.Tensor,
             labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def profile_distances(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,   # kept for API compat
    use_4bit: bool = False,
    save_path: Optional[str] = None,               # kept for API compat
    min_distance_layer: Optional[int] = None,      # kept for API compat
    token: Optional[str] = None,
    # ---- new options -----------------------------------------------------
    alpha_act: float = 1.0,
    alpha_linres: float = 0.0,
    alpha_taylor: float = 0.0,
    ridge_lambda: float = 1e-3,
    answer_only_loss: bool = False,
    answer_marker: str = "Answer:",
    distances_out: str = "distances.pth",
    csv_out: str = "layer_distances.csv",
    cleanup_every: int = 20,
) -> None:
    """Profile candidate blocks with multiple signals.

    alpha_act    weight of the all-token angular distance (v1-like signal)
    alpha_linres weight of the linear-fit relative residual (recommended: try
                 1.0 with alpha_act=0 or 0.5/0.5 -- it measures exactly what
                 lstsq will later exploit)
    alpha_taylor weight of the first-order ablation loss estimate; > 0 turns
                 on a backward pass per batch (avoid combining with use_4bit)
    """
    need_grad = alpha_taylor > 0
    need_linres = alpha_linres > 0

    if need_grad and use_4bit:
        logging.warning("taylor signal with 4-bit weights: gradients will be "
                        "noisy; prefer use_4bit=False for profiling")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if need_grad:
        # freeze weights: we only need grads w.r.t. hidden states
        model.requires_grad_(False)

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer,
    )

    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    n_cand = L - layers_to_skip

    act_dists = [[] for _ in range(n_cand)]
    taylor_vals = [[] for _ in range(n_cand)]

    # accumulators for the linear-fit signal (fp64 on CPU: cheap and exact)
    if need_linres:
        xtx = [torch.zeros(D, D, dtype=torch.float64) for _ in range(n_cand)]
        xty = [torch.zeros(D, D, dtype=torch.float64) for _ in range(n_cand)]
        yty_tr = [0.0 for _ in range(n_cand)]

    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Profiling blocks{Fore.RESET}",
        dynamic_ncols=True, colour="green",
    )):
        inputs = tokenizer(batch, return_tensors="pt", padding="longest",
                           max_length=max_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        mask = inputs["attention_mask"]

        if need_grad:
            with torch.enable_grad():
                outputs = model(**inputs)
                hs = outputs.hidden_states          # tuple, len L+1
                for h in hs:
                    if h.requires_grad:
                        h.retain_grad()
                labels = inputs["input_ids"].clone()
                labels[mask == 0] = -100
                if answer_only_loss:
                    keep = _answer_only_label_mask(batch, inputs["input_ids"],
                                                   tokenizer, answer_marker)
                    labels[~keep] = -100
                loss = _ce_loss(outputs.logits, labels)
                loss.backward()
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                hs = outputs.hidden_states

        m_flat = mask.bool().view(-1)

        for s in range(n_cand):
            h_in = hs[s + 1]                        # after layer s+1 (1-based)
            h_out = hs[s + 1 + layers_to_skip]

            # -- signal 1: all-token angular distance
            act_dists[s].append(
                _masked_angular_distance(h_in, h_out, mask))

            # -- signal 2: linear-fit accumulators
            if need_linres:
                x = h_in.detach().view(-1, D)[m_flat].to(torch.float32)
                y = h_out.detach().view(-1, D)[m_flat].to(torch.float32)
                xtx[s] += (x.t() @ x).double().cpu()
                xty[s] += (x.t() @ y).double().cpu()
                yty_tr[s] += (y * y).sum().double().cpu().item()

            # -- signal 3: first-order ablation estimate
            if need_grad and h_out.grad is not None:
                g = h_out.grad.view(-1, D)[m_flat]
                delta = (h_in.detach() - h_out.detach()).view(-1, D)[m_flat]
                # |E_token[ g . delta ]| : predicted per-token loss change
                taylor_vals[s].append(
                    (g.float() * delta.float()).sum(dim=-1).abs().mean().item())

        del outputs, hs
        if need_grad:
            model.zero_grad(set_to_none=True)
        if (batch_idx + 1) % cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- finalize signals -------------------------------------------------
    sig_act = [float(np.mean(v)) for v in act_dists]

    sig_linres = None
    if need_linres:
        sig_linres = []
        eye = torch.eye(D, dtype=torch.float64)
        for s in range(n_cand):
            reg = ridge_lambda * torch.trace(xtx[s]) / D * eye
            w = torch.linalg.solve(xtx[s] + reg, xty[s])
            # ||XW - Y||_F^2 = tr(YtY) - 2 tr(WtXtY) + tr(Wt XtX W)
            res_sq = (yty_tr[s]
                      - 2.0 * torch.sum(w * xty[s]).item()
                      + torch.sum(w * (xtx[s] @ w)).item())
            sig_linres.append(max(res_sq, 0.0) / max(yty_tr[s], 1e-12))

    sig_taylor = None
    if need_grad:
        sig_taylor = [float(np.mean(v)) if v else float("inf")
                      for v in taylor_vals]

    # ---- combine ----------------------------------------------------------
    combined = [0.0] * n_cand
    parts = [(alpha_act, sig_act)]
    if sig_linres is not None:
        parts.append((alpha_linres, sig_linres))
    if sig_taylor is not None:
        parts.append((alpha_taylor, sig_taylor))
    for weight, sig in parts:
        if weight == 0:
            continue
        for i, v in enumerate(_minmax(sig)):
            combined[i] += weight * v

    # ---- write outputs ----------------------------------------------------
    with open(csv_out, "w", newline="") as f:
        fields = ["block_start", "block_end", "dist_act"]
        if sig_linres is not None:
            fields.append("lin_res")
        if sig_taylor is not None:
            fields.append("taylor")
        fields.append("combined")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n_cand):
            row = {"block_start": i + 1,
                   "block_end": i + 1 + layers_to_skip,
                   "dist_act": sig_act[i],
                   "combined": combined[i]}
            if sig_linres is not None:
                row["lin_res"] = sig_linres[i]
            if sig_taylor is not None:
                row["taylor"] = sig_taylor[i]
            writer.writerow(row)

    torch.save(combined, distances_out)

    best = int(np.argmin(combined)) + 1
    logging.info(
        f"{Fore.GREEN}Best candidate: block ({best}, {best + layers_to_skip}) "
        f"by combined score. Full table: {csv_out}{Fore.RESET}")
    top3 = np.argsort(combined)[:3]
    for rank, i in enumerate(top3, 1):
        logging.info(f"  top-{rank}: ({i + 1}, {i + 1 + layers_to_skip}) "
                     f"score={combined[i]:.4f} act={sig_act[i]:.4f}"
                     + (f" lin_res={sig_linres[i]:.4f}" if sig_linres else "")
                     + (f" taylor={sig_taylor[i]:.3e}" if sig_taylor else ""))


def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-signal distance analysis from a config file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    profile_distances(**read_config(args.config))