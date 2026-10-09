"""Distance profiling module for analyzing transformer model layer distances.

v1 + arbitration:
  1. Angular profile over the calibration set (v1 last-token metric, unchanged).
  2. Top-k (default 3) candidate blocks by smallest angular distance.
  3. For each candidate: remove the 8-layer block, splice in a least-squares
     linear replacement W (h_in @ W ~= h_out), and measure held-out CE / PPL
     on texts NOT used for the lstsq fit.
  4. The candidate with the lowest CE/PPL wins; it is written as the forced
     argmin into distances.pth so the downstream pipeline picks it up.
  5. Full profile + arbitration results go to an Excel file (two sheets).

Convention (unchanged): profile index i <-> block (start, end) = (i+1, i+1+skip);
0-based layers removed = range(i+1, i+1+skip); the block input is
hidden_states[i+1] (output of 0-based layer i), the block output is
hidden_states[i+1+skip].
"""

import argparse
import gc
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (compute_block_distances, get_calib_dataloader,
                    get_last_non_padded_tokens, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

logging.basicConfig(
    format=(
        f"{Fore.CYAN}%(asctime)s "
        f"{Fore.YELLOW}[%(levelname)s] "
        f"{Fore.RESET}%(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

seed_all()


# --------------------------------------------------------------- loss helpers
def _mean_ce(model, tokenizer, texts: List[str], max_length: int,
             batch_size: int = 1) -> float:
    """Mean per-token cross-entropy of `model` on `texts` (padding masked)."""
    total_nll, total_tok = 0.0, 0
    device = next(model.parameters()).device
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding="longest",
                        max_length=max_length, truncation=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits

        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        total_nll += nll.item()
        total_tok += (shift_labels != -100).sum().item()

    return total_nll / max(total_tok, 1)


# ------------------------------------------------------- lstsq replacement
class _LinearBridge(nn.Module):
    """Drop-in module applied in place of a removed block: h -> h @ W.

    Mimics a decoder layer's call signature: accepts anything, returns a
    tuple whose first element is the transformed hidden states.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)  # [H, H], float32

    def forward(self, hidden_states, *args, **kwargs):
        out = (hidden_states.to(self.weight.dtype) @ self.weight)
        return (out.to(hidden_states.dtype),)


def _collect_covariances(model, tokenizer, texts: List[str],
                         cand_indices: List[int], layers_to_skip: int,
                         max_length: int, batch_size: int):
    """One pass over `texts`; for every candidate i accumulate
    G_i = X^T X and C_i = X^T Y over all non-padded tokens, where
    X = hidden_states[i+1] (block input), Y = hidden_states[i+1+skip]."""
    device = next(model.parameters()).device
    hidden = model.config.hidden_size
    stats = {
        i: {
            "G": torch.zeros(hidden, hidden, dtype=torch.float32, device=device),
            "C": torch.zeros(hidden, hidden, dtype=torch.float32, device=device),
        }
        for i in cand_indices
    }

    for b in tqdm(range(0, len(texts), batch_size),
                  desc=f"{Fore.GREEN}Collecting lstsq activations{Fore.RESET}",
                  dynamic_ncols=True, colour="green"):
        chunk = texts[b:b + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding="longest",
                        max_length=max_length, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc, use_cache=False)

        mask = enc["attention_mask"].bool().view(-1)
        hs = outputs.hidden_states
        for i in cand_indices:
            X = hs[i + 1].view(-1, hidden)[mask].float()
            Y = hs[i + 1 + layers_to_skip].view(-1, hidden)[mask].float()
            stats[i]["G"] += X.T @ X
            stats[i]["C"] += X.T @ Y

    return stats


def _solve_lstsq(G: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """W = argmin ||X W - Y||_F^2 via the normal equations (float64 on CPU)."""
    G64 = G.double().cpu()
    C64 = C.double().cpu()
    # tiny jitter for numerical safety (does not change the solution materially)
    G64 += 1e-6 * torch.eye(G64.shape[0], dtype=torch.float64)
    W = torch.linalg.solve(G64, C64)
    return W.float()


def _evaluate_candidate(model, tokenizer, start: int, end: int,
                        W: torch.Tensor, heldout_texts: List[str],
                        max_length: int, batch_size: int) -> float:
    """Swap in the pruned layer list (block removed, bridge inserted),
    measure held-out CE, always restore the original model."""
    orig_layers = model.model.layers
    device = next(model.parameters()).device
    bridge = _LinearBridge(W.to(device))
    pruned = nn.ModuleList(
        list(orig_layers[:start]) + [bridge] + list(orig_layers[end:])
    )
    try:
        model.model.layers = pruned
        ce = _mean_ce(model, tokenizer, heldout_texts, max_length, batch_size)
    finally:
        model.model.layers = orig_layers
    return ce


# ------------------------------------------------------------------ profiler
def profile_distances(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    # ---- arbitration options ----
    top_k: int = 3,
    heldout_num_texts: int = 64,
    excel_save_path: str = "layer_distances.xlsx",
    distances_save_path: str = "distances.pth",
) -> None:
    """Angular profile -> top-k shortlist -> lstsq prune + held-out PPL
    arbitration -> winner forced as argmin in distances.pth."""
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

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size,
        batch_size, tokenizer,
    )

    n_cand = model.config.num_hidden_layers - layers_to_skip
    all_distances = [[] for _ in range(n_cand)]
    all_texts: List[str] = []

    # ============== Phase A: angular profile (v1 last-token metric) ==========
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Computing Distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        texts = list(batch)
        all_texts.extend(texts)

        inputs = tokenizer(
            texts, return_tensors="pt", padding="longest",
            max_length=max_length, truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.hidden_states
        last_non_padded_hidden_states = get_last_non_padded_tokens(
            hidden_states, attention_mask
        )[1:]

        distances = compute_block_distances(
            last_non_padded_hidden_states, layers_to_skip
        )
        for i, distance in enumerate(distances):
            all_distances[i].append(distance)

    average_distances = [float(np.mean(d)) for d in all_distances]

    # ============== Phase B: shortlist + data split ==========================
    order = np.argsort(average_distances)
    cand_indices = [int(i) for i in order[:top_k]]
    logging.info(
        f"{Fore.GREEN}Top-{top_k} candidates by angular distance: "
        + ", ".join(f"({i + 1},{i + 1 + layers_to_skip})" for i in cand_indices)
        + f"{Fore.RESET}"
    )

    if len(all_texts) <= heldout_num_texts:
        raise ValueError(
            f"Need more than {heldout_num_texts} texts to split calibration/"
            f"held-out; got {len(all_texts)}. Increase dataset_size."
        )
    calib_texts = all_texts[:-heldout_num_texts]
    heldout_texts = all_texts[-heldout_num_texts:]
    logging.info(f"lstsq calibration: {len(calib_texts)} texts | "
                 f"held-out CE: {len(heldout_texts)} texts (disjoint)")

    # ============== Phase C: lstsq covariances (one pass, all candidates) ====
    stats = _collect_covariances(
        model, tokenizer, calib_texts, cand_indices, layers_to_skip,
        max_length, batch_size,
    )

    # ============== Phase D: arbitration =====================================
    results = []
    for i in cand_indices:
        start, end = i + 1, i + 1 + layers_to_skip   # 0-based removal range
        W = _solve_lstsq(stats[i]["G"], stats[i]["C"])
        ce = _evaluate_candidate(
            model, tokenizer, start, end, W, heldout_texts,
            max_length, batch_size,
        )
        ppl = float(np.exp(ce))
        results.append({
            "block_start": i + 1,
            "block_end": i + 1 + layers_to_skip,
            "profile_index": i,
            "angular_distance": average_distances[i],
            "heldout_ce_per_token": ce,
            "heldout_ppl": ppl,
        })
        logging.info(
            f"{Fore.GREEN}block ({i + 1},{i + 1 + layers_to_skip}): "
            f"CE/tok={ce:.4f}  PPL={ppl:.2f}{Fore.RESET}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    results.sort(key=lambda r: r["heldout_ce_per_token"])
    winner = results[0]
    for rank, r in enumerate(results):
        r["rank"] = rank + 1
        r["winner"] = (rank == 0)
    logging.info(
        f"{Fore.GREEN}WINNER: block ({winner['block_start']},"
        f"{winner['block_end']}) with PPL {winner['heldout_ppl']:.2f}"
        f"{Fore.RESET}"
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ============== Phase E: Excel + distances.pth ===========================
    profile_rows = [{
        "block_start": i + 1,
        "block_end": i + 1 + layers_to_skip,
        "average_angular_distance": average_distances[i],
        "shortlisted": i in cand_indices,
    } for i in range(n_cand)]

    with pd.ExcelWriter(excel_save_path, engine="openpyxl") as xw:
        pd.DataFrame(profile_rows).to_excel(
            xw, sheet_name="angular_profile", index=False)
        pd.DataFrame(results).to_excel(
            xw, sheet_name="arbitration", index=False)

    # Winner forced as the argmin so select_non_overlapping_blocks picks it.
    final_distances = list(average_distances)
    final_distances[winner["profile_index"]] = 0.0
    torch.save(final_distances, distances_save_path)

    logging.info(
        f"{Fore.GREEN}Profile + arbitration -> {excel_save_path}; "
        f"distances (winner forced argmin) -> {distances_save_path}{Fore.RESET}"
    )


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run distance profiling + arbitration from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run distance analysis based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    profile_distances(**config)