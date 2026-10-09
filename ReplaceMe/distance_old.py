"""Distance profiling module for transformer block selection (v2, multi-signal).

For every candidate cut of `layers_to_skip` consecutive layers this computes several
signals that predict how cheap the block is to remove, averaged over a calibration set:

  dist_act    - angular distance between hidden states before/after the block.
                v2: computed over ALL non-padded tokens (use_all_tokens=True), which has
                far lower variance than the classic last-token-only metric. Set
                use_all_tokens=False to reproduce v1 exactly.
  dist_grad   - (legacy v1 signal) angular distance between d(loss)/d(h) before/after
                the block. Kept for continuity; weight alpha_grad.
  taylor      - |g_after . (h_before - h_after)| averaged over tokens: the FIRST-ORDER
                Taylor estimate of the loss change if the block were replaced by
                identity. Directly loss-aware, same cost as dist_grad.
  ablate      - the strongest signal: actually remove the block (identity skip, no
                transform) and measure the per-token cross-entropy increase on a held
                subset. One extra forward pass per candidate x text. For 36 layers,
                skip=8, 32 texts this is 28*32 forwards (~10-20 min for a 4-bit 8B).

The combined score is a weighted sum of min-max-normalized signals (lower = better
candidate for removal) and is written to distances.pth, which
select_non_overlapping_blocks reads to pick the block. All raw and normalized signals
are written to layer_distances.csv for inspection and for external search tooling
(block_search.py can rank candidates by any column).

Defaults reproduce v1 behaviour-in-spirit (activation-only): alpha_act=1, others 0.

RECOMMENDED for accuracy-driven block selection on GSM8K (calibrated on Llama-3-8B
full prune->heal->restore->eval runs; Spearman(proxy, final EM) = 0.8 vs 0.6 for the
ablate-dominant preset):
    alpha_act=1.0, alpha_grad=0.0, alpha_taylor=0.25, alpha_ablate=0.25,
    ablate_num_texts=32, answer_only_loss=True
Roles: dist_act ranks WITHIN the viable valley (best single predictor of post-heal EM);
ablate acts as a VETO for catastrophic blocks (early layers, tail) — it measures
pre-heal damage, which heal largely erases inside the valley, so do not let it dominate.

Convention reminder: distances index i <-> block (start, end) = (i+1, i+1+skip);
layers removed = range(start, end); transform fused into layer start-1.
"""
import argparse
import csv
import gc
import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (compute_block_distances, compute_block_distances_all_tokens,
                    get_calib_dataloader, get_decoder_layers,
                    get_last_non_padded_tokens, seed_all)

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


# ------------------------------------------------------------------ loss helpers
def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                    reduction: str = "sum") -> torch.Tensor:
    """Cross-entropy loss for causal language modeling (shifted)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction=reduction)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def _make_labels(texts: List[str], enc: dict, tokenizer,
                 answer_only: bool, answer_marker: str) -> torch.Tensor:
    """Labels with padding masked; optionally the prompt (everything up to and
    including `answer_marker`) is masked too, so the loss reflects only answer
    generation — which is what GSM8K exact-match actually measures.

    Requires right-side padding (the default for these tokenizer calls)."""
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    if answer_only:
        for bi, txt in enumerate(texts):
            pos = txt.find(answer_marker)
            if pos == -1:
                continue
            prefix = txt[:pos + len(answer_marker)]
            plen = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            labels[bi, :min(plen, labels.shape[1])] = -100
    return labels


def _mean_ce(model, tokenizer, texts: List[str], max_length: int,
             answer_only: bool, answer_marker: str, batch_size: int = 1) -> float:
    """Mean per-token CE of `model` on `texts` (no grad). Used by the ablation phase."""
    total_nll, total_tok = 0.0, 0
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding="longest",
                        max_length=max_length, truncation=True)
        labels = _make_labels(chunk, enc, tokenizer, answer_only, answer_marker)
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        nll = _causal_lm_loss(logits, labels, reduction="sum")
        n = (labels[..., 1:] != -100).sum().item()
        total_nll += nll.item()
        total_tok += n
    return total_nll / max(total_tok, 1)


# ------------------------------------------------------------------ main profiler
def profile_distances(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "eval",
    activations_save_path: Optional[str] = None,   # unused, kept for compat
    use_4bit: bool = False,
    save_path: Optional[str] = None,               # unused, kept for compat
    min_distance_layer: Optional[int] = None,      # unused, kept for compat
    token: Optional[str] = None,
    # ---- signal weights (combined = sum of alpha_x * minmax(signal_x)) ----
    alpha_act: float = 1.0,       # activation angular distance
    alpha_grad: float = 0.0,      # legacy gradient angular distance
    alpha_taylor: float = 0.0,    # first-order |g.dh| loss-change estimate
    alpha_ablate: float = 0.0,    # true identity-skip CE increase (strongest)
    # ---- signal options ----
    use_all_tokens: bool = True,          # all non-pad tokens vs last-token-only (v1)
    ablate_num_texts: int = 32,           # texts for the ablation phase
    answer_only_loss: bool = True,        # mask the prompt in grad/ablation losses
    answer_marker: str = "Answer:",
    distances_save_path: str = "distances.pth",
    csv_save_path: str = "layer_distances.csv",
) -> List[float]:
    """Profile per-block removal signals and write the combined score to distances.pth.

    Returns the combined score list (length n_layers - layers_to_skip, lower=better).
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    if use_4bit:
        # bf16 compute needs Ampere+ (cc >= 8.0); on Turing (RTX 2080 Ti) use fp16.
        bf16_ok = (torch.cuda.is_available()
                   and torch.cuda.get_device_capability()[0] >= 8)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
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

    n_layers = model.config.num_hidden_layers
    n_cand = n_layers - layers_to_skip
    use_grad = (alpha_grad > 0) or (alpha_taylor > 0)
    use_ablate = alpha_ablate > 0 and ablate_num_texts > 0

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size, batch_size, tokenizer,
    )

    # ---- Gradient hooks: capture d(loss)/d(layer output) per layer ----
    hooks = []
    grad_store = {}
    if use_grad:
        container, attr = get_decoder_layers(model)
        layers = getattr(container, attr)

        def make_hook(i):
            def fwd(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.requires_grad:
                    h.register_hook(lambda g, i=i: grad_store.__setitem__(i, g.detach()))
            return fwd

        hooks = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    all_act = [[] for _ in range(n_cand)]
    all_grad = [[] for _ in range(n_cand)]
    all_taylor = [[] for _ in range(n_cand)]
    ablate_texts: List[str] = []

    # =========================== Phase A: activation / gradient signals ============
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Phase A: distances{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        texts = list(batch)
        if use_ablate and len(ablate_texts) < ablate_num_texts:
            ablate_texts.extend(texts[:ablate_num_texts - len(ablate_texts)])

        enc = tokenizer(
            texts, return_tensors="pt", padding="longest",
            max_length=max_length, truncation=True,
        )
        labels = _make_labels(texts, enc, tokenizer, answer_only_loss, answer_marker)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        labels = labels.to(model.device)
        attention_mask = enc["attention_mask"]

        if use_grad:
            grad_store.clear()
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**enc)
                loss = _causal_lm_loss(outputs.logits, labels)
                loss.backward()
        else:
            with torch.no_grad():
                outputs = model(**enc)

        act = list(outputs.hidden_states[1:])   # act[l] = output of layer l, [B,L,H]

        # --- activation distance ---
        if use_all_tokens:
            d_act = compute_block_distances_all_tokens(act, attention_mask, layers_to_skip)
        else:
            last = get_last_non_padded_tokens(outputs.hidden_states, attention_mask)[1:]
            d_act = compute_block_distances(last, layers_to_skip)
        for i, d in enumerate(d_act):
            all_act[i].append(d)

        # --- gradient signals ---
        if use_grad and len(grad_store) == n_layers:
            mask = attention_mask.bool()
            n_valid = mask.sum().clamp(min=1).item()

            if alpha_grad > 0:
                grad_list = [grad_store[i] for i in range(n_layers)]
                g_last = get_last_non_padded_tokens(grad_list, attention_mask)
                d_grad = compute_block_distances(g_last, layers_to_skip)
                for i, d in enumerate(d_grad):
                    all_grad[i].append(d)

            if alpha_taylor > 0:
                # taylor_i = mean_t | g_{i+skip}(t) . (h_i(t) - h_{i+skip}(t)) |
                for i in range(n_cand):
                    g = grad_store[i + layers_to_skip].float()
                    dh = (act[i] - act[i + layers_to_skip]).float()
                    t = (g * dh).sum(-1).abs()          # [B, L]
                    all_taylor[i].append((t * mask).sum().item() / n_valid)

        gc.collect()
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # =========================== Phase B: ablation (identity-skip) CE ==============
    d_ablate = [float("nan")] * n_cand
    if use_ablate:
        logging.info(f"{Fore.GREEN}Phase B: ablation over {len(ablate_texts)} texts, "
                     f"{n_cand} candidates{Fore.RESET}")
        container, attr = get_decoder_layers(model)
        orig_layers = getattr(container, attr)

        base_ce = _mean_ce(model, tokenizer, ablate_texts, max_length,
                           answer_only_loss, answer_marker, batch_size)
        logging.info(f"baseline mean CE/token: {base_ce:.4f}")

        try:
            for i in tqdm(range(n_cand), desc=f"{Fore.GREEN}Ablating blocks{Fore.RESET}",
                          dynamic_ncols=True, colour="green"):
                start, end = i + 1, i + 1 + layers_to_skip   # removed layers [start, end)
                setattr(container, attr, nn.ModuleList(
                    [l for j, l in enumerate(orig_layers) if not (start <= j < end)]
                ))
                ce = _mean_ce(model, tokenizer, ablate_texts, max_length,
                              answer_only_loss, answer_marker, batch_size)
                d_ablate[i] = ce - base_ce
        finally:
            setattr(container, attr, orig_layers)   # always restore the full model

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================== Combine, save, report =============================
    avg_act = [float(np.mean(x)) if x else float("nan") for x in all_act]
    avg_grad = [float(np.mean(x)) if x else float("nan") for x in all_grad]
    avg_taylor = [float(np.mean(x)) if x else float("nan") for x in all_taylor]

    def minmax(vals):
        v = np.array(vals, dtype=np.float64)
        if np.all(np.isnan(v)):
            return [0.0] * len(vals)
        lo, hi = np.nanmin(v), np.nanmax(v)
        rng = hi - lo
        if rng <= 0:
            return [0.0] * len(vals)
        return [0.0 if np.isnan(x) else float((x - lo) / rng) for x in v]

    n_act = minmax(avg_act)
    n_grad = minmax(avg_grad)
    n_taylor = minmax(avg_taylor)
    n_ablate = minmax(d_ablate)

    combined = [
        alpha_act * n_act[i]
        + alpha_grad * n_grad[i]
        + alpha_taylor * n_taylor[i]
        + alpha_ablate * n_ablate[i]
        for i in range(n_cand)
    ]

    with open(csv_save_path, "w", newline="") as csvfile:
        fieldnames = ["block_start", "block_end", "depth_frac",
                      "dist_act", "dist_grad", "taylor", "ablate_delta_ce",
                      "norm_act", "norm_grad", "norm_taylor", "norm_ablate",
                      "combined"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_cand):
            writer.writerow({
                "block_start": i + 1,
                "block_end": i + 1 + layers_to_skip,
                "depth_frac": round((i + 1) / n_layers, 3),
                "dist_act": avg_act[i],
                "dist_grad": avg_grad[i],
                "taylor": avg_taylor[i],
                "ablate_delta_ce": d_ablate[i],
                "norm_act": n_act[i],
                "norm_grad": n_grad[i],
                "norm_taylor": n_taylor[i],
                "norm_ablate": n_ablate[i],
                "combined": combined[i],
            })

    torch.save(combined, distances_save_path)

    order = np.argsort(combined)
    logging.info(f"{Fore.GREEN}Top-5 candidate blocks (lower combined = better):{Fore.RESET}")
    for r, i in enumerate(order[:5]):
        logging.info(
            f"  #{r + 1}: layers {i + 1}..{i + layers_to_skip} "
            f"(combined={combined[i]:.4f}, act={avg_act[i]:.4f}, "
            f"taylor={avg_taylor[i] if avg_taylor[i] == avg_taylor[i] else float('nan'):.6f}, "
            f"ablate_dCE={d_ablate[i] if d_ablate[i] == d_ablate[i] else float('nan'):.4f})"
        )
    logging.info(f"{Fore.GREEN}Signals -> {csv_save_path}; combined -> "
                 f"{distances_save_path}{Fore.RESET}")
    return combined


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run multi-signal distance profiling from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run multi-signal block-removal profiling from a configuration file."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    config = read_config(args.config)
    profile_distances(**config)