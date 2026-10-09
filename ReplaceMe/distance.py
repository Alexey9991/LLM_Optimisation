"""Distance profiling module for transformer block selection (final version).

Consolidates the D0->D8 ablation ladder (Llama-3-8B / GSM8K, July 2026).
Every design decision below was accepted or rejected experimentally:

  [D1] Engineering: argmin computed once (no argument shadowing), periodic
       memory cleanup instead of per-batch, capability-aware 4-bit compute
       dtype (fp16 on Turing, bf16 on Ampere+), output paths as parameters.
  [D3] dist_act is averaged over ALL non-padded tokens (use_all_tokens=True
       by default). Halves the between-subset variance of the profile
       (rel_std 0.0042 -> 0.0018). use_all_tokens=False reproduces the v1
       last-token metric exactly.
  [D4] The masked all-token metric is invariant to batch_size / padding
       (max rel diff 1.6e-3 between bs=1 and bs=4), so batch_size > 1 is
       safe and faster.
  [D5] lin_res (ridge-regression residual h_in @ W ~= h_out) is REJECTED as
       a ranker: it is monotone in depth (Spearman with post-heal EM = -0.40)
       because the energy-weighted Frobenius residual is dominated by trivially
       linear high-norm directions. Kept as an optional DIAGNOSTIC column.
  [D6] taylor = mean_t |g_out . (h_in - h_out)| (first-order loss change under
       identity ablation) breaks the depth artifact (U-shaped profile) but is
       NOT a fine ranker either (its argmin lost the D8 arbitration). Role:
       PRE-HEAL DAMAGE VETO for catastrophic blocks. Optional (needs backward).
  [D7] True identity-skip ablation: skipped. taylor delivers the same signal
       class at ~1/10 the cost (99 s vs 10-20 min), and ablation's Spearman
       with post-heal EM was ~0 in independent calibration.
  [D8] FINAL PROCEDURE: proxy signals only SHORTLIST candidates; the decision
       is made by direct measurement: run lstsq (no healing) on the top-k
       candidates and compare held-out cross-entropy. In the ladder this
       procedure selected block (21, 29) -- the true best-known block --
       while both proxy argmins ((22,30) for act, (18,26) for taylor) lost.

Convention (unchanged from v1): distances index i <-> block
(start, end) = (i+1, i+1+layers_to_skip); layers removed = range(start, end);
the transform is fused into layer start-1. distances.pth is a plain list
readable by select_non_overlapping_blocks.
"""

import argparse
import csv
import gc
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (compute_block_distances, get_calib_dataloader,
                    get_last_non_padded_tokens, seed_all)

init(autoreset=True)
logging.basicConfig(
    format=(f"{Fore.CYAN}%(asctime)s "
            f"{Fore.YELLOW}[%(levelname)s] "
            f"{Fore.RESET}%(message)s"),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
seed_all()


# ===================================================================== helpers
def _quant_config(use_4bit: bool):
    """[D1] Capability-aware 4-bit config (fp16 compute on pre-Ampere GPUs)."""
    if not use_4bit:
        return None
    bf16_ok = (torch.cuda.is_available()
               and torch.cuda.get_device_capability()[0] >= 8)
    if not bf16_ok:
        logging.info("GPU capability < 8.0: using float16 compute for 4-bit")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )


def compute_block_distances_all_tokens(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    layers_to_skip: int,
    eps: float = 1e-7,
) -> List[float]:
    """[D3] Angular distance per candidate over ALL non-padded tokens.

    hidden_states: list of (B, S, D); element l = output of layer l+1
    (the embedding entry must already be stripped by the caller).
    """
    mask_flat = attention_mask.bool().view(-1)
    out: List[float] = []
    for i in range(len(hidden_states) - layers_to_skip):
        d = hidden_states[i].shape[-1]
        a = hidden_states[i].reshape(-1, d)[mask_flat].float()
        b = hidden_states[i + layers_to_skip].reshape(-1, d)[mask_flat].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        ang = torch.arccos(cos.clamp(-1 + eps, 1 - eps)) / torch.pi
        out.append(ang.mean().item())
    return out


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    sl = logits[..., :-1, :].contiguous()
    st = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        sl.view(-1, sl.size(-1)), st.view(-1),
        ignore_index=-100, reduction="sum")


def _make_labels(texts, input_ids, attention_mask, tokenizer,
                 answer_only: bool, marker: str = "Answer:"):
    """Padding masked; optionally the prompt up to `marker` is masked too."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if answer_only:
        for bi, txt in enumerate(texts):
            pos = txt.find(marker)
            if pos == -1:
                continue
            plen = len(tokenizer(txt[:pos + len(marker)],
                                 add_special_tokens=True)["input_ids"])
            labels[bi, :min(plen, labels.shape[1])] = -100
    return labels


def _minmax(vals: Sequence[float]) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64)
    if np.all(np.isnan(v)):
        return np.zeros_like(v)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(hi - lo) or hi - lo <= 0:
        return np.zeros_like(v)
    return np.nan_to_num((v - lo) / (hi - lo), nan=0.0)


# ================================================================ main profiler
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
    min_distance_layer: Optional[int] = None,      # unused, kept for compat [D1]
    token: Optional[str] = None,
    # ---- metric [D3/D4] ----
    use_all_tokens: bool = True,
    # ---- optional diagnostic / veto signals [D5/D6] ----
    compute_lin_res: bool = False,     # [D5] diagnostic only; NEVER a ranker
    compute_taylor: bool = False,      # [D6] pre-heal damage veto (needs backward)
    compute_grad_cos: bool = False,    # [D9] angular distance between g_in and g_out
                                       #      (Jacobian-vs-identity probe; needs backward)
    answer_only_loss: bool = False,    # loss mask for taylor (GSM8K-style data)
    answer_marker: str = "Answer:",
    ridge_lambda: float = 1e-3,
    # ---- combination [D8] ----
    alpha_act: float = 1.0,            # dist_act is the only default ranker
    taylor_veto_quantile: float = 0.0, # e.g. 0.75: exclude worst-quartile taylor
    # ---- engineering [D1] ----
    cleanup_every: int = 20,
    distances_save_path: str = "distances.pth",
    csv_save_path: str = "layer_distances.csv",
    top_k_report: int = 4,
) -> Dict[str, List[float]]:
    """Profile candidate blocks and write the ranking score to distances.pth.

    Default configuration (all-token dist_act only) is the validated ranker.
    The score written to distances.pth is dist_act-based with an optional
    taylor veto; extra signals go to the CSV as diagnostics.

    IMPORTANT [D8]: the profile is a SHORTLIST, not a decision. For the final
    choice run `arbitrate_blocks` on the top-k candidates (direct held-out CE
    of the actually-pruned models). In our experiments the proxy argmin was
    wrong and arbitration recovered the true best block.

    Returns a dict with all computed signals and the final 'score' list.
    """
    need_grad = compute_taylor or compute_grad_cos

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        quantization_config=_quant_config(use_4bit),
        output_hidden_states=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    if need_grad:
        if use_4bit:
            logging.warning("taylor with 4-bit weights: gradients are noisy "
                            "(acceptable for the veto role, not for ranking)")
        # [D6] gradients w.r.t. activations only; weights stay frozen
        model.requires_grad_(False)
        try:
            model.enable_input_require_grads()
        except AttributeError:
            model.get_input_embeddings().register_forward_hook(
                lambda m, i, o: o.requires_grad_(True))

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column,
        dataset_size, batch_size, tokenizer)

    D = model.config.hidden_size
    n_cand = model.config.num_hidden_layers - layers_to_skip

    acc_act = [[] for _ in range(n_cand)]
    acc_tay = [[] for _ in range(n_cand)]
    acc_gcos = [[] for _ in range(n_cand)]
    if compute_lin_res:
        xtx = [torch.zeros(D, D, dtype=torch.float64) for _ in range(n_cand)]
        xty = [torch.zeros(D, D, dtype=torch.float64) for _ in range(n_cand)]
        yty = [0.0] * n_cand

    logging.info(
        f"metric: {'all-token [D3]' if use_all_tokens else 'last-token (v1)'}"
        f" | lin_res: {compute_lin_res} | taylor: {compute_taylor}"
        f" | dist_grad: {compute_grad_cos}")

    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Profiling blocks{Fore.RESET}",
        dynamic_ncols=True, colour="green",
    )):
        texts = list(batch)
        enc = tokenizer(texts, return_tensors="pt", padding="longest",
                        max_length=max_length, truncation=True)
        if need_grad:
            labels = _make_labels(texts, enc["input_ids"], enc["attention_mask"],
                                  tokenizer, answer_only_loss, answer_marker)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        am = enc["attention_mask"]

        if need_grad:
            labels = labels.to(model.device)
            with torch.enable_grad():
                outputs = model(**enc)
                hs_full = outputs.hidden_states
                for h in hs_full:
                    if h.requires_grad:
                        h.retain_grad()
                loss = _causal_lm_loss(outputs.logits, labels)
                loss.backward()
        else:
            with torch.no_grad():
                outputs = model(**enc)
            hs_full = outputs.hidden_states
            loss = None

        hs = list(hs_full[1:])                 # hs[l] = output of layer l+1
        m_flat = am.bool().view(-1)

        # --- dist_act [D3] ---
        if use_all_tokens:
            dists = compute_block_distances_all_tokens(
                [h.detach() for h in hs], am, layers_to_skip)
        else:
            last = get_last_non_padded_tokens(
                [h.detach() for h in hs_full], am)[1:]
            dists = compute_block_distances(last, layers_to_skip)
        for i, d in enumerate(dists):
            acc_act[i].append(d)

        # --- lin_res accumulators [D5, diagnostic] ---
        if compute_lin_res:
            for i in range(n_cand):
                x = hs[i].detach().reshape(-1, D)[m_flat].float()
                y = hs[i + layers_to_skip].detach().reshape(-1, D)[m_flat].float()
                xtx[i] += (x.t() @ x).double().cpu()
                xty[i] += (x.t() @ y).double().cpu()
                yty[i] += float((y * y).sum().double())

        # --- taylor [D6, veto] ---
        if compute_taylor:
            n_valid = int(am.sum().clamp(min=1))
            for i in range(n_cand):
                g = hs[i + layers_to_skip].grad
                if g is None:
                    continue
                dh = (hs[i].detach() - hs[i + layers_to_skip].detach()).float()
                t = (g.float() * dh).sum(-1).abs()
                acc_tay[i].append(float((t * am).sum()) / n_valid)

        # --- dist_grad [D9, experimental]: angular distance g_in vs g_out ---
        # g_in = J^T g_out, so this probes how far the block's Jacobian is from
        # identity along the directions the loss actually cares about.
        # Diagnostic only until it clears the acceptance criteria; never a ranker.
        if compute_grad_cos:
            for i in range(n_cand):
                g_in = hs[i].grad
                g_out = hs[i + layers_to_skip].grad
                if g_in is None or g_out is None:
                    continue
                a = g_in.detach().reshape(-1, D)[m_flat].float()
                b = g_out.detach().reshape(-1, D)[m_flat].float()
                cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                ang = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7)) / torch.pi
                acc_gcos[i].append(float(ang.mean()))

        if need_grad:
            model.zero_grad(set_to_none=True)
            for h in hs_full:
                h.grad = None

        del outputs, hs_full, hs
        if loss is not None:
            del loss
        if (batch_idx + 1) % cleanup_every == 0:      # [D1]
            gc.collect()
            torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- finalize signals ----------------
    dist_act = [float(np.mean(v)) for v in acc_act]
    taylor = [float(np.mean(v)) if v else float("nan") for v in acc_tay]
    dist_grad = [float(np.mean(v)) if v else float("nan") for v in acc_gcos]

    lin_res = [float("nan")] * n_cand
    if compute_lin_res:
        eye = torch.eye(D, dtype=torch.float64)
        for i in range(n_cand):
            reg = ridge_lambda * torch.trace(xtx[i]) / D * eye
            w = torch.linalg.solve(xtx[i] + reg, xty[i])
            r2 = (yty[i] - 2 * float((w * xty[i]).sum())
                  + float((w * (xtx[i] @ w)).sum()))
            lin_res[i] = max(r2, 0.0) / max(yty[i], 1e-12)

    # ---------------- score [D8]: act ranker + optional taylor veto ----------
    score = (alpha_act * _minmax(dist_act)).tolist()
    vetoed: List[int] = []
    if compute_taylor and taylor_veto_quantile > 0:
        thr = float(np.nanquantile(np.asarray(taylor), taylor_veto_quantile))
        for i, t in enumerate(taylor):
            if np.isfinite(t) and t > thr:
                score[i] = float("inf")
                vetoed.append(i)
        logging.info(f"taylor veto (>q{taylor_veto_quantile:.2f}={thr:.4g}): "
                     f"excluded {len(vetoed)} candidates")

    # ---------------- outputs ----------------
    with open(csv_save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "block_start", "block_end", "dist_act", "dist_grad", "lin_res",
            "taylor", "score", "vetoed"])
        writer.writeheader()
        for i in range(n_cand):
            writer.writerow({
                "block_start": i + 1,
                "block_end": i + 1 + layers_to_skip,
                "dist_act": dist_act[i],
                "dist_grad": dist_grad[i],
                "lin_res": lin_res[i],
                "taylor": taylor[i],
                "score": score[i],
                "vetoed": i in vetoed,
            })

    torch.save(score, distances_save_path)

    order = np.argsort([s if np.isfinite(s) else np.inf for s in score])
    logging.info(f"{Fore.GREEN}Top-{top_k_report} candidates "
                 f"(shortlist for arbitration, NOT the final answer):{Fore.RESET}")
    for r, i in enumerate(order[:top_k_report], 1):
        logging.info(f"  #{r}: block ({i+1}, {i+1+layers_to_skip}) "
                     f"score={score[i]:.4f} act={dist_act[i]:.4f}"
                     + (f" taylor={taylor[i]:.4g}" if compute_taylor else ""))
    logging.info(f"{Fore.GREEN}signals -> {csv_save_path}; "
                 f"score -> {distances_save_path}{Fore.RESET}")
    logging.info("Recommended next step [D8]: arbitrate_blocks(...) on the "
                 "top candidates for the direct held-out-CE decision.")

    return {"dist_act": dist_act, "dist_grad": dist_grad, "lin_res": lin_res,
            "taylor": taylor, "score": score}


# ============================================================== arbitration [D8]
def arbitrate_blocks(
    candidates: Sequence[Tuple[int, int]],
    heldout_texts: Sequence[str],
    model_path: str,
    layers_to_skip: int,
    lstsq_kwargs: Optional[dict] = None,
    max_length: int = 256,
    use_4bit_eval: bool = True,
    distances_save_path: Optional[str] = "distances.pth",
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """[D8] Direct arbitration: prune each candidate with lstsq (no healing)
    and compare mean held-out cross-entropy per token. The winner is forced
    to be the argmin of `distances_save_path` so the standard pipeline
    (select_non_overlapping_blocks -> lstsq) picks it automatically.

    heldout_texts must NOT overlap with the lstsq calibration texts.
    lstsq_kwargs are forwarded to ReplaceMe.lstsq.lstsq (model_path,
    layers_to_skip and selected_blocks are set here).
    """
    from . import lstsq as _lstsq_mod    # lazy import, avoids cycles

    base_kwargs = dict(model_path=model_path, layers_to_skip=layers_to_skip)
    base_kwargs.update(lstsq_kwargs or {})

    results: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (s, e) in candidates:
        logging.info(f"{Fore.GREEN}=== arbitration candidate ({s},{e}) ==={Fore.RESET}")
        kw = dict(base_kwargs)
        kw["selected_blocks"] = [(s, e)]
        kw.setdefault("save_path", f"arbitr_{s}_{e}")
        out_dir = _lstsq_mod.lstsq(**kw)
        gc.collect()
        torch.cuda.empty_cache()

        m = AutoModelForCausalLM.from_pretrained(
            out_dir, device_map={"": 0} if torch.cuda.is_available() else "cpu",
            quantization_config=_quant_config(use_4bit_eval))
        t = AutoTokenizer.from_pretrained(out_dir)
        if not t.pad_token:
            t.pad_token = t.eos_token
        m.eval()

        tot_nll, tot_tok = 0.0, 0
        for txt in tqdm(heldout_texts, desc=f"held-out CE ({s},{e})",
                        dynamic_ncols=True):
            enc = t(txt, return_tensors="pt", truncation=True,
                    max_length=max_length)
            labels = enc["input_ids"].clone()
            enc = {k: v.to(m.device) for k, v in enc.items()}
            labels = labels.to(m.device)
            with torch.no_grad():
                logits = m(**enc).logits
            sl = logits[..., :-1, :].contiguous()
            st = labels[..., 1:].contiguous()
            tot_nll += float(torch.nn.functional.cross_entropy(
                sl.view(-1, sl.size(-1)), st.view(-1), reduction="sum"))
            tot_tok += st.numel()
        del m
        gc.collect()
        torch.cuda.empty_cache()

        ce = tot_nll / max(tot_tok, 1)
        results[(s, e)] = {"ce_per_token": ce, "ppl": float(np.exp(ce)),
                           "model_dir": out_dir}
        logging.info(f"({s},{e}): CE/tok={ce:.4f}  PPL={np.exp(ce):.2f}")

    winner = min(results, key=lambda b: results[b]["ce_per_token"])
    logging.info(f"{Fore.GREEN}ARBITRATION WINNER: {winner} "
                 f"(PPL {results[winner]['ppl']:.2f}){Fore.RESET}")

    if distances_save_path:
        n_layers_guess = max(e for _, e in candidates) - 1 + layers_to_skip
        # write a profile whose argmin is the winner (v1-compatible format)
        prof = [1.0] * max(n_layers_guess - layers_to_skip,
                           max(s for s, _ in candidates))
        prof[winner[0] - 1] = 0.0
        torch.save(prof, distances_save_path)
        logging.info(f"winner written as argmin -> {distances_save_path}")

    return results


# ==================================================================== config CLI
def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-signal block profiling (final ladder version).")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    profile_distances(**read_config(args.config))