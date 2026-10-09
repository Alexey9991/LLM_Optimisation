"""Memory-safe LSTSQ block replacement for ReplaceMe-style pruning.

Drop-in replacement for lstsq.py. Key properties:

  * alpha_grad == 0 (default): pure forward pass under no_grad. No autograd
    graph, no gradient tensors, ONE pair of covariance accumulators
    (hidden x hidden) instead of per-layer stacks. Fits in ~7 GiB with a
    4-bit 36-layer model -- this is the fix for the Joint-LSTSQ OOM on
    11 GiB cards.
  * alpha_grad > 0: joint activation-gradient objective solved via a
    Sylvester equation (SciPy, float64, on CPU). Needs backward passes, so
    it is guarded: refuses to run if free VRAM < 14 GiB.
  * The transform T is fused into down_proj of the layer preceding the
    removed block (no new parameters, standard checkpoint on disk).

Fusing formulation (residual-corrected, as in ReplaceMe):
  Let x1 = hidden state after layer s-1 (block input), x2 = after layer
  e-1 (block output), m1 = the MLP (down_proj) branch output of layer s-1,
  r = x1 - m1 the residual part. We re-target the MLP branch:

      minimize_T || M1 @ T - (M1 + X2 - X1) ||_F^2 + alpha ||T||_F^2

  Then W_down_new = T^T @ W_down, and the pruned layer s-1 outputs
      r + m1 @ T = (x1 - m1) + m1 + (x2 - x1) = x2   (in the lstsq sense).

Signature is kept compatible with the previous joint lstsq; unused legacy
arguments are accepted and ignored (with a log line) so existing callers
and shims keep working. `selected_blocks` is supported natively.
"""

import gc
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import get_calib_dataloader, select_non_overlapping_blocks

init(autoreset=True)
logging.basicConfig(
    format=(f"{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] "
            f"{Fore.RESET}%(message)s"),
    level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
def _bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _load_model(model_path: str, use_4bit: bool, token: Optional[str],
                need_grad: bool):
    compute_dtype = torch.bfloat16 if _bf16_ok() else torch.float16
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": 0} if torch.cuda.is_available() else "cpu",
        quantization_config=quant, output_hidden_states=True,
        token=token, trust_remote_code=True)
    model.eval()
    if need_grad:
        model.requires_grad_(False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    tok = AutoTokenizer.from_pretrained(model_path, token=token,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _mlp_output_hook(layer, store: dict):
    """Capture the down_proj (MLP branch) output of a decoder layer."""
    def hook(_m, _inp, out):
        store["m1"] = out.detach() if not out.requires_grad else out
    return layer.mlp.down_proj.register_forward_hook(hook)


def _causal_lm_loss(logits, input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=-100)


# --------------------------------------------------------------------------
def lstsq(
    model_path: str,
    dataset: str,
    dataset_column: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "train",
    activations_save_path: Optional[str] = None,     # legacy, ignored
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,        # legacy, ignored
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,                              # legacy, ignored
    alpha: float = 0.0,                              # ridge on T
    alpha_act: float = 1.0,
    alpha_grad: float = 0.0,
    normalize_inputs: bool = False,                  # legacy, ignored
    identity_blend_threshold: float = 0.0,           # legacy, ignored
    distances_path: Optional[str] = None,
    num_A: int = 1,                                  # legacy, ignored
    merge_consecutive: bool = True,
    validate_sylvester: bool = False,
    selected_blocks: Optional[List[Tuple[int, int]]] = None,
) -> str:
    """Prune one block of `layers_to_skip` layers and fuse the lstsq
    transform into the preceding layer's down_proj. Returns save dir."""

    for name, val, default in [("diag", diag, False), ("num_A", num_A, 1),
                               ("normalize_inputs", normalize_inputs, False),
                               ("identity_blend_threshold",
                                identity_blend_threshold, 0.0)]:
        if val != default:
            logging.info(f"legacy argument {name}={val} is ignored "
                         f"in memory-safe lstsq")

    need_grad = alpha_grad > 0
    if need_grad:
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 2**30
        if free < 14:
            raise RuntimeError(
                f"alpha_grad={alpha_grad} needs backward passes; free VRAM "
                f"{free:.1f} GiB < 14 GiB. Run with alpha_grad=0 on this GPU "
                f"or move to a >=16 GiB card.")

    # ---------------- block selection ----------------
    if selected_blocks:
        start, end = selected_blocks[0]
        logging.info(f"selected_blocks given: pruning ({start},{end})")
    else:
        if distances_path is None:
            raise ValueError(
                "Either selected_blocks or distances_path must be provided "
                "(this lstsq does not profile by itself).")
        distances = torch.load(distances_path, weights_only=False)
        start, end = select_non_overlapping_blocks(
            distances, layers_to_skip, num_blocks=1,
            merge_consecutive=merge_consecutive)[0]
        logging.info(f"block from {distances_path}: ({start},{end})")

    # 0-based: layers removed = range(start, end); block input = hs[start],
    # block output = hs[end]; MLP branch of layer start-1 is re-targeted.
    assert start >= 1, "cannot fuse into down_proj of layer -1 (start must be >=1)"

    # ---------------- phase 1: covariance accumulation ----------------
    model, tok = _load_model(model_path, use_4bit, token, need_grad)
    hidden = model.config.hidden_size
    dev = next(model.parameters()).device

    dataloader = get_calib_dataloader(
        dataset, dataset_subset, dataset_column, dataset_size,
        batch_size, tok)

    G = torch.zeros(hidden, hidden, dtype=torch.float64, device=dev)
    C = torch.zeros(hidden, hidden, dtype=torch.float64, device=dev)
    Bg = torch.zeros(hidden, hidden, dtype=torch.float64, device=dev) \
        if need_grad else None
    Cg = torch.zeros(hidden, hidden, dtype=torch.float64, device=dev) \
        if need_grad else None

    m1_store: dict = {}
    hook = _mlp_output_hook(model.model.layers[start - 1], m1_store)

    desc = ("Joint Activation-Gradient LSTSQ" if need_grad
            else "Forward-only LSTSQ (memory-safe)")
    ctx = torch.enable_grad if need_grad else torch.no_grad
    grad_store: dict = {}

    if need_grad:
        def keep_grad(name):
            def h(g):
                grad_store[name] = g.detach()
            return h

    for batch in tqdm(dataloader, desc=f"{Fore.GREEN}{desc}{Fore.RESET}",
                      dynamic_ncols=True, colour="green"):
        enc = tok(list(batch), return_tensors="pt", padding="longest",
                  max_length=max_length, truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}

        with ctx():
            out = model(**enc, use_cache=False)
            hs = out.hidden_states
            x1, x2 = hs[start], hs[end]
            m1 = m1_store.pop("m1")

            if need_grad:
                x2.register_hook(keep_grad("g_out"))
                m1.register_hook(keep_grad("g_in"))
                loss = _causal_lm_loss(out.logits, enc["input_ids"],
                                       enc["attention_mask"])
                loss.backward()

        mask = enc["attention_mask"].bool().view(-1)
        M1 = m1.detach().reshape(-1, hidden)[mask].double()
        Y = (m1.detach() + x2.detach() - x1.detach()) \
            .reshape(-1, hidden)[mask].double()
        G += M1.T @ M1
        C += M1.T @ Y

        if need_grad and "g_out" in grad_store and "g_in" in grad_store:
            Go = grad_store.pop("g_out").reshape(-1, hidden)[mask].double()
            Gi = grad_store.pop("g_in").reshape(-1, hidden)[mask].double()
            Bg += Go.T @ Go
            Cg += Gi.T @ Go
            model.zero_grad(set_to_none=True)

        del out, hs, x1, x2, m1, M1, Y

    hook.remove()

    # ---------------- phase 2: solve T ----------------
    A = (alpha_act * G).cpu().numpy()
    A += alpha * np.eye(hidden)
    Cn = (alpha_act * C).cpu().numpy()

    if need_grad:
        from scipy.linalg import solve_sylvester
        B = (alpha_grad * Bg).cpu().numpy()
        Cn = Cn + alpha_grad * Cg.cpu().numpy().T @ np.eye(hidden)  # G_in^T G_out
        logging.info("solving Sylvester A T + T B = C on CPU (float64)...")
        T = solve_sylvester(A, B, Cn)
        if validate_sylvester:
            res = np.linalg.norm(A @ T + T @ B - Cn) / np.linalg.norm(Cn)
            logging.info(f"Sylvester relative residual: {res:.2e}")
    else:
        A += 1e-6 * np.eye(hidden)          # numerical jitter
        T = np.linalg.solve(A, Cn)
        rel = np.linalg.norm(A @ T - Cn) / np.linalg.norm(Cn)
        logging.info(f"normal equations solved, relative residual {rel:.2e}")

    T_t = torch.from_numpy(T).float()       # [hidden, hidden]

    del model, G, C, Bg, Cg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if activations_save_path:
        torch.save({"T": T_t, "block": (start, end)}, activations_save_path)
    if save_transform_only and save_path:
        torch.save({"T": T_t, "block": (start, end)}, f"{save_path}_T.pth")

    # ---------------- phase 3: fuse + truncate + save ----------------
    logging.info("reloading full-precision model on CPU for surgery...")
    dtype = torch.bfloat16 if _bf16_ok() else torch.float16
    full = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="cpu",
        token=token, trust_remote_code=True)

    down = full.model.layers[start - 1].mlp.down_proj
    # m_new = m @ T = act @ W_down^T @ T = act @ (T^T W_down)^T
    with torch.no_grad():
        W_new = (T_t.T.to(torch.float32) @ down.weight.to(torch.float32))
        down.weight.copy_(W_new.to(down.weight.dtype))
    logging.info(f"T fused into layers[{start - 1}].mlp.down_proj")

    keep = [l for i, l in enumerate(full.model.layers)
            if i < start or i >= end]
    full.model.layers = nn.ModuleList(keep)
    full.config.num_hidden_layers = len(keep)
    lt = getattr(full.config, "layer_types", None)
    if lt is not None:
        full.config.layer_types = [t for i, t in enumerate(lt)
                                   if i < start or i >= end]

    out_dir = save_path or f"pruned_{model_path.split('/')[-1]}_{start}_{end}"
    full.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    logging.info(f"{Fore.GREEN}pruned model ({len(keep)} layers) -> "
                 f"{out_dir}{Fore.RESET}")

    del full, keep
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir