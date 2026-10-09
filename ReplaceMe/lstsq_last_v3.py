"""Least Squares Transformation module for transformer model optimization (v2).

Computes linear transformations between transformer model layers using joint
activation-gradient least squares estimation via Sylvester equation solving:

    alpha_act * A1^T A1 * W  +  W * alpha_grad * G2^T G2
        =  alpha_act * A1^T A2  +  alpha_grad * G1^T G2

which is the stationarity condition of the joint objective

    J(W) = alpha_act * ||A1 W - A2||_F^2  +  alpha_grad * ||W G2^T - G1^T||_F^2,

i.e. the forward map matches activations (A1 = MLP output of the layer before the cut,
A2 = residual-corrected block output) and the backward map W^T matches gradients
(G2 = dL/dh at the block output incl. residual path, G1 = dL/d(MLP output)).
The transform is fused into mlp.down_proj of the layer before the cut:
W_down_new = W^T @ W_down, so that y_new = y @ W.

v2 changes vs v1 (same public interface; all v1 kwargs still accepted):
 1. SYLVESTER SOLVE ALWAYS ON CPU (float64). The v1 GPU float64 eigh OOM'd on 11GB
    cards, silently fell back to a diagonal transform, and produced a broken model.
 2. NO silent diagonal fallback: solver failure now raises (on_solver_failure="raise",
    set "identity" to fall back to an identity transform with a loud warning instead).
 3. alpha_grad == 0 -> NO backward pass at all (~2-3x faster activation-only runs).
 4. Hooks are registered ONLY on the layers each selected block actually needs,
    instead of every layer (less VRAM, less Python overhead).
 5. Padding tokens are excluded from all accumulated statistics via attention_mask
    (v1 silently polluted A1^T A1 with pad rows whenever batch_size > 1).
 6. balance_terms=True (default): the activation and gradient terms are normalized by
    their traces before applying alpha_act / alpha_grad. Raw gradient magnitudes are
    orders of magnitude smaller than activations, so in v1 alpha_grad=0.5 barely
    changed W; with balancing, alpha_grad in [0, 1] is genuinely comparable.
    Set balance_terms=False for exact v1 semantics.
 7. selected_blocks parameter: pass [(start, end), ...] explicitly (same convention as
    select_non_overlapping_blocks) to bypass distances.pth. Used by block-search
    tooling; the distances-based selection stays the default.
 8. Per-batch GEMM in float32 on GPU, accumulation in float64 (v1 did fp64 GEMM on
    the GPU, ~30x slower on consumer cards).
 9. save_accumulators_path: optionally save the raw accumulated matrices, then re-solve
    for ANY (alpha_act, alpha_grad, alpha) via solve_from_accumulators() in seconds --
    the alpha_grad sweep no longer needs a full calibration pass per value.
10. bnb compute dtype auto-selects fp16 on pre-Ampere GPUs (RTX 2080 Ti has no bf16).
11. lstsq_diagnostics.json (blocks, residuals, token counts, config) is always saved
    into the output dir -- block-search tooling can rank candidates with it.
"""

import argparse
import gc
import json
import logging
import os
from typing import List, Optional, Tuple

import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, get_decoder_layers,
                    select_non_overlapping_blocks, seed_all, truncate_model)

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


# ------------------------------------------------------------------ solver
def solve_sylvester_standard(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                             validate: bool = False) -> torch.Tensor:
    """Solve A W + W B = C via eigendecomposition of the symmetric A and B.

    ALWAYS solves on CPU in float64: the fp64 eigh workspace for 4096x4096 matrices
    does not fit next to a quantized 8B model on an 11GB GPU, and v1's silent
    diagonal fallback after the OOM produced garbage transforms.

    A = U Lambda U^T, B = V Sigma V^T  =>  W = U [ (U^T C V)_{ij} / (l_i + s_j) ] V^T
    """
    dtype = torch.float64

    with torch.no_grad():
        A = A.detach().to("cpu", dtype)
        B = B.detach().to("cpu", dtype)
        C = C.detach().to("cpu", dtype)

        # Enforce exact symmetry (accumulation noise breaks eigh's assumptions).
        A = 0.5 * (A + A.t())
        B = 0.5 * (B + B.t())

        Lambda, U = torch.linalg.eigh(A)
        Sigma, V = torch.linalg.eigh(B)

        # PSD by construction; clamp tiny negative eigenvalues from rounding.
        Lambda = Lambda.clamp(min=0.0)
        Sigma = Sigma.clamp(min=0.0)

        R_rhs = U.t() @ C @ V
        denom = Lambda.unsqueeze(1) + Sigma.unsqueeze(0)
        # Relative floor: near-zero (l_i + s_j) would amplify noise into W.
        floor = max(1e-10, 1e-9 * float(denom.max()))
        R = R_rhs / denom.clamp(min=floor)

        W = U @ R @ V.t()

        if validate:
            residual = A @ W + W @ B - C
            rel_norm = torch.norm(residual, p="fro") / (torch.norm(C, p="fro") + 1e-10)
            print(f"[Validation] ||AW + WB - C||_F / ||C||_F: {rel_norm:.4e}")

        return W


def _relative_residual(A, B, C, W) -> float:
    """||AW + WB - C||_F / ||C||_F on CPU float64."""
    A, B, C, W = [x.detach().to("cpu", torch.float64) for x in (A, B, C, W)]
    r = A @ W + W @ B - C
    return float(torch.norm(r, p="fro") / (torch.norm(C, p="fro") + 1e-10))


def blend_with_identity(W: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """Blend transform W with identity to prevent overfitting (v1 semantics):
    W_blended = (1 - a) W + a I, a = min(threshold, ||W - I||_F / ||W||_F)."""
    if W is None:
        return None

    m = W.shape[0]
    I = torch.eye(m, dtype=W.dtype, device=W.device)

    norm_diff = torch.norm(W - I, p="fro")
    norm_W = torch.norm(W, p="fro")

    alpha_blend = min(threshold, (norm_diff / (norm_W + 1e-10)).item())
    return (1 - alpha_blend) * W + alpha_blend * I


def _compute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for causal language modeling."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ------------------------------------------------------------------ re-solve helper
def solve_from_accumulators(
    accumulators_path: str,
    alpha_act: float = 1.0,
    alpha_grad: float = 1.0,
    alpha: float = 0.0,
    balance_terms: bool = True,
    identity_blend_threshold: float = 0.0,
    validate: bool = True,
) -> List[torch.Tensor]:
    """Re-solve the Sylvester equations for new alphas from saved raw accumulators
    (see save_accumulators_path in lstsq). Runs in seconds -- use this to sweep
    alpha_grad without re-running the calibration pass. Returns list of transforms
    (bfloat16), one per block, in the same order as saved selected_blocks."""
    payload = torch.load(accumulators_path, map_location="cpu", weights_only=False)
    transforms = []
    for acc in payload["blocks"]:
        A_mat, B_mat, C_mat = _assemble_system(
            acc["S_aa"], acc["S_ab"], acc["S_gg"], acc["S_g1g2"],
            alpha_act, alpha_grad, alpha, balance_terms,
        )
        W = solve_sylvester_standard(A_mat, B_mat, C_mat, validate=validate)
        if identity_blend_threshold > 0:
            W = blend_with_identity(W, threshold=identity_blend_threshold)
        transforms.append(W.to(torch.bfloat16))
    return transforms


def _assemble_system(S_aa, S_ab, S_gg, S_g1g2,
                     alpha_act, alpha_grad, alpha, balance_terms):
    """Build (A, B, C) of the Sylvester system from RAW accumulated matrices.

    balance_terms=True normalizes each term by its trace so alpha_act/alpha_grad
    weigh the two objectives comparably regardless of raw magnitudes.
    """
    d = S_aa.shape[0]
    S_aa = S_aa.to(torch.float64)
    S_ab = S_ab.to(torch.float64)
    S_gg = S_gg.to(torch.float64)
    S_g1g2 = S_g1g2.to(torch.float64)

    if balance_terms:
        s_a = float(torch.diagonal(S_aa).sum() / d)
        s_g = float(torch.diagonal(S_gg).sum() / d)
        s_a = s_a if s_a > 0 else 1.0
        s_g = s_g if s_g > 0 else 1.0
    else:
        s_a = s_g = 1.0

    reg = alpha * torch.eye(d, dtype=torch.float64, device=S_aa.device)
    # Regularization placement matches v1 (Eq. 16 in the report):
    # A += lambda I;  B += alpha_grad * lambda I.
    A_mat = (alpha_act / s_a) * S_aa + reg
    B_mat = (alpha_grad / s_g) * S_gg + alpha_grad * reg
    C_mat = (alpha_act / s_a) * S_ab + (alpha_grad / s_g) * S_g1g2
    return A_mat, B_mat, C_mat


# ------------------------------------------------------------------ main
def lstsq(
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
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,      # unused, kept for compat
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    alpha: float = 0.0,            # Tikhonov regularization (lambda in the report)
    alpha_act: float = 0.3,        # forward (activation) weight
    alpha_grad: float = 1.0,       # backward (gradient) weight
    normalize_inputs: bool = True,  # deprecated no-op in v1; superseded by balance_terms
    identity_blend_threshold: float = 0.15,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    validate_sylvester: bool = True,
    # ---- v2 additions ----
    selected_blocks: Optional[List[Tuple[int, int]]] = None,
    balance_terms: bool = True,
    on_solver_failure: str = "raise",       # "raise" | "identity"
    save_accumulators_path: Optional[str] = None,
    save_dtype: str = "bfloat16",           # dtype of the saved pruned model
) -> str:
    """Estimate joint activation-gradient transforms and save the pruned model.

    Returns the output directory: f"{save_path}_ReplaceMe_joint_lstsq_{num_A}".
    """
    use_grad = alpha_grad > 0
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None

    if use_4bit:
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
    hidden_size = model.config.hidden_size

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer,
    )

    # ---- Resolve blocks: explicit override or distances-based selection ----
    if selected_blocks is None:
        average_distances = torch.load(distances_path, weights_only=False)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            layers_to_skip,
            num_blocks=num_A,
            merge_consecutive=merge_consecutive,
        )
    else:
        selected_blocks = [tuple(b) for b in selected_blocks]
        logging.info(f"Using explicitly provided blocks: {selected_blocks}")

    # Validate: sorted, non-overlapping, start >= 1 (layer start-1 must exist).
    selected_blocks = sorted(selected_blocks)
    for (s, e) in selected_blocks:
        assert 1 <= s < e <= model.config.num_hidden_layers, f"bad block {(s, e)}"
    for (_, e_prev), (s_next, _) in zip(selected_blocks, selected_blocks[1:]):
        assert s_next >= e_prev, f"overlapping blocks: {selected_blocks}"

    start_ids = [b[0] for b in selected_blocks]
    end_ids = [b[1] for b in selected_blocks]
    n_blocks = len(selected_blocks)

    # ---- Hooks ONLY on the layers each block needs ----
    container, attr = get_decoder_layers(model)
    layers = getattr(container, attr)

    mlp_activations = {}     # 'layer_{i}_mlp'   -> MLP output (a1 source)
    mlp_gradients = {}       # 'layer_{i}_mlp'   -> dL/d(MLP output)  (g1)
    block_gradients = {}     # 'layer_{i}_block' -> dL/d(block output) (g2)

    def save_mlp_activation(name: str):
        def hook(module, input, output):
            mlp_activations[name] = output.detach().clone()
            if use_grad and output.requires_grad:
                def grad_hook(grad):
                    mlp_gradients[name] = grad.detach().clone()
                output.register_hook(grad_hook)
        return hook

    def save_block_gradient(name: str):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.requires_grad:
                def grad_hook(grad):
                    block_gradients[name] = grad.detach().clone()
                out.register_hook(grad_hook)
        return hook

    hooks = []
    for (s, e) in selected_blocks:
        pre_layer = s - 1        # layer whose MLP hosts the transform
        last_layer = e - 1       # last removed layer (block output = h_after)
        hooks.append(layers[pre_layer].mlp.register_forward_hook(
            save_mlp_activation(f"layer_{pre_layer}_mlp")))
        if use_grad:
            hooks.append(layers[last_layer].register_forward_hook(
                save_block_gradient(f"layer_{last_layer}_block")))

    # ---- Raw accumulators: float64 storage, float32 per-batch GEMM ----
    accum_device = "cuda" if torch.cuda.is_available() else "cpu"
    S_aa = [torch.zeros(hidden_size, hidden_size, device=accum_device, dtype=torch.float64) for _ in range(n_blocks)]
    S_ab = [torch.zeros(hidden_size, hidden_size, device=accum_device, dtype=torch.float64) for _ in range(n_blocks)]
    S_gg = [torch.zeros(hidden_size, hidden_size, device=accum_device, dtype=torch.float64) for _ in range(n_blocks)]
    S_g1g2 = [torch.zeros(hidden_size, hidden_size, device=accum_device, dtype=torch.float64) for _ in range(n_blocks)]

    n_tokens_used = 0
    outputs = None
    for batch_idx, batch in enumerate(tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Running Joint Activation-Gradient LSTSQ{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    )):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = labels.to(model.device)
        mask_flat = inputs["attention_mask"].reshape(-1).bool()

        # Forward (+ backward only if the gradient term is actually used).
        if use_grad:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**inputs, use_cache=False)
                loss = _compute_causal_lm_loss(outputs.logits, labels)
                loss.backward()
        else:
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

        hidden_states = outputs.hidden_states[1:]   # [l] = output of layer l
        n_tokens_used += int(mask_flat.sum())

        for idx in range(n_blocks):
            start_layer = start_ids[idx] - 1   # 0-indexed layer before the cut
            end_layer = end_ids[idx] - 1       # 0-indexed last removed layer

            a1_key = f"layer_{start_layer}_mlp"
            if a1_key not in mlp_activations:
                logging.warning(f"Missing activations for block {idx}, skipping batch")
                continue

            def rows(t):
                """[B, L, H] -> [N_valid, H] float32 rows (pad tokens dropped)."""
                return t.reshape(-1, hidden_size)[mask_flat].to(torch.float32)

            # ReplaceMe residual formulation: Y = mlp_out, target = h_after - (h_before - Y)
            a1 = rows(mlp_activations[a1_key])
            h_before = rows(hidden_states[start_layer])
            h_after = rows(hidden_states[end_layer])
            a2 = h_after + a1 - h_before

            S_aa[idx] += (a1.t() @ a1).to(torch.float64)
            S_ab[idx] += (a1.t() @ a2).to(torch.float64)

            if use_grad:
                g1_raw = mlp_gradients.get(a1_key)
                g2_raw = block_gradients.get(f"layer_{end_layer}_block")
                if g1_raw is None or g2_raw is None:
                    logging.warning(f"Missing gradients for block {idx}, skipping batch")
                    continue
                g1 = rows(g1_raw)
                g2 = rows(g2_raw)
                S_gg[idx] += (g2.t() @ g2).to(torch.float64)
                S_g1g2[idx] += (g1.t() @ g2).to(torch.float64)

        mlp_activations.clear()
        mlp_gradients.clear()
        block_gradients.clear()

        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ---- Optionally save raw accumulators for offline alpha sweeps ----
    if save_accumulators_path:
        torch.save({
            "blocks": [{
                "block": selected_blocks[i],
                "S_aa": S_aa[i].cpu(), "S_ab": S_ab[i].cpu(),
                "S_gg": S_gg[i].cpu(), "S_g1g2": S_g1g2[i].cpu(),
            } for i in range(n_blocks)],
            "selected_blocks": selected_blocks,
            "n_tokens": n_tokens_used,
            "model_path": model_path,
        }, save_accumulators_path)
        logging.info(f"Raw accumulators saved -> {save_accumulators_path} "
                     f"(re-solve any alphas via solve_from_accumulators)")

    # ---- Solve per block (ALWAYS on CPU float64; no silent fallback) ----
    transforms = []
    diagnostics = []
    for idx in range(n_blocks):
        if torch.all(S_aa[idx] == 0):
            raise RuntimeError(
                f"No valid calibration data accumulated for block {selected_blocks[idx]} "
                f"— check the dataloader and hooks before trusting any output."
            )

        A_mat, B_mat, C_mat = _assemble_system(
            S_aa[idx], S_ab[idx], S_gg[idx], S_g1g2[idx],
            alpha_act, alpha_grad, alpha, balance_terms,
        )

        if diag:
            # Deliberate diagonal solve (an intentional option, NOT a fallback).
            denom = torch.diagonal(A_mat) + torch.diagonal(B_mat)
            W64 = torch.diag(torch.diagonal(C_mat) / (denom + 1e-8))
        else:
            try:
                W64 = solve_sylvester_standard(A_mat, B_mat, C_mat,
                                               validate=validate_sylvester)
            except Exception as e:
                if on_solver_failure == "identity":
                    logging.error(f"Sylvester solver failed for block "
                                  f"{selected_blocks[idx]}: {e}. USING IDENTITY — "
                                  f"the pruned model will be weaker than expected.")
                    W64 = torch.eye(hidden_size, dtype=torch.float64)
                else:
                    raise RuntimeError(
                        f"Sylvester solver failed for block {selected_blocks[idx]}. "
                        f"v1 silently fell back to a diagonal transform here, which "
                        f"produced broken models — failing loudly instead."
                    ) from e

        rel_res = _relative_residual(A_mat, B_mat, C_mat, W64)
        diagnostics.append({
            "block": list(selected_blocks[idx]),
            "relative_residual": rel_res,
            "w_fro_norm": float(torch.norm(W64, p="fro")),
            "w_minus_I_fro_norm": float(torch.norm(
                W64 - torch.eye(hidden_size, dtype=torch.float64), p="fro")),
        })
        logging.info(f"Block {selected_blocks[idx]}: relative residual {rel_res:.4e}")

        transform = W64.to(torch.bfloat16)
        if identity_blend_threshold > 0:
            transform = blend_with_identity(transform, threshold=identity_blend_threshold)
            logging.info(f"Block {idx}: applied identity blending "
                         f"(threshold={identity_blend_threshold})")
        transforms.append(transform)

    # ---- Clean up the calibration model ----
    for hook in hooks:
        hook.remove()
    del model, outputs, S_aa, S_ab, S_gg, S_g1g2
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Reload on CPU, fuse transforms, truncate, save ----
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    save_torch_dtype = dtype_map.get(save_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=save_torch_dtype,
    )
    container, attr = get_decoder_layers(model)

    for i in range(len(selected_blocks)):
        removed_before = sum(end_ids[j] - start_ids[j] for j in range(i))
        start_adj = start_ids[i] - removed_before
        end_adj = end_ids[i] - removed_before

        model = truncate_model(model, start_adj, end_adj)

        layer_idx = start_adj - 1
        transform = transforms[i]

        layer = getattr(container, attr)[layer_idx]
        original_weight = layer.mlp.down_proj.weight
        transformed_weight = (
            transform.t().to(torch.float64).to(original_weight.device)
            @ original_weight.to(torch.float64)
        ).to(save_torch_dtype)
        layer.mlp.down_proj.weight.data = transformed_weight

    # ---- Save ----
    if save_path is None:
        os.makedirs("output_models", exist_ok=True)
        layer_indices_for_name = "__".join(
            [f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = os.path.join(
            "output_models",
            f"{model_path}_{layers_to_skip}_layers_{layer_indices_for_name}_"
            f"{dataset}_{dataset_size}".replace("/", "_")
        )

    output_dir = f"{save_path}_ReplaceMe_joint_lstsq_{num_A}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Diagnostics always saved: block-search tooling ranks candidates with this.
    with open(os.path.join(output_dir, "lstsq_diagnostics.json"), "w") as f:
        json.dump({
            "selected_blocks": [list(b) for b in selected_blocks],
            "diagnostics": diagnostics,
            "n_calibration_tokens": n_tokens_used,
            "config": {
                "alpha_act": alpha_act,
                "alpha_grad": alpha_grad,
                "alpha_reg": alpha,
                "balance_terms": balance_terms,
                "identity_blend_threshold": identity_blend_threshold,
                "layers_to_skip": layers_to_skip,
                "num_A": num_A,
                "model_path": model_path,
            },
        }, f, indent=2)

    if save_transform_only:
        torch.save({
            "transforms": transforms,
            "selected_blocks": selected_blocks,
            "config": {
                "alpha_act": alpha_act,
                "alpha_grad": alpha_grad,
                "alpha_reg": alpha,
                "balance_terms": balance_terms,
                "identity_blend_threshold": identity_blend_threshold,
            },
        }, f"{output_dir}_transform")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run joint activation-gradient transformation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run joint activation-gradient LSTSQ for transform estimation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    lstsq(**config)
