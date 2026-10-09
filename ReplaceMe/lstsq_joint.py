
import argparse
import gc
import logging
import os
from typing import Optional, Tuple, List
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks,
                    truncate_model, seed_all)

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


def _normalize_tensor(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if t is None:
        return None
    norm = torch.norm(t, p='fro')
    return t / (norm + eps) if norm > eps else t


def solve_sylvester_standard(A, B, C, validate=False):

    dtype = torch.float64
    device = C.device

    with torch.no_grad():
        A, B, C = [x.to(dtype) for x in (A, B, C)]

        Lambda, U = torch.linalg.eigh(A)
        Sigma, V = torch.linalg.eigh(B)

        R_rhs = torch.mm(U.t(), C)
        R_rhs = torch.mm(R_rhs, V)

        Lambda = Lambda.unsqueeze(1)  # (m, 1)
        Sigma = Sigma.unsqueeze(0)  # (1, n)
        denom = Lambda + Sigma

        R = R_rhs / (denom + 1e-12)

        W = torch.mm(U, R)
        W = torch.mm(W, V.t())

        if validate:
            residual = torch.mm(A, W) + torch.mm(W, B)
            abs_norm = torch.norm(residual - C, p='fro')
            rel_norm = abs_norm / (torch.norm(C, p='fro') + 1e-10)
            print(f"[Validation] ||AW + WB - C||_F / ||C||_F: {rel_norm:.4e}")

        return W


def blend_with_identity(W, threshold=0.15):
    """
    Blend transform W with identity to prevent overfitting.
    W_blended = (1 - alpha) * W + alpha * I
    """
    if W is None:
        return None

    m = W.shape[0]
    I = torch.eye(m, dtype=W.dtype, device=W.device)

    W_minus_I = W - I
    norm_diff = torch.norm(W_minus_I, p='fro')
    norm_W = torch.norm(W, p='fro')

    alpha_blend = min(threshold, (norm_diff / (norm_W + 1e-10)).item())
    W_blended = (1 - alpha_blend) * W + alpha_blend * I

    return W_blended


def _compute_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for causal language modeling."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )


def lstsq(
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
    save_transform_only: bool = False,
    diag: bool = False,
    alpha: float = 0.0,         
    alpha_act: float = 1.0,  
    alpha_grad: float = 0.5,    
    normalize_inputs: bool = True,  
    identity_blend_threshold: float = 0.15,  
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    validate_sylvester: bool = True,
    selected_blocks: Optional[List[Tuple[int, int]]] = None,
) -> str:
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16),
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
    num_hidden_layers = model.config.num_hidden_layers

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

    mlp_activations = {}
    mlp_gradients = {}
    block_gradients = {}

    def save_mlp_activation(name: str):
        def hook(module, input, output):
            mlp_activations[name] = output.detach().clone()
            if output.requires_grad:
                def grad_hook(grad):
                    mlp_gradients[name] = grad.detach().clone()
                output.register_hook(grad_hook)
        return hook

    def save_block_gradient(name: str):
        def hook(module, input, output):
            if output.requires_grad:
                def grad_hook(grad):
                    block_gradients[name] = grad.detach().clone()
                output.register_hook(grad_hook)
        return hook

    hooks = []
    model_type = 'falcon' if 'falcon' in model_path.lower() else 'default'

    if model_type == 'falcon':
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
            hooks.append(layer.register_forward_hook(save_block_gradient(f'layer_{i}_block')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
            hooks.append(layer.register_forward_hook(save_block_gradient(f'layer_{i}_block')))

    if selected_blocks is not None:
        selected_blocks = [tuple(b) for b in selected_blocks]
        logging.info(f"selected_blocks given explicitly: {selected_blocks}")
    else:
        average_distances = torch.load(distances_path, weights_only=False)
        selected_blocks = select_non_overlapping_blocks(
            average_distances,
            layers_to_skip,
            num_blocks=num_A,
            merge_consecutive=merge_consecutive,
        )

    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    n_blocks = len(selected_blocks)

    device_for_accum = 'cpu'
    a1t_a1 = [torch.zeros(hidden_size, hidden_size, device=device_for_accum, dtype=torch.float64) for _ in range(n_blocks)]
    g2tg2 = [torch.zeros(hidden_size, hidden_size, device=device_for_accum, dtype=torch.float64) for _ in range(n_blocks)]
    a1t_a2 = [torch.zeros(hidden_size, hidden_size, device=device_for_accum, dtype=torch.float64) for _ in range(n_blocks)]
    g1tg2 = [torch.zeros(hidden_size, hidden_size, device=device_for_accum, dtype=torch.float64) for _ in range(n_blocks)]

    total_samples = 0
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
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        labels = labels.to(model.device)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            outputs = model(**inputs)
            loss = _compute_causal_lm_loss(outputs.logits, labels)
            loss.backward()

        hidden_states = outputs.hidden_states[1:]  

        batch_samples = input_ids.shape[0] * input_ids.shape[1]
        total_samples += batch_samples

        for idx in range(n_blocks):
            start_layer = start_ids[idx] - 1  
            end_layer = end_ids[idx] - 1   

            a1_key = f'layer_{start_layer}_mlp'
            if a1_key not in mlp_activations:
                logging.warning(f"Missing activations for block {idx}, skipping")
                continue

            mlp_start = mlp_activations[a1_key].view(-1, hidden_size).to(torch.float64)
            hidden_state_after_end = hidden_states[end_layer].view(-1, hidden_size).to(torch.float64)
            hidden_state_after_start = hidden_states[start_layer].view(-1, hidden_size).to(torch.float64)

            a1_raw = mlp_start
            a2_raw = hidden_state_after_end + mlp_start - hidden_state_after_start

            g1_raw = mlp_gradients.get(a1_key)

            g2_key = f'layer_{end_layer}_block'
            g2_raw = block_gradients.get(g2_key)

            if g1_raw is None or (alpha_grad > 0 and g2_raw is None):
                if alpha_grad > 0:
                    logging.warning(f"Missing gradients for block {idx}, skipping")
                    continue

            if g1_raw is not None:
                g1_raw = g1_raw.view(-1, hidden_size).to(torch.float64)
            if g2_raw is not None:
                g2_raw = g2_raw.view(-1, hidden_size).to(torch.float64)

            dev = a1t_a1[idx].device
            a1_dev = a1_raw.to(dev)
            a2_dev = a2_raw.to(dev)

            a1t_a1[idx].add_(alpha_act * (a1_dev.t() @ a1_dev))
            a1t_a2[idx].add_(alpha_act * (a1_dev.t() @ a2_dev))

            if alpha_grad > 0 and g1_raw is not None and g2_raw is not None:
                g1_dev = g1_raw.to(dev)
                g2_dev = g2_raw.to(dev)
                g2tg2[idx].add_(alpha_grad * (g2_dev.t() @ g2_dev))
                g1tg2[idx].add_(alpha_grad * (g1_dev.t() @ g2_dev))

        mlp_activations.clear()
        mlp_gradients.clear()
        block_gradients.clear()

        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    transforms = []
    for idx in range(n_blocks):
        if torch.all(a1t_a1[idx] == 0):
            logging.warning(f"No valid data for block {idx}, using identity transform")
            transforms.append(torch.eye(hidden_size, dtype=torch.bfloat16))
            continue

        reg = alpha * torch.eye(hidden_size, device=device_for_accum, dtype=torch.float64)
        A_mat = a1t_a1[idx] + reg
        B_mat = g2tg2[idx] + (alpha_grad * reg)
        C_mat = a1t_a2[idx] + g1tg2[idx]

        if diag:
            A_diag = torch.diag(A_mat)
            B_diag = torch.diag(B_mat)
            C_diag = torch.diag(C_mat)
            denom = A_diag + B_diag
            W_diag = C_diag / (denom + 1e-8)
            transform = torch.diag(W_diag).to(torch.bfloat16)
        else:
            try:
                transform_float64 = solve_sylvester_standard(
                    A_mat, B_mat, C_mat, validate=validate_sylvester
                )
                transform = transform_float64.to(torch.bfloat16)
            except Exception as e:
                logging.warning(f"Sylvester solver failed for block {idx}: {e}. Using diagonal fallback.")
                A_diag = torch.diag(A_mat)
                B_diag = torch.diag(B_mat)
                C_diag = torch.diag(C_mat)
                denom = A_diag + B_diag
                W_diag = C_diag / (denom + 1e-8)
                transform = torch.diag(W_diag).to(torch.bfloat16)

        if identity_blend_threshold > 0:
            transform = blend_with_identity(transform, threshold=identity_blend_threshold)
            logging.info(f"Block {idx}: Applied identity blending (threshold={identity_blend_threshold})")

        transforms.append(transform)

    for hook in hooks:
        hook.remove()
    del model, outputs
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        dtype=torch.bfloat16,
    )

    for i in range(len(selected_blocks)):
        start_adj = start_ids[i] - sum(end_ids[j] - start_ids[j] for j in range(i))
        end_adj = end_ids[i] - sum(end_ids[j] - start_ids[j] for j in range(i))

        model = truncate_model(model, start_adj, end_adj)

        layer_idx = start_adj - 1
        transform = transforms[i]

        if model_type == 'falcon':
            original_weight = model.transformer.h[layer_idx].mlp.down_proj.weight
            transformed_weight = (transform.t().to(torch.float64).to(original_weight.device) @ original_weight.to(torch.float64)).to(torch.bfloat16)
            model.transformer.h[layer_idx].mlp.down_proj.weight.data = transformed_weight
        else:
            original_weight = model.model.layers[layer_idx].mlp.down_proj.weight
            transformed_weight = (transform.t().to(torch.float64).to(original_weight.device) @ original_weight.to(torch.float64)).to(torch.bfloat16)
            model.model.layers[layer_idx].mlp.down_proj.weight.data = transformed_weight

    if save_path is None:
        os.makedirs('output_models', exist_ok=True)
        layer_indices_for_name = '__'.join([f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = os.path.join(
            "output_models",
            f"{model_path}_{layers_to_skip}_layers_{layer_indices_for_name}_{dataset}_{dataset_size}".replace("/", "_")
        )

    output_dir = f"{save_path}_ReplaceMe_joint_lstsq_{num_A}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if save_transform_only:
        torch.save({
            'transforms': transforms,
            'selected_blocks': selected_blocks,
            'config': {
                'alpha_act': alpha_act,
                'alpha_grad': alpha_grad,
                'normalize_inputs': normalize_inputs,
                'alpha_reg': alpha,
                'identity_blend_threshold': identity_blend_threshold,
            }
        }, f"{output_dir}_transform")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
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