"""Least Squares Transformation module for transformer model optimization.

This module computes linear transformations between transformer model layers using
least squares estimation to enable model compression and optimization.
"""

import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml
from colorama import Fore, init
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from .utils import (get_calib_dataloader, select_non_overlapping_blocks,
                    truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
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
    alpha: float = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
) -> str:
    """Compute least squares transformations between model layers.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use for calibration
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers between compared blocks
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations (unused)
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of the layer to start the cut
        token: Authentication token for private models
        save_transform_only: Whether to only save the transform
        diag: Whether to use diagonal matrix approximation
        alpha: Regularization strength
        distances_path: Path to precomputed distance metrics
        num_A: Number of LT transforms to estimate
        merge_consecutive: Whether to merge consecutive LTs
        
    Returns:
        Path where transformed model is saved
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model and tokenizer
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

    # Setup activation hooks
    def save_mlp_activation(name: str):
        """Returns a hook function that saves module outputs."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook

    hooks = []
    mlp_activations = {}
    model_type = 'falcon' if 'falcon' in model_path.lower() else 'default'
    
    if model_type == 'falcon':
        for i, layer in enumerate(model.transformer.h):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    else:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))

    # Load precomputed distances and select blocks
    #average_distances = torch.load(distances_path)
    #selected_blocks = select_non_overlapping_blocks(
    #    average_distances,
    #    layers_to_skip,
    #    num_blocks=num_A,
    #    merge_consecutive=merge_consecutive,
    #)
    selected_blocks = [(21, 29)]
    # Initialize accumulation matrices
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    a1t_a1 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]
    a1t_a2 = [
        torch.zeros(hidden_size, hidden_size, device='cuda').to(torch.float64)
        for _ in range(len(selected_blocks))
    ]

    # Process batches
    for batch in tqdm(
        dataloader,
        desc=f"{Fore.GREEN}Running LSTSQ{Fore.RESET}",
        dynamic_ncols=True,
        colour="green",
    ):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states[1:]
        
        # Get relevant hidden states
        hidden_states_mlp = [
            mlp_activations[f'layer_{i}_mlp'].view(-1, hidden_size).to(torch.float64)
            for i in range(model.config.num_hidden_layers) if i+1 in start_ids
        ]
        hidden_states_i = [
            hidden_states[i].view(-1, hidden_size).to(torch.float64)
            for i in range(model.config.num_hidden_layers) if i+1 in start_ids
        ]
        hidden_states_n = [
            hidden_states[i].view(-1, hidden_size).to(torch.float64)
            for i in range(model.config.num_hidden_layers) if i+1 in end_ids
        ]

        # Accumulate matrices
        for idx in range(len(selected_blocks)):
            dev = hidden_states_mlp[idx].device
            if idx == 0 or start_ids[idx] != end_ids[idx-1]:
                a1_batch = hidden_states_mlp[idx].to(dev)
                a2_batch = hidden_states_n[idx].to(dev) + hidden_states_mlp[idx].to(dev) - hidden_states_i[idx].to(dev)
            else:
                a1_batch = hidden_states_i[idx].to(dev)
                a2_batch = hidden_states_n[idx].to(dev)
                
            a1t_a1[idx] += a1_batch.t().to(a1t_a1[idx].device) @ a1_batch.to(a1t_a1[idx].device)
            a1t_a2[idx] += a1_batch.t().to(a1t_a2[idx].device) @ a2_batch.to(a1t_a2[idx].device)

    # Compute transformations
    transforms = []
    for idx in range(len(selected_blocks)):
        if diag:
            transform = torch.diag(
                torch.linalg.inv(
                    a1t_a1[idx] * torch.eye(hidden_size, device='cuda').to(torch.float64)
                ) @ torch.diag(a1t_a2[idx])
            )
        else:
            reg_term = alpha * torch.eye(hidden_size, device='cuda').to(torch.float64)
            transform = torch.linalg.inv(a1t_a1[idx] + reg_term) @ a1t_a2[idx]
        transforms.append(transform)

    # Clean up
    for hook in hooks:
        hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Load fresh model for transformation
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
    )

    # Apply transformations
    for i in range(len(selected_blocks)):
        model = truncate_model(model, start_ids[i] - num_layers[i], end_ids[i] - num_layers[i])
        model.model.layers[start_ids[i] - num_layers[i] - 1].mlp.down_proj.load_state_dict({
            "weight": (transforms[i].T.cpu() @ 
                      model.model.layers[start_ids[i] - num_layers[i] - 1].mlp.down_proj.weight.to(torch.float64)
                     ).to(torch.bfloat16)
        })

    # Save results
    if save_path is None:
        os.makedirs('output_models', exist_ok=True)
        layer_indices_for_name = '__'.join([f"{start_ids[i]}_{end_ids[i]}" for i in range(len(selected_blocks))])
        save_path = os.path.join(
            "output_models",
            f"{model_path}_{layers_to_skip}_layers_{layer_indices_for_name}_{dataset}_{dataset_size}".replace("/", "_")
        )

    model.save_pretrained(f"{save_path}_ReplaceMe_lstsq_{num_A}")
    tokenizer.save_pretrained(f"{save_path}_ReplaceMe_lstsq_{num_A}")

    if save_transform_only:
        torch.save(transforms, f"{save_path}_ReplaceMe_lstsq_{num_A}_transform")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return f"{save_path}_ReplaceMe_lstsq_{num_A}"


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config() -> None:
    """Run least squares transformation from configuration file."""
    parser = argparse.ArgumentParser(
        description="Run LSTSQ for linear transform estimation based on a configuration file."
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