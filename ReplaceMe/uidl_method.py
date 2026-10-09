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

from .utils import (select_non_overlapping_blocks, truncate_model, 
                    seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()
def uidl(
    model_path: str,
    layers_to_skip: int,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    token: Optional[str] = None,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    
) -> str:
    """Truncate the model based on this work https://arxiv.org/abs/2403.17887.
    Args:
        model_path: Path to pretrained model
        layers_to_skip: Number of layers to skip
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        token: Authentication token
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of blocks
        merge_consecutive: Whether to merge consecutive blocks
    
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
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        output_hidden_states=True,
        token=token
    )
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if save_path is None:
        if not os.path.exists('output_models'):
            os.mkdir('output_models')
        save_path = "output_models/" + (
            f"{model_path}_{layers_to_skip}_layers_{start_id}_"
            f"{end_id}"
        ).replace("/", "_")
    
    model.save_pretrained(f"{save_path}_UIDL")
    tokenizer.save_pretrained(f"{save_path}_UIDL")
    
    # Final cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return f"{save_path}_UIDL"


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the cosine distance calculation from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run numerical solvers for linear transform estimation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    
    average_distances = torch.load(config['distances_path'])
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        config['layers_to_skip'],
        num_blocks=config['num_A'],
        merge_consecutive=config['merge_consecutive']
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    for i in range(len(selected_blocks)):
        path = uidl(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        config["model_path"] = path
