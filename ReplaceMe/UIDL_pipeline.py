import argparse
import inspect
import logging

import torch
import yaml
from colorama import Fore, Style, init

# Import local modules
from .distance import profile_distances
from .evaluator import evaluator
from .uidl_method import uidl
from .utils import seed_all, select_non_overlapping_blocks

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages with timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set random seeds for reproducibility
seed_all()

def UIDL_pipeline(config):
    # Extract the relevant parameters based on function signatures
    signature = inspect.signature(profile_distances)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    if config['distances_path'] is None:
        # Profile distances using filtered configuration
        profile_distances(**filtered_config)
        config['distances_path'] = "./distances.pth"  
    signature = inspect.signature(uidl)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    # Load average distances and select non-overlapping blocks
    average_distances = torch.load(filtered_config['distances_path'])  
    selected_blocks = select_non_overlapping_blocks(
        average_distances, 
        filtered_config['layers_to_skip'], 
        num_blocks=filtered_config['num_A'], 
        merge_consecutive=filtered_config['merge_consecutive']
    )
    # Calculate start and end IDs, and number of layers
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]    
    # Iterate over each selected block
    for i in range(len(selected_blocks)):
        path = uidl(**filtered_config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
        filtered_config["model_path"] = path
    
    # Evaluate using the updated configuration
    signature = inspect.signature(evaluator)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    filtered_config["model_path"] = path
    evaluator(**filtered_config)

def read_config(config_path: str):
    # Read and return the YAML configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_from_config():
    # Parse command-line arguments for configuration file path
    parser = argparse.ArgumentParser(
        description="Run UIDL depth pruning based on a configuration file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    # Execute pipeline based on parsed configuration
    args = parser.parse_args()
    config = read_config(args.config)
    UIDL_pipeline(config)
