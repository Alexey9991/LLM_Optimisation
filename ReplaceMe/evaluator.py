"""Model evaluation module for assessing transformer model performance.

This module provides functionality to evaluate transformer models on various tasks,
either using default evaluation metrics or task-specific configurations.
"""

import argparse
import logging
from typing import Dict, List, Union
import yaml
from colorama import Fore, init

from .utils import eval_model, eval_model_specific, seed_all

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


def evaluator(
    model_path: str,
    tasks: Union[str, List[str], Dict[str, dict]] = "default",
    **kwargs
) -> dict:
    """Evaluate a transformer model on specified tasks.
    
    Args:
        model_path: Path to pretrained model or model identifier
        tasks: Evaluation tasks configuration. Can be:
               - "default" for default evaluation
               - List of task names
               - Dictionary of task-specific configurations
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary containing evaluation results
    """
    if tasks == "default":
        logging.info(f"{Fore.GREEN}Running default evaluation on {model_path}{Fore.RESET}")
        results = eval_model(model_path, **kwargs)
    else:
        logging.info(f"{Fore.GREEN}Running task-specific evaluation on {model_path}{Fore.RESET}")
        results = eval_model_specific(model_path, tasks, **kwargs)
    
    logging.info(f"{Fore.GREEN}Evaluation completed{Fore.RESET}")
    return results


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"{Fore.RED}Config file not found: {config_path}{Fore.RESET}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"{Fore.RED}Invalid YAML in config file: {config_path}{Fore.RESET}")
        raise


def run_from_config() -> None:
    """Run model evaluation from configuration file.
    
    Reads evaluation parameters from a YAML config file and executes the evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Run model evaluation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    
    try:
        config = read_config(args.config)
        evaluator(**config)
    except Exception as e:
        logging.error(f"{Fore.RED}Evaluation failed: {str(e)}{Fore.RESET}")
        raise