"""ReplaceMe experiments on Llama-3-8B: prune a block of layers, heal with LoRA, restore the layers, evaluate on GSM8K.

The heavy lifting (T estimation, block profiling) is done by the ReplaceMe repo itself, imported from
`cfg.replaceme_repo`. This package only wires it together and keeps the evaluation / training protocol fixed.
"""
from .config import Cfg, ExperimentSpec, load_config, COMPUTE_DTYPE, vram, free_disk_gb
from .pipeline import Experiment, resolve_blocks

__all__ = ["Cfg", "ExperimentSpec", "load_config", "COMPUTE_DTYPE", "vram", "free_disk_gb",
           "Experiment", "resolve_blocks"]
