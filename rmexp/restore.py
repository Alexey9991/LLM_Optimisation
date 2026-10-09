"""Put the deleted layers back. Kept layers come from pruned + heal (LoRA merged), deleted layers from the base
model. The carrier layers (the ones holding T in down_proj) either keep T^T W + LoRA delta, as in Misha's
ras/restore.py, or get W_orig + LoRA delta (T removed)."""
import copy
import gc
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_decoder_layers(model) -> nn.ModuleList:
    if hasattr(model, "transformer"):
        return model.transformer.h
    return model.model.layers


def set_decoder_layers(model, layers: nn.ModuleList) -> None:
    if hasattr(model, "transformer"):
        model.transformer.h = layers
    else:
        model.model.layers = layers


def restore_blocks(base_model_path: str, pruned_dir, adapter_dir, output_dir, blocks: List[Tuple[int, int]],
                   n_layers: int, remove_T: bool = False) -> Path:
    """blocks are 1-based (start, end): 0-based layers start..end-1 were deleted, layer start-1 carries T.
    Everything on CPU in bf16 (two full models in RAM, ~32 GB for 8B)."""
    output_dir = Path(output_dir)
    deleted = sorted({l for s, e in blocks for l in range(s, e)})
    carriers = {s - 1 for s, _ in blocks}
    assert not carriers & set(deleted), (blocks, "a carrier layer is inside a deleted block")
    kept = [o for o in range(n_layers) if o not in deleted]          # kept[p] = original index of pruned layer p

    pruned = AutoModelForCausalLM.from_pretrained(str(pruned_dir), dtype=torch.bfloat16, device_map="cpu")
    pruned_layers = get_decoder_layers(pruned)
    w_t = {o: pruned_layers[p].mlp.down_proj.weight.detach().clone()      # T^T W before heal
           for p, o in enumerate(kept) if o in carriers}
    healed = PeftModel.from_pretrained(pruned, str(adapter_dir)).merge_and_unload()

    base = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.bfloat16, device_map="cpu")
    healed_layers, base_layers = get_decoder_layers(healed), get_decoder_layers(base)
    assert len(healed_layers) == len(kept), (len(healed_layers), len(kept))

    layers = []
    for o in range(n_layers):
        if o in deleted:
            layers.append(copy.deepcopy(base_layers[o]))
            continue
        layer = healed_layers[kept.index(o)]
        if o in carriers:
            w = layer.mlp.down_proj.weight
            w_orig = base_layers[o].mlp.down_proj.weight
            delta = w.data - w_t[o]                                    # ≈ B·A of the LoRA on the carrier
            print(f"restore: carrier {o}: |T^T W - W|/|W| = {((w_t[o] - w_orig).float().norm() / w_orig.float().norm()):.3f}, "
                  f"|LoRA|/|W| = {(delta.float().norm() / w_orig.float().norm()):.4f}"
                  + (" -> T removed" if remove_T else " -> T kept"))
            if remove_T:
                w.data = (w_orig + delta).to(w.dtype)
        layers.append(layer)

    set_decoder_layers(healed, nn.ModuleList(layers))
    healed.config.num_hidden_layers = n_layers
    if getattr(base.config, "layer_types", None) is not None:
        healed.config.layer_types = list(base.config.layer_types)
    if len(layers) != base.config.num_hidden_layers:
        raise RuntimeError(f"restored {len(layers)} layers, base has {base.config.num_hidden_layers}")

    healed.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))

    del healed, base, pruned
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
