"""Imports ReplaceMe.distance_scored / lstsq_joint, applies the two in-place fixes to distance_scored.py
(hidden_states[1:], extra block signals), binds the calibration texts and the GPU placement."""
import hashlib
import importlib
import inspect
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM

from .data import calibration_batches

MARK_3SCORES = "# --- 3scores patch ---"


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


class Loader:
    """AutoModelForCausalLM stand-in: fixed device_map, frozen weights, activations still get gradients."""

    def __init__(self, device_map):
        self.device_map = device_map

    def from_pretrained(self, path, *args, **kw):
        if kw.get("device_map") == "cpu" or not torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(path, *args, **kw)
        kw["device_map"] = self.device_map
        model = AutoModelForCausalLM.from_pretrained(path, *args, **kw)
        model.requires_grad_(False)
        model.enable_input_require_grads()
        dmap = getattr(model, "hf_device_map", None) or {"": str(next(model.parameters()).device)}
        print("device map:", dict(Counter(str(d) for d in dmap.values())),
              "| vram GiB:", round(torch.cuda.memory_allocated() / 2**30, 2))
        return model


def _fix_hidden_states_index(path: Path) -> bool:
    text = path.read_text()
    if any(k in text for k in ("hidden_states[1:]", "hs_full[1:]", "enter, leave")):
        return False
    shutil.copy(path, path.with_name("distance_scored.py.bak_nofix"))
    text, n = re.subn(r"(\n[ \t]*hs = out\.hidden_states)[ \t]*\n",
                      r"\1[1:]   # hs[l] = output of layer l: candidate i = block (i+1, i+1+skip)\n", text, count=1)
    assert n == 1, "line 'hs = out.hidden_states' not found in distance_scored.py"
    path.write_text(text)
    return True


_EDITS_3SCORES = [
    ("    acc_tay = [[] for _ in range(n_cand)]\n",
     "    acc_tay = [[] for _ in range(n_cand)]\n"
     f"    {MARK_3SCORES}\n"
     "    acc_contrib = [[] for _ in range(n_cand)]\n"
     "    acc_contrib_fro = [[] for _ in range(n_cand)]\n"
     "    acc_coher = [[] for _ in range(n_cand)]\n"
     "    _eps = 1e-8\n"),
    ("                acc_grad[i].append(float(_angular(g_in, g_out).mean()))\n",
     "                acc_grad[i].append(float(_angular(g_in, g_out).mean()))\n"
     "                # grad_contrib: relative change of the residual-stream gradient across the window\n"
     "                d_tok = (g_in - g_out).norm(dim=-1)\n"
     "                acc_contrib[i].append(float((d_tok / (g_out.norm(dim=-1) + _eps)).mean()))\n"
     "                acc_contrib_fro[i].append(float((g_in - g_out).norm() / (g_out.norm() + _eps)))\n"
     "                # grad_coher: ||sum_j g_j|| / sum_j ||g_j|| over the residual points of the window\n"
     "                gs = [hs[j].grad for j in range(i, i + layers_to_skip + 1)]\n"
     "                if b_idx == 0:\n"
     "                    assert all(g is not None for g in gs), f'candidate {i}: a gradient inside the window is None'\n"
     "                if all(g is not None for g in gs):\n"
     "                    gs = [g.detach().reshape(-1, D)[am].float() for g in gs]\n"
     "                    num = torch.stack(gs, 0).sum(0).norm(dim=-1)\n"
     "                    den = sum(g.norm(dim=-1) for g in gs) + _eps\n"
     "                    acc_coher[i].append(float((num / den).mean()))\n"
     "                    del gs, num, den\n"),
    ("    taylor = [float(np.mean(v)) if v else float(\"nan\") for v in acc_tay]\n",
     "    taylor = [float(np.mean(v)) if v else float(\"nan\") for v in acc_tay]\n"
     "    grad_contrib = [float(np.mean(v)) if v else float(\"nan\") for v in acc_contrib]\n"
     "    grad_contrib_fro = [float(np.mean(v)) if v else float(\"nan\") for v in acc_contrib_fro]\n"
     "    grad_coher = [float(np.mean(v)) if v else float(\"nan\") for v in acc_coher]\n"
     "    r_contrib = np.argsort(np.argsort(grad_contrib)).astype(float)\n"
     "    r_coher = np.argsort(np.argsort([1.0 - c for c in grad_coher])).astype(float)\n"),
    ("    score = (r_act + r_grad) / 2.0\n",
     "    score = (r_act + r_grad) / 2.0\n"
     "    score_contrib = r_contrib.copy()\n"
     "    score_coher = r_coher.copy()\n"),
    ("        for i in vetoed:\n            score[i] = float(\"inf\")\n",
     "        for i in vetoed:\n            score[i] = float(\"inf\")\n"
     "            score_contrib[i] = float(\"inf\")\n"
     "            score_coher[i] = float(\"inf\")\n"),
    ("            \"rank_act\", \"rank_grad\", \"score\", \"vetoed\", \"selected\"])\n",
     "            \"rank_act\", \"rank_grad\", \"score\", \"vetoed\", \"selected\",\n"
     "            \"grad_contrib\", \"grad_contrib_fro\", \"grad_coher\",\n"
     "            \"rank_contrib\", \"rank_coher\", \"score_cos\", \"score_contrib\", \"score_coher\"])\n"),
    ("                \"selected\": i == best})\n",
     "                \"selected\": i == best,\n"
     "                \"grad_contrib\": grad_contrib[i], \"grad_contrib_fro\": grad_contrib_fro[i],\n"
     "                \"grad_coher\": grad_coher[i],\n"
     "                \"rank_contrib\": int(r_contrib[i]), \"rank_coher\": int(r_coher[i]),\n"
     "                \"score_cos\": score[i], \"score_contrib\": score_contrib[i],\n"
     "                \"score_coher\": score_coher[i]})\n"),
    ("    logging.info(f\"{Fore.GREEN}SELECTED: {selected_block}{Fore.RESET} \"\n",
     "    for name, sc in ((\"score_contrib\", score_contrib), (\"score_coher\", score_coher)):\n"
     "        logging.info(f\"top-3 by {name}:\")\n"
     "        for rank, i in enumerate(np.argsort(sc)[:3], 1):\n"
     "            logging.info(f\"  #{rank}: block ({i + 1},{i + 1 + layers_to_skip})  \"\n"
     "                         f\"contrib={grad_contrib[i]:.4f} fro={grad_contrib_fro[i]:.4f} \"\n"
     "                         f\"coher={grad_coher[i]:.4f}  {name}={sc[i]:.1f}\")\n"
     "    logging.info(f\"{Fore.GREEN}SELECTED: {selected_block}{Fore.RESET} \"\n"),
    ("            \"score\": score.tolist(), \"selected_block\": selected_block}\n",
     "            \"score\": score.tolist(), \"selected_block\": selected_block,\n"
     "            \"grad_contrib\": grad_contrib, \"grad_contrib_fro\": grad_contrib_fro,\n"
     "            \"grad_coher\": grad_coher, \"score_cos\": score.tolist(),\n"
     "            \"score_contrib\": score_contrib.tolist(), \"score_coher\": score_coher.tolist()}\n"),
]


def _patch_3scores(path: Path) -> bool:
    src = path.read_text()
    if MARK_3SCORES in src:
        return False
    bak = path.with_name(f"distance_scored.py.bak_3scores_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(path, bak)
    for old, new in _EDITS_3SCORES:
        n = src.count(old)
        assert n == 1, f"3scores patch: anchor found {n} times, file left untouched (backup {bak}):\n{old}"
        src = src.replace(old, new)
    path.write_text(src)
    print("distance_scored.py: 3scores patch applied, backup ->", bak)
    return True


def load_replaceme(cfg, patch_3scores: bool = True, lstsq_device_map=None) -> SimpleNamespace:
    """Returns a namespace with ds (distance_scored), lj (lstsq_joint), utils, signatures and md5s."""
    repo = Path(cfg.replaceme_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import ReplaceMe.distance_scored as ds
    import ReplaceMe.lstsq_joint as lj
    import ReplaceMe.utils as rm_utils

    ds_file = Path(ds.__file__)
    if _fix_hidden_states_index(ds_file):
        print("distance_scored.py: hidden_states[1:] fix applied, backup -> distance_scored.py.bak_nofix")
    if patch_3scores:
        _patch_3scores(ds_file)
    ds = importlib.reload(ds)
    lj = importlib.reload(lj)

    for m in (rm_utils, ds, lj):
        print(f"{m.__name__:28s} {m.__file__}  md5 {md5(m.__file__)}")
    ds_sig = inspect.signature(ds.profile_distances).parameters
    lj_sig = inspect.signature(lj.lstsq).parameters
    assert {"answer_only_loss", "csv_save_path", "distances_save_path", "compute_taylor"} <= set(ds_sig), list(ds_sig)
    assert {"selected_blocks", "alpha_grad", "alpha_act", "identity_blend_threshold", "num_A"} <= set(lj_sig), list(lj_sig)
    body = inspect.getsource(ds.profile_distances)
    assert any(k in body for k in ("hidden_states[1:]", "hs_full[1:]", "enter, leave"))
    if patch_3scores:
        assert "grad_coher" in body

    ds.get_calib_dataloader = calibration_batches
    lj.get_calib_dataloader = calibration_batches
    # one card for the profile: with "auto" retain_grad on the hidden states gives None
    ds.AutoModelForCausalLM = Loader({"": 0})
    if lstsq_device_map is None:
        lstsq_device_map = {"model": 0, "lm_head": 1} if torch.cuda.device_count() > 1 else {"": 0}
    lj.AutoModelForCausalLM = Loader(lstsq_device_map)
    return SimpleNamespace(ds=ds, lj=lj, utils=rm_utils, ds_sig=ds_sig, lj_sig=lj_sig,
                           ds_md5=md5(ds.__file__), lj_md5=md5(lj.__file__))
