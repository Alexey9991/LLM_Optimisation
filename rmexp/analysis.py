"""CPU-only analysis: how the block signals agree with each other, what each signal would have chosen,
and the per-token gradient diagnostic that showed grad_contrib to be an artefact of the answer-only loss."""
from typing import Dict, List

import numpy as np
import pandas as pd

from .profile import RANK_COLS, SIGNALS, greedy_by_signal, rank_table


def signal_correlations(p: pd.DataFrame) -> pd.DataFrame:
    q = p.copy()
    q["one_minus_coher"] = 1.0 - q["grad_coher"]
    cols = [c for c in SIGNALS if c != "grad_coher" and c in q] + ["one_minus_coher"]
    return q[cols].corr(method="spearman").round(3)


def top_k(p: pd.DataFrame, k: int = 3) -> Dict[str, list]:
    out = {}
    for sc in ("score_cos", "score_contrib", "score_coher"):
        if sc in p:
            order = p.sort_values([sc, "dist_act"])
            out[sc] = [(int(s), int(e)) for s, e in zip(order["block_start"].head(k), order["block_end"].head(k))]
    return out


def series_tables(cfg, exps_blocks: Dict[str, List[tuple]], profiles: Dict[int, pd.DataFrame],
                  em: Dict[str, float] = None):
    """Rank tables per window length with the series' blocks marked, the rank of every used block under every
    signal, and the greedy choice of every signal. exps_blocks: experiment name -> blocks."""
    em = em or {}
    tables = {k: rank_table(p) for k, p in profiles.items()}
    for name, blocks in exps_blocks.items():
        lengths = sorted({e - s for s, e in blocks}, reverse=True)
        print(f"\n{'=' * 100}\n{name} | blocks {blocks}" + (f" | restored EM {em[name]}" if name in em else ""))
        for k in lengths:
            t = tables[k].copy()
            t["chosen"] = ["<--" if (s, e) in blocks else "" for s, e in zip(t.block_start, t.block_end)]
            print(f"\nwindows of {k} layers ({len(t)} candidates), ranks 0 = best, in layer order:")
            print(t.to_string(index=False))

    rows = []
    for name, blocks in exps_blocks.items():
        for s, e in blocks:
            t = tables[e - s]
            r = t[(t.block_start == s) & (t.block_end == e)].iloc[0]
            rows.append({"experiment": name, "block": (s, e), "len": e - s, "of": len(t),
                         **{c: int(r[c]) for c in RANK_COLS}, "EM restored": em.get(name)})
    print(f"\n{'=' * 100}\nblocks used, rank under each signal:")
    print(pd.DataFrame(rows).to_string(index=False))

    rows = []
    for name, blocks in exps_blocks.items():
        lengths = [e - s for s, e in blocks]
        row = {"experiment": name}
        for col in RANK_COLS:
            b = greedy_by_signal(lengths, tables, col, cfg.gap)
            row[col.removeprefix("r_")] = " ".join(f"({s},{e})" for s, e in b) if b else "—"
        row["earliest deleted (used)"] = min(s for s, _ in blocks)
        rows.append(row)
    print(f"\n{'=' * 100}\ngreedy choice by each signal (0-based deleted layers = s .. e-1):")
    print(pd.DataFrame(rows).set_index("experiment").T.to_string())
    return tables


def grad_contrib_diagnostic(cfg, rm, windows: List[int], n_texts: int = 16, skip: int = 8) -> pd.DataFrame:
    """Per-token grad_contrib split into question / answer tokens for a few windows (candidate i = block
    (i+1, i+1+skip)). With the answer-only loss the question tokens get gradient only through attention
    from above, so near the top ||g_out|| -> 0 and the per-token ratio explodes. Fresh kernel, ~5 min."""
    import gc
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from .config import COMPUTE_DTYPE, vram
    from .data import calibration_batches
    from .replaceme_repo import Loader

    gc.collect(); torch.cuda.empty_cache()
    assert vram("before load") < 0.5, "GPU0 is not empty: restart the kernel first"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    model = Loader({"": 0}).from_pretrained(cfg.model_path, quantization_config=quant, output_hidden_states=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    tok.pad_token = tok.pad_token or tok.eos_token
    marker = tok("Answer:", add_special_tokens=False)["input_ids"]
    batches = calibration_batches("openai/gsm8k", "train", "text", n_texts, 1, tok)

    acc = {i: {g: {"ratio": [], "gout": [], "num2": 0.0, "den2": 0.0} for g in ("question", "answer")}
           for i in windows}
    for batch in batches:
        enc = tok(batch, return_tensors="pt", max_length=cfg.calib_max_length, truncation=True).to(0)
        ids = enc["input_ids"][0].tolist()
        pos = next((j + len(marker) for j in range(len(ids) - len(marker), -1, -1)
                    if ids[j:j + len(marker)] == marker), 0)
        with torch.enable_grad():
            out = model(**enc, use_cache=False)
            hs = out.hidden_states[1:]
            for h in hs:
                if h.requires_grad:
                    h.retain_grad()
            loss = rm.ds._lm_loss(out.logits, enc["input_ids"], enc["attention_mask"], [pos])
            loss.backward()
        L = len(ids)
        groups = {"question": torch.arange(L - 1) < pos, "answer": torch.arange(L - 1) >= pos}
        for i in windows:
            gi = hs[i].grad[0, :-1].float()
            go = hs[i + skip].grad[0, :-1].float()
            d = (gi - go).norm(dim=-1)
            n_out = go.norm(dim=-1)
            for g, m in groups.items():
                m = m.to(d.device)
                if m.any():
                    acc[i][g]["ratio"] += (d[m] / (n_out[m] + 1e-8)).tolist()
                    acc[i][g]["gout"] += n_out[m].tolist()
                    acc[i][g]["num2"] += float((d[m] ** 2).sum())
                    acc[i][g]["den2"] += float((n_out[m] ** 2).sum())
        model.zero_grad(set_to_none=True)
        del out, hs, loss

    rows = []
    for i in windows:
        for g in ("question", "answer"):
            a = acc[i][g]
            r, go = np.array(a["ratio"]), np.array(a["gout"])
            rows.append({"block": (i + 1, i + 1 + skip), "tokens": g, "n": len(r),
                         "median ||g_out||": np.median(go), "mean ratio": r.mean(), "median ratio": np.median(r),
                         "fro ratio": np.sqrt(a["num2"]) / (np.sqrt(a["den2"]) + 1e-8)})
    del model
    gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows)
