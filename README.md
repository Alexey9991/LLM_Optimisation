# replaceme-llama

Depth pruning of Llama-3-8B with ReplaceMe: 8 consecutive decoder layers are removed and replaced by one
linear map T folded into `down_proj` of the layer before the block, the pruned model is healed with LoRA,
the layers are put back, and the result is compared with LoRA fine-tuning of the full model. Metric: exact
match on GSM8K test (1319 questions), everything in nf4 on a single RTX 2080 Ti.

```
ReplaceMe/              the ReplaceMe package (block profiling, T estimation, pruning)
  distance_scored.py    block profile: dist_act, dist_grad (+ grad_contrib, grad_contrib_fro, grad_coher), combined rank score
  lstsq_joint.py        joint activation-gradient least squares for T, folds T into the carrier layer
  utils.py              calibration loader, truncate_model, block selection helpers
  ...                   other ReplaceMe variants kept for reference (lstsq.py, cosine_dist.py, distance*.py)
reproduce/              ReplaceMe YAML configs (upstream defaults; the notebook passes its own arguments)
notebooks/Llama8B.ipynb the pipeline: config → baseline → SFT control → profile → prune → heal → restore → McNemar
```

## Setup

Python 3.10+, CUDA. On the lab hub the kernel is `run_replacement` (torch 2.10, transformers 5.x, peft,
bitsandbytes, datasets). Elsewhere: `pip install -r requirements.txt`.

The notebook adds the repo root to `sys.path` and imports `ReplaceMe.distance_scored` / `ReplaceMe.lstsq_joint`
from it; nothing needs installing. Two environment variables, both optional:

* `REPLACEME_REPO` — repo root, if the notebook is not run from `notebooks/`;
* `REPLACEME_RUNS` — output folder for models, adapters and results (default `/home/tnn/LLM_optimisatoin/runs_reproduce`).

## Protocol (fixed in cell 1 of the notebook)

* calibration: 128 GSM8K train items as `Question: ...\nAnswer: ...`, max 256 tokens, the same texts for the profile and for T;
* T: `alpha_act 1.0`, `alpha_grad 0.5`, identity blend 0.15; the carrier's `||T^T W|| / ||W||` is printed, above ~3.1 the block did not recover;
* LoRA r8, alpha 16, MLP projections only, batch 2×4, lr 2e-4, one epoch, paged AdamW 8-bit, fp16 — the same for the SFT control and for heal;
* evaluation: HF generate, greedy, 256 new tokens, batch 8, 4-bit base + adapter without merge.

Blocks are 1-based `(start, end)`: 0-based layers `start..end-1` are deleted, layer `start-1` carries T.
EM on 1319 questions has SE ≈ 1.4 pt; models are compared with McNemar on per-question predictions.

## Running

Open `notebooks/Llama8B.ipynb`. Memory on an 11 GiB card is tight, so one heavy stage per kernel:

| Cell | Stage | Kernel |
|---|---|---|
| 1, A, B, C, D | config and code | after every restart |
| 2 | eval base | restart after |
| 3, 4 | train SFT, eval SFT | restart after each |
| 4b, 4c | profile (`distance_scored`), top-1 block | restart after 4b |
| 5 | prune (`lstsq_joint`) | restart after |
| 5b | eval pruned, T only | restart after |
| 6, 7 | heal, eval pruned + heal | restart after each |
| 8 | restore (T kept) + eval | restart after |
| 10 | summary, McNemar | — |

Every stage skips outputs that already exist under `REPLACEME_RUNS`.

## Results (nf4, HF generate, block (22,30))

| model | EM % |
|---|---|
| baseline | 10.77 |
| fine-tuned, LoRA on the full model | 59.59 |
| pruned, T only | 7.43 |
| pruned + heal, 24 layers | 50.11 |
| restored with T | 49.73 |
| restored without T | 50.95 |

Fine-tuned minus restored: +9.86 pt, 95 % CI [+7.06; +12.65], p < 1e-4.

## Notes on `ReplaceMe/distance_scored.py`

Two changes against upstream, both in place: `hs = out.hidden_states[1:]` (upstream indexed from the
embeddings and labelled every block one layer too late) and three extra columns in the profile CSV
(`grad_contrib`, `grad_contrib_fro`, `grad_coher`, with their ranks and scores). The `score` column and the
selected block are unchanged.
