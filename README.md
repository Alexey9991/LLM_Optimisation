# replaceme-llama

Depth pruning of Llama-3-8B with ReplaceMe: a block of decoder layers is removed and replaced by one linear
map T folded into `down_proj` of the layer before the block, the pruned model is healed with LoRA, the layers
are put back, and the result is compared with LoRA fine-tuning of the full model. Metric: exact match on
GSM8K test (1319 questions), everything in nf4 on one RTX 2080 Ti.

```
ReplaceMe/                 the ReplaceMe code that is actually used
  distance_scored.py       block profile: dist_act, dist_grad (+ grad_contrib, grad_contrib_fro, grad_coher), combined rank score
  lstsq_joint.py           joint activation-gradient least squares for T, folds T into the carrier layer
  utils.py                 calibration loader, truncate_model, block helpers
rmexp/                     the pipeline around it (the cells of the notebook as importable modules)
  data.py                  GSM8K prompts, training / calibration texts, exact-match metric
  evaluate.py              HF generate evaluation, results json with per-question predictions
  train.py                 LoRA r8 on the MLP, one epoch (SFT control and heal)
  restore.py               deleted layers back from the base model, with or without T in the carriers
  replaceme_repo.py        imports ReplaceMe, binds the calibration texts and GPU placement, keeps the two patches idempotent
  profile.py / prune.py    block profiles and choice rules; lstsq call, norm ratio of every carrier
  pipeline.py              Experiment: prune → heal → restore → evaluate, every stage skips what is on disk
  stats.py / analysis.py   McNemar tables; signal correlations, rank tables, gradient diagnostic
configs/
  llama3_8b_nf4.yaml       the protocol and the list of experiments (which blocks go)
  lstsq_joint_gsm8k.yaml   arguments for one standalone lstsq_joint run
scripts/
  run_stage.py             pipeline stages from the shell, one stage per process
  lstsq_from_yaml.py       one pruning run from the YAML above
notebooks/
  Llama8B.ipynb            the run that produced the numbers below: self-contained cells, outputs kept
  run_experiments.ipynb    the same stages through rmexp, one stage per cell
```

## Setup

Python 3.10+, CUDA. On the lab hub the kernel is `run_replacement` (torch 2.10, transformers 5.x, peft,
bitsandbytes, datasets). Elsewhere: `pip install -r requirements.txt`. Then either `pip install -e .`
or run from the repo folder: the notebooks and scripts add the repo root to `sys.path` themselves.

Set `runs` in `configs/llama3_8b_nf4.yaml` (output folder for models, adapters, results). `replaceme_repo`
is `.`: the `ReplaceMe/` folder of this repo. For `notebooks/Llama8B.ipynb` the same two things are the
environment variables `REPLACEME_REPO` and `REPLACEME_RUNS`.

## From the shell

```
python scripts/run_stage.py configs/llama3_8b_nf4.yaml baseline     # eval the base model
python scripts/run_stage.py configs/llama3_8b_nf4.yaml sft          # LoRA on the full model + eval
python scripts/run_stage.py configs/llama3_8b_nf4.yaml profile      # block profiles for every length in the config
python scripts/run_stage.py configs/llama3_8b_nf4.yaml plan         # blocks of every experiment -> plan.json
python scripts/run_stage.py configs/llama3_8b_nf4.yaml prune        # T + pruned model for every experiment
python scripts/run_stage.py configs/llama3_8b_nf4.yaml heal  --only llama-3-8b-numA4-shift20-nf4
python scripts/run_stage.py configs/llama3_8b_nf4.yaml restore      # restore + eval (T kept; without T where asked)
python scripts/run_stage.py configs/llama3_8b_nf4.yaml summary      # McNemar against fine-tuned

python scripts/lstsq_from_yaml.py configs/lstsq_joint_gsm8k.yaml --blocks 22,30   # one pruning run, nothing else
```

One stage per process keeps the GPU clean between stages; every stage skips outputs that already exist.
An experiment is one entry under `experiments:` in the YAML: explicit `blocks`, or `lengths` chosen greedily
from the profiles, or `top1_length`.

## Protocol

* calibration: 128 GSM8K train items as `Question: ...\nAnswer: ...`, max 256 tokens, the same texts for the profile and for T;
* T: `alpha_act 1.0`, `alpha_grad 0.5`, identity blend 0.15; `||T^T W|| / ||W||` of every carrier is printed, above ~3.1 the block did not recover;
* LoRA r8, alpha 16, MLP projections only, batch 2×4, lr 2e-4, one epoch, paged AdamW 8-bit, fp16 — the same for the SFT control and for heal;
* evaluation: HF generate, greedy, 256 new tokens, batch 8, 4-bit base + adapter without merge.

Blocks are 1-based `(start, end)`: 0-based layers `start..end-1` are deleted, layer `start-1` carries T.
EM on 1319 questions has SE ≈ 1.4 pt; models are compared with McNemar on per-question predictions.

## Results (nf4, HF generate)

| model | blocks | EM % |
|---|---|---|
| baseline | — | 10.77 |
| fine-tuned, LoRA on the full model | — | 59.59 |
| pruned, T only | (22,30) | 7.43 |
| pruned + heal, 24 layers | (22,30) | 50.11 |
| restored| (22,30) | 49.73 |
| restored | (19,23) (24,28) | 44.43 |
| restored | (20,23) (24,27) (28,30) | 48.29 |
| restored | (18,20) (21,23) (24,26) (27,29) | 39.65 |

Fine-tuned minus restored (22,30): +9.86 pt, 95 % CI [+7.06; +12.65], p < 1e-4. Splitting the 8 layers into
several blocks does not close the gap; the drop tracks how early the first deleted layer sits. The two
`shift` experiments in the config test position against fragmentation directly.

## Changes against upstream ReplaceMe

`distance_scored.py`: `hs = out.hidden_states[1:]` (upstream indexed from the embeddings and labelled every
block one layer too late) and three extra columns in the profile CSV (`grad_contrib`, `grad_contrib_fro`,
`grad_coher` with their ranks and scores). The `score` column and the selected block are unchanged.
`rmexp/replaceme_repo.py` re-applies both to a fresh upstream checkout, with a backup next to the file.
`lstsq_joint.py` and `utils.py` are unchanged.
