"""Command-line entry for the same stages as the notebook, one stage per process (clean GPU every time).

  python scripts/run_stage.py configs/llama3_8b_nf4.yaml baseline
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml sft            # train + eval
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml profile
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml plan
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml prune   [--only NAME]
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml heal    [--only NAME]
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml restore [--only NAME] [--keep-restored]
  python scripts/run_stage.py configs/llama3_8b_nf4.yaml summary
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rmexp import Experiment, load_config, resolve_blocks, stats  # noqa: E402
from rmexp.evaluate import result_path  # noqa: E402
from rmexp.pipeline import eval_baseline, eval_sft, needed_lengths, train_sft  # noqa: E402
from rmexp.profile import profile, profile_csv  # noqa: E402
from rmexp.replaceme_repo import load_replaceme  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stage", choices=["baseline", "sft", "profile", "plan", "prune", "heal", "restore", "summary"])
    ap.add_argument("--only", help="experiment name")
    ap.add_argument("--keep-restored", action="store_true")
    a = ap.parse_args()

    cfg, specs = load_config(a.config)
    if a.only:
        specs = [s for s in specs if s.name == a.only]
        assert specs, a.only
    rm = load_replaceme(cfg) if a.stage in ("profile", "plan", "prune") else None

    if a.stage == "baseline":
        eval_baseline(cfg)
    elif a.stage == "sft":
        train_sft(cfg)
        eval_sft(cfg)
    elif a.stage == "profile":
        for k in needed_lengths(specs):
            profile(cfg, rm, k)
    else:
        profiles = {k: pd.read_csv(profile_csv(cfg, k)) for k in needed_lengths(specs) if profile_csv(cfg, k).exists()}
        exps = [Experiment(cfg, s, resolve_blocks(cfg, s, profiles, rm)) for s in specs]
        if a.stage == "plan":
            for e in exps:
                print(e)
        elif a.stage == "prune":
            for e in exps:
                e.prune(rm)
        elif a.stage == "heal":
            for e in exps:
                e.heal()
        elif a.stage == "restore":
            for e in exps:
                if e.spec.eval_pruned:
                    e.eval_pruned()
                    e.eval_healed()
                e.eval_restored(False, delete_after=not a.keep_restored)
                if e.spec.restore_without_T:
                    e.eval_restored(True, delete_after=not a.keep_restored)
        elif a.stage == "summary":
            runs = {"baseline": (result_path(cfg.results_dir, "baseline", cfg.precision), "—"),
                    "fine-tuned": (result_path(cfg.results_dir, "sft", cfg.precision), "—")}
            for e in exps:
                f = e.result_files()["replaceme"]
                if f.exists():
                    runs[f"restored | {e.spec.name}"] = (f, " ".join(f"({s},{t})" for s, t in e.blocks))
            pd.set_option("display.width", 220)
            print(stats.summary(runs).to_string(index=False))


if __name__ == "__main__":
    main()
