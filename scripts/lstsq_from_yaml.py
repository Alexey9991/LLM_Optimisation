"""ReplaceMe.lstsq_joint.lstsq from a YAML of its arguments.

  python scripts/lstsq_from_yaml.py configs/lstsq_joint_gsm8k.yaml
  python scripts/lstsq_from_yaml.py configs/lstsq_joint_gsm8k.yaml --blocks 20,22 23,25 26,28 29,31 --save_path /tmp/x/pruned
"""
import argparse
import inspect
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rmexp.config import Cfg  # noqa: E402
from rmexp.replaceme_repo import load_replaceme  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--blocks", nargs="*", help="override selected_blocks, e.g. 22,30 or 20,22 23,25")
    ap.add_argument("--save_path")
    a = ap.parse_args()

    kw = yaml.safe_load(open(a.config))
    if a.blocks:
        kw["selected_blocks"] = [tuple(int(v) for v in b.split(",")) for b in a.blocks]
    if a.save_path:
        kw["save_path"] = a.save_path
    rm = load_replaceme(Cfg(replaceme_repo=str(ROOT)))
    unknown = set(kw) - set(inspect.signature(rm.lj.lstsq).parameters)
    assert not unknown, f"unknown lstsq arguments: {sorted(unknown)}"
    Path(kw["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    out = rm.lj.lstsq(**kw)
    print("pruned model ->", out)


if __name__ == "__main__":
    main()
