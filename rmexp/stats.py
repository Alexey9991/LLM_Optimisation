"""Result tables and the paired test. EM on 1319 questions has SE ~1.4 pt, so neighbours are compared with
McNemar on the per-question predictions, never by the difference of two EM numbers."""
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_result(path) -> Tuple[Dict[int, bool], dict]:
    r = json.load(open(path))
    return {int(x["id"]): bool(x["correct"]) for x in r["predictions"]}, r


def mcnemar(a: Dict[int, bool], b: Dict[int, bool]) -> dict:
    """Exact McNemar on the discordant pairs; diff_pt = EM(A) - EM(B) in points with a normal 95% CI."""
    ids = sorted(set(a) & set(b))
    n01 = sum(a[i] and not b[i] for i in ids)
    n10 = sum(b[i] and not a[i] for i in ids)
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n) if n else 1.0
    d = (n01 - n10) / len(ids)
    se = math.sqrt(n - (n01 - n10)**2 / len(ids)) / len(ids) if n else 0.0
    return dict(A_only=n01, B_only=n10, diff_pt=round(d * 100, 2),
                ci95=f"[{(d - 1.96 * se) * 100:+.2f}; {(d + 1.96 * se) * 100:+.2f}]", p=round(p, 4))


def summary(runs: Dict[str, Tuple[Path, str]], reference: str = "fine-tuned") -> pd.DataFrame:
    """runs: label -> (results json, blocks text). Every row is compared with `reference` by McNemar."""
    res = {k: load_result(f) for k, (f, _) in runs.items() if Path(f).exists()}
    rows = []
    for k, (_, blocks) in runs.items():
        if k not in res:
            continue
        r = {"model": k, "blocks": blocks, "EM %": round(res[k][1]["exact_match"] * 100, 2),
             "correct": res[k][1]["correct"], "layers": res[k][1].get("layers"), "hw": res[k][1].get("hw", "")}
        if k != reference and reference in res:
            m = mcnemar(res[reference][0], res[k][0])
            r.update({f"{reference} minus this, pt": m["diff_pt"], "95% CI": m["ci95"], "p": m["p"]})
        rows.append(r)
    return pd.DataFrame(rows)


def pairs(runs: Dict[str, Tuple[Path, str]], against: str) -> pd.DataFrame:
    res = {k: load_result(f) for k, (f, _) in runs.items() if Path(f).exists()}
    if against not in res:
        return pd.DataFrame()
    return pd.DataFrame([{"A": against, "B": k, **mcnemar(res[against][0], res[k][0])}
                         for k in res if k != against])
