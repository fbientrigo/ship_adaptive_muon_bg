#!/usr/bin/env python
"""Weight-class provenance audit of a muon PKL (structure only, no interpretation).

Reports exact unique ``w`` values with counts and sum(w), cross-tabs against id,
zero-weight rows, exact duplicate rows, x/y zero and +-7.5 m clipping fractions,
and the sha256 of the file. It never asserts what ``w`` means.

    python scripts/audit_weight_provenance.py --input X.pkl[.gz] --label afterMS --out audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ship_muon_bg.data_contracts import loader  # noqa: E402

# Predicted classes from FairShip muonShieldOptimization/extractMuonsAndUpdateWeight.py
# (weightMbias, weightCharm, weightBeauty; optional /100 boost). A prediction, not a fact.
BASE = {"mbias": 768.75, "charm": 326.16, "beauty": 9.37}
PREDICTED = sorted({b / f for b in BASE.values() for f in (1.0, 100.0)} | {0.0})


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audit(arr, label, path=None):
    w, ids = arr[:, 7], arr[:, 6]
    p = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    vals, inv, counts = np.unique(w, return_inverse=True, return_counts=True)
    classes = []
    for k, (v, n) in enumerate(zip(vals, counts)):
        sel = inv == k
        classes.append({
            "w": float(v), "n": int(n), "frac": float(n) / len(w), "sum_w": float(w[sel].sum()),
            "n_mu_plus": int((ids[sel] == -13).sum()), "n_mu_minus": int((ids[sel] == 13).sum()),
            "p_min": float(p[sel].min()), "p_med": float(np.median(p[sel])), "p_max": float(p[sel].max()),
            "predicted": bool(np.isclose(v, PREDICTED, rtol=1e-6, atol=1e-9).any()),
        })
    return {
        "label": label, "file": None if path is None else os.path.basename(path),
        "sha256": None if path is None else sha256(path),
        "n_rows": int(len(w)), "sum_w": float(w.sum()), "n_unique_w": int(len(vals)),
        "n_zero_w": int((w == 0).sum()),
        "unpredicted_w": [c["w"] for c in classes if not c["predicted"]],
        "n_exact_duplicate_rows": int(len(arr) - len(np.unique(arr, axis=0))),
        "frac_x_zero": float((arr[:, 3] == 0).mean()), "frac_y_zero": float((arr[:, 4] == 0).mean()),
        "n_x_clip_7p5": int((np.abs(arr[:, 3]) == 7.5).sum()), "n_y_clip_7p5": int((np.abs(arr[:, 4]) == 7.5).sum()),
        "z_min": float(arr[:, 5].min()), "z_max": float(arr[:, 5].max()),
        "classes": classes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = audit(loader.load_muon_pkl(a.input), a.label, a.input)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(a.label, res["n_rows"], "rows; classes:", [(c["w"], c["n"]) for c in res["classes"]],
          "unpredicted:", res["unpredicted_w"])


if __name__ == "__main__":
    main()
