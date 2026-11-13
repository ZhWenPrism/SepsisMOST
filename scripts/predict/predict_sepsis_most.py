"""Score a cohort with a frozen Sepsis-MOST model.

The temporal file needs the array ``X`` and the static file the array ``S``
(docs/DATA_FORMAT.md). The model is applied as trained, with no refitting.

    python scripts/predict/predict_sepsis_most.py --checkpoint runs/sepsis_most.pt \\
        --temporal external_tensors.npz --static external_static.npz --out predictions.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sepsis_most.training import SepsisMOSTEnsemble
from sepsis_most.training.data import load_scoring_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--out", required=True, help="output .npz with cif and event_prob")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calibrated", action="store_true", help="add isotonic-calibrated 7-day CIFs")
    args = ap.parse_args()

    X, S = load_scoring_inputs(args.temporal, args.static)
    ens = SepsisMOSTEnsemble.load(args.checkpoint, device=args.device)
    pred = ens.predict(X, S, device=args.device, calibrated=args.calibrated)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, bin_edges=ens.bin_edges, **pred)
    print(f"scored {len(X):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
