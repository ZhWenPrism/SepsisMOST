"""Train Sepsis-MOST.

Steps
1. load the temporal tensor with labels and the static matrix (docs/DATA_FORMAT.md);
2. split the rows 70:15:15 into training, validation and internal test sets,
   stratified by ICU death (or read the split from ``--split-file``);
3. fit the standardisation on the training split;
4. train one network per configured seed with early stopping on the
   validation mean C-index;
5. save the frozen model, the split indices and a training summary.

    python scripts/train/train_sepsis_most.py \\
        --temporal tensors.npz --static static.npz --graph sofae_graph.npz \\
        --config configs/model/sepsis_most.yaml --out runs/sepsis_most.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sepsis_most.constants import ORGANS
from sepsis_most.graph import SOFAEGraph
from sepsis_most.training import (SepsisMOSTEnsemble, Standardiser, TrainingConfig, isotonic_fit,
                                  load_cohort, resolve_device, stratified_split, train_network)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temporal", required=True, help="temporal tensor with labels (.npz)")
    ap.add_argument("--static", required=True, help="static matrix (.npz)")
    ap.add_argument("--graph", required=True, help="SOFAE graph (.npz) from build_sofae_graph.py")
    ap.add_argument("--config", default="configs/model/sepsis_most.yaml")
    ap.add_argument("--out", required=True, help="output checkpoint (.pt)")
    ap.add_argument("--split-file", help="optional .npz with integer arrays train, val and test")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--isotonic", action="store_true",
                    help="also fit isotonic maps of the 7-day organ CIF on the validation split")
    args = ap.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    data = load_cohort(args.temporal, args.static)
    if args.split_file:
        sp = np.load(args.split_file, allow_pickle=False)
        tr, va, te = sp["train"], sp["val"], sp["test"]
    else:
        tr, va, te = stratified_split(data.events[:, 0], cfg.data.split_seed, cfg.data.split_fractions)
    print(f"device {device} | rows {len(data.X):,} | train/val/test {len(tr):,}/{len(va):,}/{len(te):,}")

    std = Standardiser.fit(data.X, data.S, tr, cfg.data.n_raw_channels)
    Xn, Sn = std.transform(data.X, data.S)
    graph = SOFAEGraph.load(args.graph)
    emb, coup = graph.organ_embeddings, graph.coupling

    members, summary = [], {"config": cfg.to_dict(), "n_train": int(len(tr)), "n_val": int(len(va)),
                            "n_test": int(len(te)), "seeds": {}}
    for seed in cfg.seeds:
        model, best, history = train_network(seed, Xn, Sn, data, tr, va, emb, coup, cfg, device, log=print)
        members.append(model.cpu())
        summary["seeds"][str(seed)] = {"best_val_mean_cindex": float(best), "epochs": len(history),
                                       "val_mean_cindex": [float(v) for v in history]}
        print(f"seed {seed}: best validation mean C-index {best:.4f} after {len(history)} epochs")

    ens = SepsisMOSTEnsemble(members, std, emb, coup, data.bin_edges, cfg)
    if args.isotonic:
        cif_val = ens.predict(data.X[va], data.S[va])["cif"]
        maps = []
        for k in range(len(ORGANS)):
            m = data.atrisk[va, k].astype(bool)
            maps.append(isotonic_fit(cif_val[m, k, -1], (data.evt[va, k][m] == 1).astype(float)))
        ens.isotonic = maps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ens.save(out)
    np.savez(out.with_suffix(".split.npz"), train=tr, val=va, test=te)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out}, {out.with_suffix('.split.npz')} and {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
