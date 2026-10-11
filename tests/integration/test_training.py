import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from sepsis_most.graph import build_sofae_graph
from sepsis_most.training import (SepsisMOSTEnsemble, Standardiser, TrainingConfig, load_cohort,
                                  stratified_split, train_network)

ROOT = Path(__file__).resolve().parents[2]


def small_config(**opt):
    cfg = TrainingConfig.from_yaml(ROOT / "configs/model/sepsis_most.yaml").to_dict()
    cfg["optimisation"].update(epochs=2, batch_size=64, **opt)
    cfg["seeds"] = [42]
    return TrainingConfig.from_dict(cfg)


def test_config_file_matches_defaults():
    assert TrainingConfig.from_yaml(ROOT / "configs/model/sepsis_most.yaml") == TrainingConfig()


def test_split_is_stratified_and_disjoint():
    death = (np.random.default_rng(0).random(1000) < 0.1).astype(int)
    tr, va, te = stratified_split(death)
    assert len(set(tr) | set(va) | set(te)) == 1000
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert abs(death[tr].mean() - death.mean()) < 0.01


def test_training_is_reproducible_and_checkpoint_round_trips(cohort_files, onsets, tmp_path):
    data = load_cohort(*cohort_files)
    cfg = small_config()
    tr, va, _ = stratified_split(data.events[:, 0], cfg.data.split_seed, cfg.data.split_fractions)
    std = Standardiser.fit(data.X, data.S, tr)
    Xn, Sn = std.transform(data.X, data.S)
    assert np.all(Xn[:, :, :7] == data.X[:, :, :7])          # SOFA channels keep their scale
    g = build_sofae_graph(onsets, min_support=5)
    dev = torch.device("cpu")
    m1, best1, h1 = train_network(42, Xn, Sn, data, tr, va, g.organ_embeddings, g.coupling, cfg, dev)
    m2, best2, h2 = train_network(42, Xn, Sn, data, tr, va, g.organ_embeddings, g.coupling, cfg, dev)
    assert h1 == h2
    for k, v in m1.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k])
    ens = SepsisMOSTEnsemble([m1], std, g.organ_embeddings, g.coupling, data.bin_edges, cfg)
    pred = ens.predict(data.X, data.S)
    assert pred["cif"].shape == (len(data.X), 6, 8) and pred["event_prob"].shape == (len(data.X), 3)
    ens.save(tmp_path / "m.pt")
    pred2 = SepsisMOSTEnsemble.load(tmp_path / "m.pt").predict(data.X, data.S)
    np.testing.assert_array_equal(pred["cif"], pred2["cif"])


def test_command_line_pipeline(cohort_files, onsets, tmp_path):
    onsets.to_csv(tmp_path / "onsets.csv", index=False)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(small_config().to_dict()))
    py = [sys.executable]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(py + [str(ROOT / "scripts/build/build_sofae_graph.py"), "--onsets", str(tmp_path / "onsets.csv"),
                         "--out", str(tmp_path / "graph.npz"), "--min-support", "5"], check=True, env=env)
    subprocess.run(py + [str(ROOT / "scripts/train/train_sepsis_most.py"), "--temporal", str(cohort_files[0]),
                         "--static", str(cohort_files[1]), "--graph", str(tmp_path / "graph.npz"),
                         "--config", str(tmp_path / "cfg.yaml"), "--out", str(tmp_path / "model.pt"),
                         "--device", "cpu", "--isotonic"], check=True, env=env)
    subprocess.run(py + [str(ROOT / "scripts/predict/predict_sepsis_most.py"),
                         "--checkpoint", str(tmp_path / "model.pt"),
                         "--temporal", str(cohort_files[0]), "--static", str(cohort_files[1]),
                         "--out", str(tmp_path / "pred.npz"), "--calibrated"], check=True, env=env)
    pred = np.load(tmp_path / "pred.npz")
    assert pred["cif"].shape[1:] == (6, 8) and pred["cif_7d_calibrated"].shape[1] == 6
    summary = json.loads((tmp_path / "model.summary.json").read_text())
    assert summary["seeds"]["42"]["epochs"] == 2
    split = np.load(tmp_path / "model.split.npz")
    assert len(split["train"]) + len(split["val"]) + len(split["test"]) == 240
