"""Манифест ансамбля, веса и лимит 2 ГБ (ТЗ §7), без скачивания самих весов."""
import json
import re

from conftest import ROOT

MODELS = ROOT / "models"
LIMIT_MIB = 2048


def load(name):
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def test_every_ensemble_member_is_delivered_with_sha256():
    weights = {f["name"]: f for f in load("weights.json")["files"]}
    for m in load("ensemble.json")["models"]:
        name = m["ckpt"].split("/")[-1]
        assert name in weights, f"{name} не поставляется через weights.json"
        assert re.fullmatch(r"[0-9a-f]{64}", weights[name]["sha256"])
        assert weights[name]["url"].startswith("https://github.com/Enoti4Ka44/FalconREID/releases/")
    assert "gallery_index.npz" in weights, "индекс галереи сервиса тоже поставляется"


def test_weights_fit_limit():
    total = sum(f["bytes"] for f in load("weights.json")["files"]) / 2**20
    assert total < LIMIT_MIB, f"веса {total:.0f} МиБ превышают лимит {LIMIT_MIB}"


def test_threshold_and_postproc_consistent():
    thr = load("threshold.json")
    assert 0.0 < thr["threshold"] < 1.0
    assert abs(0.7 * thr["heldout_F1"] + 0.3 * thr["heldout_TNR"] - thr["heldout_score"]) < 1e-3
    pp = load("postproc.json")
    assert len(pp["ensemble"]["weights"]) >= 1 and pp["dba_k"] >= 0
