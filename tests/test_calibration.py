"""Phase 4 calibration: thresholds are env-overridable (with safe
fallbacks), and the banding/recommendation logic behaves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

import calibrate  # noqa: E402
from backend.validation import confidence_estimator  # noqa: E402


def test_thresholds_default(monkeypatch):
    monkeypatch.delenv("NR_RETURN_THRESHOLD", raising=False)
    monkeypatch.delenv("NR_ESCALATE_THRESHOLD", raising=False)
    assert confidence_estimator.thresholds() == (0.85, 0.60)


def test_thresholds_env_override(monkeypatch):
    monkeypatch.setenv("NR_RETURN_THRESHOLD", "0.80")
    monkeypatch.setenv("NR_ESCALATE_THRESHOLD", "0.55")
    assert confidence_estimator.thresholds() == (0.80, 0.55)


def test_thresholds_reject_nonsense(monkeypatch):
    monkeypatch.setenv("NR_RETURN_THRESHOLD", "0.40")   # below escalate
    monkeypatch.setenv("NR_ESCALATE_THRESHOLD", "0.55")
    assert confidence_estimator.thresholds() == (0.85, 0.60)
    monkeypatch.setenv("NR_RETURN_THRESHOLD", "banana")
    assert confidence_estimator.thresholds() == (0.85, 0.60)


def _rows(*pairs):
    return [{"confidence": c, "quality": q} for c, q in pairs]


def test_band_stats_separates_bands():
    rows = _rows((0.95, 1.0), (0.90, 0.9), (0.70, 0.6), (0.40, 0.2))
    bands = calibrate.band_stats(rows, 0.85, 0.60)
    assert bands["return"] == {"n": 2, "mean_quality": 0.95, "min_quality": 0.9}
    assert bands["verify"]["n"] == 1
    assert bands["escalate"]["n"] == 1


def test_recommend_keeps_current_when_clean():
    # Returned band is clean already at 0.70 — but the rule picks the
    # LOWEST clean threshold, so this data recommends loosening.
    rows = _rows((0.72, 1.0), (0.90, 1.0), (0.99, 0.98), (0.65, 0.4))
    assert calibrate.recommend(rows) == 0.70


def test_recommend_none_when_band_dirty_everywhere():
    rows = _rows((0.99, 0.3), (0.90, 1.0))  # a bad answer at the top
    assert calibrate.recommend(rows) is None


def test_band_components_decompose_bands():
    rows = [
        {"confidence": 0.95, "quality": 1.0, "evidence_coverage": 0.9,
         "retrieval_relevance": 0.8, "model_agreement": 0.85,
         "validation_passed": True},
        {"confidence": 0.10, "quality": 1.0, "evidence_coverage": 0.1,
         "retrieval_relevance": 0.2, "model_agreement": 0.30,
         "validation_passed": False},
        {"confidence": 0.92, "quality": 1.0},  # no components: skipped
    ]
    comps = calibrate.band_components(rows, 0.85, 0.60)
    assert comps["return"]["evidence_coverage"] == 0.9
    assert comps["return"]["validation_rate"] == 1.0
    assert comps["escalate"]["evidence_coverage"] == 0.1
    assert comps["escalate"]["validation_rate"] == 0.0
    assert comps["verify"] is None
