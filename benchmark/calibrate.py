"""Confidence-threshold calibration (Phase 4).

Reads a benchmark results file whose neuroroute rows carry the
final confidence score and measured quality (keyword-coverage
proxy), then asks the only question calibration can answer: *do
the confidence bands separate good answers from bad ones, and
where should the return threshold sit?*

For the current policy (return >= 0.85, escalate < 0.60) and a
sweep of alternative return thresholds it reports, per band:
count, mean quality, min quality. Recommendation rule: the lowest
return threshold whose band keeps mean quality >= 0.95 AND has no
row below 0.50 — loosening the threshold is only worth it if the
returned band stays clean; if the current 0.85 already satisfies
the rule at the lowest cost, it stands confirmed.

Caveat printed with the results: the recorded confidence is the
FINAL attempt's (post-escalation), and quality is the benchmark's
keyword proxy on 24 queries — directionally useful, not a
production calibration set.

Usage:
    python benchmark/calibrate.py [results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RETURN_CANDIDATES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CURRENT_RETURN, CURRENT_ESCALATE = 0.85, 0.60


def band_stats(rows: list[dict], return_t: float,
               escalate_t: float) -> dict:
    bands = {"return": [], "verify": [], "escalate": []}
    for r in rows:
        c = r["confidence"]
        band = ("return" if c >= return_t
                else "verify" if c >= escalate_t else "escalate")
        bands[band].append(r["quality"])
    return {name: {"n": len(qs),
                   "mean_quality": round(sum(qs) / len(qs), 3) if qs else None,
                   "min_quality": min(qs) if qs else None}
            for name, qs in bands.items()}


def recommend(rows: list[dict]) -> float | None:
    """Lowest return threshold whose returned band stays clean
    (mean quality >= 0.95, no row < 0.50), or None if no candidate
    qualifies — including the current one."""
    for cand in RETURN_CANDIDATES:
        band = band_stats(rows, cand, CURRENT_ESCALATE)["return"]
        if (band["n"] and band["mean_quality"] >= 0.95
                and band["min_quality"] >= 0.50):
            return cand
    return None


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else Path(__file__).with_name("results.json"))
    data = json.loads(path.read_text())
    rows = [r["neuroroute"] for r in data["rows"]
            if "confidence" in r.get("neuroroute", {})]
    if not rows:
        print("No rows with confidence scores — re-run the benchmark "
              "with the updated runner first.")
        return 1
    print(f"Calibration over {len(rows)} neuroroute rows from {path.name}\n")
    cur = band_stats(rows, CURRENT_RETURN, CURRENT_ESCALATE)
    print(f"Current policy (return >= {CURRENT_RETURN}, "
          f"escalate < {CURRENT_ESCALATE}):")
    for name, s in cur.items():
        print(f"  {name:<9} n={s['n']:<3} mean quality={s['mean_quality']} "
              f"min={s['min_quality']}")
    print("\nReturn-threshold sweep (escalate threshold fixed at "
          f"{CURRENT_ESCALATE}):")
    for cand in RETURN_CANDIDATES:
        band = band_stats(rows, cand, CURRENT_ESCALATE)["return"]
        print(f"  return >= {cand:.2f}: n={band['n']:<3} "
              f"mean={band['mean_quality']} min={band['min_quality']}")
    rec = recommend(rows)
    if rec is None:
        print("\nRecommendation: no candidate keeps the returned band "
              "clean — keep 0.85 and gather more data.")
    elif rec == CURRENT_RETURN:
        print(f"\nRecommendation: {CURRENT_RETURN} stands confirmed — "
              "it is the lowest threshold whose returned band stays clean.")
    else:
        print(f"\nRecommendation: return threshold {rec} — the lowest "
              "whose returned band stays clean (apply with "
              "NR_RETURN_THRESHOLD).")
    print("\nCaveat: final-attempt confidence, keyword-proxy quality, "
          f"n={len(rows)} — directional, not production calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
