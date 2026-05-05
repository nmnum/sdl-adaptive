"""
analyse_decisions.py — Inspect switch_logs.json to see what the LLM actually decided.

Usage:
    python analyse_decisions.py --results_dir results_prompt_test/

Prints per-seed decision sequences and computes:
  - For approach_b: strategy used at each step, coherence score
  - For approach_a: beta value at each step, beta_range per seed
"""

import argparse
import json
import pathlib
import numpy as np

DATASETS = ["coatings", "pareto_20201218", "pareto_20201223", "pareto_20210104", "pareto_20210112"]


def analyse_approach_a(logs_path: pathlib.Path, budget: int):
    with open(logs_path) as f:
        logs = json.load(f)

    beta_ranges, beta_seqs = [], []
    for i, seed_log in enumerate(logs):
        betas = [p.get("beta", None) for _, _, p in seed_log["decisions"]]
        betas = [b for b in betas if b is not None]
        steps = [t for t, _, p in seed_log["decisions"] if p.get("beta") is not None]
        beta_seqs.append(list(zip(steps, betas)))
        if betas:
            beta_ranges.append(max(betas) - min(betas))
            print(f"  Seed {i}: beta_range={max(betas)-min(betas):.1f}  "
                  f"min={min(betas):.2f}  max={max(betas):.2f}")
            # Show the sequence
            seq_str = "  ".join(f"t{t}:β={b:.1f}" for t, b in zip(steps, betas))
            print(f"    {seq_str}")
        else:
            print(f"  Seed {i}: no beta values logged")

    if beta_ranges:
        print(f"  Mean beta_range: {np.mean(beta_ranges):.1f} ± {np.std(beta_ranges):.1f}")
        print(f"  (Expected: ~400 early → ~0.5 late = range ~400 if working correctly)")


def analyse_approach_b(logs_path: pathlib.Path, budget: int):
    PHASE_MAP = {
        "lhs": "explore", "random": "explore",
        "ucb": "mid", "thompson": "mid",
        "ei": "exploit", "pi": "exploit",
    }

    with open(logs_path) as f:
        logs = json.load(f)

    for i, seed_log in enumerate(logs):
        decisions = seed_log["decisions"]
        strategies = [s for _, s, _ in decisions]
        steps = [t for t, _, _ in decisions]

        # Count actual switches
        switches = sum(strategies[j] != strategies[j-1] for j in range(1, len(strategies)))

        # Coherence: does the phase of the strategy match expected phase?
        coherent, total = 0, 0
        for t, s, _ in decisions:
            progress = t / budget
            phase = PHASE_MAP.get(s, "unknown")
            if progress < 0.20:
                expected = "explore"
            elif progress < 0.60:
                expected = "mid"
            elif progress < 0.80:
                expected = "mid"  # ei/ucb-low both count as mid-to-exploit
            else:
                expected = "exploit"
            if phase == expected or (expected == "mid" and phase in ("mid", "exploit")):
                coherent += 1
            total += 1

        coherence = coherent / total if total > 0 else 0
        print(f"  Seed {i}: switches={switches}  coherence={coherence:.0%}")
        seq = "  ".join(f"t{t}:{s}" for t, s, _ in decisions)
        print(f"    {seq}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_prompt_test")
    args = parser.parse_args()
    results_dir = pathlib.Path(args.results_dir)

    for ds in DATASETS:
        ds_dir = results_dir / ds
        if not ds_dir.exists():
            continue

        # Try to get budget from curves.npy
        budget = 91  # fallback
        for cond in ["approach_a", "approach_b"]:
            curves_path = ds_dir / cond / "curves.npy"
            if curves_path.exists():
                import numpy as np_
                budget = np_.load(curves_path).shape[1]
                break

        print(f"\n{'='*55}")
        print(f"Dataset: {ds}  (budget={budget})")

        for cond in ["approach_a", "approach_b", "approach_c"]:
            logs_path = ds_dir / cond / "switch_logs.json"
            if not logs_path.exists():
                continue
            print(f"\n  [{cond}]")
            if cond == "approach_a":
                analyse_approach_a(logs_path, budget)
            else:
                analyse_approach_b(logs_path, budget)


if __name__ == "__main__":
    main()
