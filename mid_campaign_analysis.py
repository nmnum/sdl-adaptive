"""
mid_campaign_analysis.py — Compute mid-campaign regret conditioned on bad starts.

Filters to seeds where best_norm < threshold at step n_init.
Computes AUC over the mid-campaign window only.

Can be pointed at a power run (LLM conditions only) alongside the original
results dir for fixed baselines via --baseline_results_dir.

Usage:
    # Original full results:
    python mid_campaign_analysis.py --results_dir results/ --threshold 0.80

    # Power run (LLM only) with baselines from original run:
    python mid_campaign_analysis.py \
        --results_dir results_power/ \
        --baseline_results_dir results/ \
        --threshold 0.80
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

DATASETS = [
    "coatings", "pareto_20201218", "pareto_20201223",
    "pareto_20210104", "pareto_20210112"
]
LLM_CONDITIONS   = ["approach_a", "approach_b", "approach_c"]
FIXED_CONDITIONS = [
    "fixed_random", "fixed_lhs", "fixed_ucb_low",
    "fixed_ucb_high", "fixed_ei",
    "mock_approach_a", "mock_approach_b", "mock_approach_c",
]
ALL_CONDITIONS = FIXED_CONDITIONS + LLM_CONDITIONS

LABELS = {
    "fixed_random": "Random",    "fixed_lhs": "LHS",
    "fixed_ucb_low": "UCB β=0.2", "fixed_ucb_high": "UCB β=400",
    "fixed_ei": "EI (fixed)",    "mock_approach_a": "Mock-A",
    "mock_approach_b": "Mock-B", "mock_approach_c": "Mock-C",
    "approach_a": "LLM-A",       "approach_b": "LLM-B",
    "approach_c": "LLM-C",
}


def load_curves_norm(cond_dir, global_min, y_range, n_init, window_end):
    """Load and normalise curves, return (n_seeds, budget) array."""
    curves_path = cond_dir / "curves.npy"
    if not curves_path.exists():
        return None
    curves = np.load(curves_path)
    normed = np.clip((curves - global_min) / y_range, 0, 1)
    return normed


def get_global_stats(ds, results_dir, baseline_dir):
    """Get global_best and global_min from metrics_summary if available."""
    for rdir in [results_dir, baseline_dir]:
        if rdir is None:
            continue
        summary = rdir / "metrics_summary.csv"
        if summary.exists():
            df = pd.read_csv(summary)
            row = df[df.dataset == ds].head(1)
            if len(row):
                gb = float(row["global_best"].iloc[0])
                gm = float(row["global_min"].iloc[0])
                return gb, gm
    # Fall back: infer from any available curves + metrics
    for rdir in [results_dir, baseline_dir]:
        if rdir is None:
            continue
        for cond in ALL_CONDITIONS:
            mp = rdir / ds / cond / "metrics_per_seed.json"
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)[0]
                fn = m["final_best_normalised"]
                if fn and fn > 0:
                    return m["final_best"] / fn, 0.0
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--baseline_results_dir", default=None,
                        help="Directory with fixed baseline curves (if different from results_dir)")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--window_frac", type=float, default=0.5)
    args = parser.parse_args()

    results_dir  = pathlib.Path(args.results_dir)
    baseline_dir = pathlib.Path(args.baseline_results_dir) if args.baseline_results_dir else results_dir

    print(f"Mid-campaign AUC | threshold={args.threshold} | "
          f"window=steps {args.n_init}–{args.window_frac*100:.0f}% of budget")
    if baseline_dir != results_dir:
        print(f"LLM conditions:      {results_dir}")
        print(f"Baseline conditions: {baseline_dir}")
    print()

    rows = []

    for ds in DATASETS:
        global_best, global_min = get_global_stats(ds, results_dir, baseline_dir)
        if global_best is None:
            continue

        y_range = global_best - global_min
        if y_range < 1e-9:
            continue

        # Identify qualifying seeds from a reference condition
        # Use first available condition in either directory
        ref_curves_norm = None
        ref_budget = None
        for rdir in [results_dir, baseline_dir]:
            for cond in LLM_CONDITIONS + FIXED_CONDITIONS:
                p = rdir / ds / cond / "curves.npy"
                if p.exists():
                    c = np.load(p)
                    ref_budget = c.shape[1]
                    ref_curves_norm = np.clip((c - global_min) / y_range, 0, 1)
                    break
            if ref_curves_norm is not None:
                break

        if ref_curves_norm is None:
            continue

        window_end = max(args.n_init + 1, int(args.window_frac * ref_budget))
        qualifying = [i for i in range(len(ref_curves_norm))
                      if ref_curves_norm[i, args.n_init] < args.threshold]

        print(f"{'='*60}")
        print(f"Dataset: {ds}  (budget={ref_budget})")
        print(f"  Qualifying seeds (best_norm < {args.threshold} "
              f"at step {args.n_init}): {len(qualifying)} / {len(ref_curves_norm)}")

        if len(qualifying) < 3:
            print(f"  Too few qualifying seeds — try higher threshold")
            continue

        print(f"\n  {'Condition':<20} {'n':>4} {'mid-AUC':>9} {'±SD':>7}")
        print(f"  {'-'*44}")

        cond_aucs = {}
        for cond in ALL_CONDITIONS:
            # Look in primary results_dir first, then baseline_dir
            for rdir in [results_dir, baseline_dir]:
                cn = load_curves_norm(rdir / ds / cond, global_min, y_range,
                                      args.n_init, window_end)
                if cn is not None:
                    break
            if cn is None:
                continue

            # Use qualifying seed indices — but cap at available seeds
            q = [i for i in qualifying if i < len(cn)]
            if not q:
                continue

            mid_aucs = cn[q, args.n_init:window_end].mean(axis=1)
            cond_aucs[cond] = mid_aucs
            label = LABELS.get(cond, cond)
            print(f"  {label:<20} {len(mid_aucs):>4} {mid_aucs.mean():>9.3f} "
                  f"{mid_aucs.std():>7.3f}")
            rows.append({
                "dataset": ds, "condition": cond, "label": label,
                "n_qualifying": len(mid_aucs),
                "mid_auc_mean": float(mid_aucs.mean()),
                "mid_auc_std":  float(mid_aucs.std()),
                "threshold": args.threshold,
            })

        # Statistical tests: each LLM vs best available fixed baseline
        print(f"\n  Statistical tests (paired on qualifying seeds):")

        best_fixed = max(
            [(c, cond_aucs[c].mean()) for c in FIXED_CONDITIONS if c in cond_aucs],
            key=lambda x: x[1], default=(None, None)
        )
        if best_fixed[0]:
            print(f"  Best fixed baseline: {LABELS[best_fixed[0]]} "
                  f"(mean={best_fixed[1]:.3f})")

        for llm_cond in LLM_CONDITIONS:
            if llm_cond not in cond_aucs:
                continue
            llm_aucs = cond_aucs[llm_cond]

            # vs best fixed
            if best_fixed[0] and best_fixed[0] in cond_aucs:
                fix_aucs = cond_aucs[best_fixed[0]]
                n = min(len(llm_aucs), len(fix_aucs))
                t, p = stats.ttest_ind(llm_aucs[:n], fix_aucs[:n])
                sig = " *" if p < 0.05 else ""
                print(f"    {LABELS[llm_cond]} vs {LABELS[best_fixed[0]]}: "
                      f"Δ={llm_aucs.mean()-fix_aucs.mean():+.3f}, p={p:.3f}{sig}")

        # LLM pairwise
        llm_present = [c for c in LLM_CONDITIONS if c in cond_aucs]
        if len(llm_present) >= 2:
            print(f"  LLM pairwise:")
            for i, c1 in enumerate(llm_present):
                for c2 in llm_present[i+1:]:
                    a1, a2 = cond_aucs[c1], cond_aucs[c2]
                    n = min(len(a1), len(a2))
                    t, p = stats.ttest_ind(a1[:n], a2[:n])
                    sig = " *" if p < 0.05 else ""
                    print(f"    {LABELS[c1]} vs {LABELS[c2]}: "
                          f"Δ={a1.mean()-a2.mean():+.3f}, p={p:.3f}{sig}")
        print()

    out_df = pd.DataFrame(rows)
    out_path = results_dir / "mid_campaign_conditioned.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if len(out_df):
        print(f"\n{'='*60}")
        print("SUMMARY: Mean mid-campaign AUC (qualifying seeds, all datasets)")
        print(f"{'='*60}")
        summary = (out_df.groupby("label")["mid_auc_mean"]
                   .mean().sort_values(ascending=False))
        for label, mean in summary.items():
            marker = " ◄" if label.startswith("LLM") else ""
            print(f"  {label:<20} {mean:.3f}{marker}")


if __name__ == "__main__":
    main()
