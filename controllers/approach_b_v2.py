"""
controllers/approach_b.py — Approach B: LLM selects strategy + tunes parameters.

Design: the phase→strategy mapping is explicit (table-completion, proven to work
with qwen2.5-coder:7b). The LLM's genuine agency is in choosing the UCB beta
and overriding for stagnation. This separates two tasks:
  - Strategy selection: handled by the table (LLM confirms the row)
  - Parameter tuning: handled by the LLM based on campaign state signals

This makes approach_b meaningfully distinct from approach_a (which fixes the
strategy to UCB and only tunes beta) while remaining tractable for a 7b model.
"""

import pathlib
from typing import Any, Dict

from controllers.base import OllamaController

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_b.txt"

VALID_STRATEGIES = {"ucb", "ei", "pi", "thompson", "random", "lhs"}

# Phase table: no beta here — LLM chooses beta from context
_PHASE_TABLE = [
    (0.00, 0.20, "lhs", "EARLY  (0–20%)"),
    (0.20, 0.65, "ucb", "MID    (20–65%)"),
    (0.65, 1.01, "ei",  "LATE   (65–100%)"),
]


def _lookup_phase(progress: float):
    for lo, hi, strategy, label in _PHASE_TABLE:
        if lo <= progress < hi:
            return strategy, label
    return "ei", "LATE"


class ApproachBController(OllamaController):
    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen2.5-coder:7b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_strategy = "lhs"
        self._current_params: Dict = {}
        self._consecutive_calls = 0

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        phase_strategy, current_phase_label = _lookup_phase(progress)
        stagnating = context["improvement_rate"] < 0.10

        # Build table with current phase marked
        rows = []
        for lo, hi, strat, label in _PHASE_TABLE:
            marker = " ← YOU ARE HERE" if lo <= progress < hi else ""
            rows.append(f"  | {label} | {strat:<9} |{marker}")
        table = "\n".join(rows)

        stagnation_note = (
            "\nNOTE: improvement_rate < 0.10 — stagnating. "
            "Override to 'thompson' if UCB is not finding improvements.\n"
            if stagnating else ""
        )

        beta_guidance = ""
        if phase_strategy == "ucb":
            beta_guidance = (
                f"\nFor UCB, choose beta based on campaign state:\n"
                f"  - High uncertainty (gp_uncertainty > 0.3): beta in [10, 50]\n"
                f"  - Good progress (improvement_rate > 0.3): beta in [1, 5]\n"
                f"  - Stagnating (improvement_rate < 0.1): beta in [20, 100]\n"
                f"  - Otherwise: beta in [3, 20]\n"
            )

        return (
            f"Strategy table:\n"
            f"  | Phase         | Strategy  |\n"
            f"  |---------------|----------|\n"
            f"{table}\n"
            f"{stagnation_note}"
            f"{beta_guidance}\n"
            f"Campaign state:\n"
            f"  Progress: {100*progress:.0f}% (step {context['step']} / {context['budget']})\n"
            f"  Best found (normalised): {context['best_normalised']:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty: {context['gp_uncertainty']:.3f}\n\n"
            f"Return the JSON for the '{current_phase_label}' row.\n"
            f"{'Choose an appropriate beta for UCB given the campaign state above.' if phase_strategy == 'ucb' else ''}\n"
            f"Respond with JSON only:\n"
            f'  {{"strategy": "ucb", "params": {{"beta": <float>}}}}  ← if UCB phase\n'
            f'  {{"strategy": "ei", "params": {{}}}}                  ← if LATE phase'
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        progress = context["step"] / context["budget"]
        phase_strategy, _ = _lookup_phase(progress)

        parsed = self._extract_json(text)
        if parsed and "strategy" in parsed:
            strategy = str(parsed.get("strategy", "")).lower().strip()
            if strategy in VALID_STRATEGIES:
                params = parsed.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                if strategy == "ucb":
                    if "beta" not in params:
                        # LLM forgot to specify beta — derive from context
                        rate = context["improvement_rate"]
                        unc = context["gp_uncertainty"]
                        if rate < 0.10:
                            params["beta"] = 30.0
                        elif unc > 0.3:
                            params["beta"] = 15.0
                        elif rate > 0.3:
                            params["beta"] = 2.0
                        else:
                            params["beta"] = 8.0
                    try:
                        params["beta"] = float(max(0.01, min(1000.0, params["beta"])))
                    except (ValueError, TypeError):
                        params["beta"] = 8.0
                self._consecutive_calls = (
                    self._consecutive_calls + 1
                    if strategy == self._current_strategy else 1
                )
                self._current_strategy = strategy
                self._current_params = params
                return {"strategy": strategy, "params": params}

        # Fallback: use phase table
        self._consecutive_calls += 1
        fallback_params = {}
        if phase_strategy == "ucb":
            fallback_params = {"beta": 8.0}
        self._current_strategy = phase_strategy
        self._current_params = fallback_params
        return {"strategy": phase_strategy, "params": fallback_params}


_DEFAULT_SYSTEM = """Return the JSON for the current phase row from the strategy table.
For UCB phases, choose beta based on campaign state. Respond with JSON only."""
