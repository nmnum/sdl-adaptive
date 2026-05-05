"""
controllers/approach_b.py — Approach B: LLM selects strategy via table-completion.

The prompt presents the full phase→strategy table and asks the model to return
the JSON for the current phase row. This reframes strategy selection as a
structured lookup task rather than an open-ended decision, which is more
reliable for code-oriented LLMs like qwen2.5-coder.

The LLM retains genuine agency in two ways:
  1. It can adjust UCB beta based on campaign state signals
  2. It can override to thompson/random if improvement_rate < 0.10
These are the decisions that require reasoning; the phase mapping is explicit.
"""

import pathlib
from typing import Any, Dict

from controllers.base import OllamaController

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_b.txt"

VALID_STRATEGIES = {"ucb", "ei", "pi", "thompson", "random", "lhs"}

_PHASE_TABLE = [
    (0.00, 0.20, "lhs",      {},           "EARLY  (0–20%)"),
    (0.20, 0.45, "ucb",      {"beta": 20}, "MID-A  (20–45%)"),
    (0.45, 0.65, "ucb",      {"beta": 3},  "MID-B  (45–65%)"),
    (0.65, 0.80, "ei",       {},           "LATE   (65–80%)"),
    (0.80, 1.01, "ei",       {},           "FINAL  (80–100%)"),
]


def _lookup_phase(progress: float):
    for lo, hi, strategy, params, label in _PHASE_TABLE:
        if lo <= progress < hi:
            return strategy, dict(params), label
    return "ei", {}, "FINAL"


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
        _, _, current_phase_label = _lookup_phase(progress)

        # Build the table with the current phase highlighted
        rows = []
        for lo, hi, strat, params, label in _PHASE_TABLE:
            marker = " ← YOU ARE HERE" if lo <= progress < hi else ""
            param_str = f'"beta": {params["beta"]}' if "beta" in params else ""
            params_display = "{" + param_str + "}"
            rows.append(f'  | {label} | {strat:<9} | {params_display:<20} |{marker}')
        table = "\n".join(rows)

        stagnating = context["improvement_rate"] < 0.10
        override_note = (
            "\nNOTE: improvement_rate < 0.10 — campaign is stagnating. "
            "Consider thompson instead of the phase default.\n"
            if stagnating else ""
        )

        return (
            f"Strategy table:\n"
            f"  | Phase         | Strategy  | Params               |\n"
            f"  |---------------|-----------|----------------------|\n"
            f"{table}\n"
            f"{override_note}\n"
            f"Campaign state:\n"
            f"  Progress: {100*progress:.0f}% | "
            f"Best: {context['best_normalised']:.3f} | "
            f"Improvement rate: {context['improvement_rate']:.2f} | "
            f"GP uncertainty: {context['gp_uncertainty']:.3f}\n\n"
            f"Return the JSON for the '{current_phase_label}' row.\n"
            f"You may adjust beta for UCB. You may use 'thompson' if stagnating.\n"
            f"Respond with JSON only."
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        progress = context["step"] / context["budget"]
        phase_strategy, phase_params, _ = _lookup_phase(progress)

        parsed = self._extract_json(text)
        if parsed and "strategy" in parsed:
            strategy = str(parsed.get("strategy", "")).lower().strip()
            if strategy in VALID_STRATEGIES:
                params = parsed.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                if strategy == "ucb":
                    if "beta" not in params:
                        params["beta"] = phase_params.get("beta", 3.0)
                    try:
                        params["beta"] = float(max(0.01, min(1000.0, params["beta"])))
                    except (ValueError, TypeError):
                        params["beta"] = phase_params.get("beta", 3.0)
                self._consecutive_calls = (
                    self._consecutive_calls + 1
                    if strategy == self._current_strategy else 1
                )
                self._current_strategy = strategy
                self._current_params = params
                return {"strategy": strategy, "params": params}

        # Fallback: use the phase table directly (no LLM needed)
        self._consecutive_calls += 1
        self._current_strategy = phase_strategy
        self._current_params = phase_params
        return {"strategy": phase_strategy, "params": dict(phase_params)}


_DEFAULT_SYSTEM = """Return the JSON for the current phase row from the strategy table.
Respond with JSON only: {"strategy": "<name>", "params": {}}"""
