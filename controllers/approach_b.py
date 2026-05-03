"""
controllers/approach_b.py — Approach B: LLM selects from 6 named strategies.

The LLM receives campaign context and outputs a strategy name and parameters.
Available strategies: ucb, ei, pi, thompson, random, lhs.
"""

import pathlib
from typing import Any, Dict

from controllers.base import OllamaController

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_b.txt"

VALID_STRATEGIES = {"ucb", "ei", "pi", "thompson", "random", "lhs"}


class ApproachBController(OllamaController):
    """
    Approach B: LLM switches between 6 named strategies.

    Decision output: {"strategy": <name>, "params": {<optional kwargs>}}
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen2.5-coder:7b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_strategy = "random"
        self._current_params: Dict = {}

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        return (
            f"Campaign status:\n"
            f"  Step: {context['step']} / {context['budget']} ({100*progress:.0f}% complete)\n"
            f"  Best found (normalised): {context['best_normalised']:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty (mean posterior std): {context['gp_uncertainty']:.4f}\n"
            f"  Current strategy: {self._current_strategy}\n\n"
            f"Available strategies: ucb, ei, pi, thompson, random, lhs\n"
            f"For ucb, you may specify beta (float, default 1.0).\n\n"
            f"Respond with JSON only:\n"
            f'  {{"strategy": "<name>", "params": {{}}}}\n'
            f"or for UCB:\n"
            f'  {{"strategy": "ucb", "params": {{"beta": 2.0}}}}'
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self._extract_json(text)
        if parsed and "strategy" in parsed:
            strategy = str(parsed.get("strategy", "")).lower().strip()
            if strategy in VALID_STRATEGIES:
                params = parsed.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                # Validate beta for UCB
                if strategy == "ucb" and "beta" in params:
                    try:
                        params["beta"] = float(max(0.01, min(1000.0, params["beta"])))
                    except (ValueError, TypeError):
                        params["beta"] = 1.0
                self._current_strategy = strategy
                self._current_params = params
                return {"strategy": strategy, "params": params}

        return {"strategy": self._current_strategy, "params": self._current_params}


_DEFAULT_SYSTEM = """You are an autonomous optimisation controller for a self-driving laboratory.
Your task is to select the best optimisation strategy given the current campaign state.
Choose from: ucb (exploration-exploitation via β), ei (expected improvement), pi (probability of
improvement), thompson (Thompson sampling), random (uniform random), lhs (space-filling).
Respond with JSON only."""
