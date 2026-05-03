"""
controllers/approach_c.py — Approach C: LLM rewrites the optimiser Python file.

The LLM receives campaign context and the current optimiser code, then returns
a complete Python file defining a suggest(X_obs, y_obs, bounds) function.
The new code is executed in a subprocess sandbox with a 10-second timeout
and a whitelist of allowed imports.

Safety constraints:
  - Subprocess isolation (no shared memory with main process)
  - 10-second wall-clock timeout
  - Import whitelist: numpy, scipy, sklearn, math, random, json, itertools
  - No network access, no file writes
"""

import json
import logging
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from controllers.base import OllamaController

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = pathlib.Path(__file__).parent.parent / "prompts" / "system_c.txt"
_TEMPLATE = pathlib.Path(__file__).parent.parent / "optimiser_template.py"

ALLOWED_IMPORTS = {"numpy", "scipy", "sklearn", "math", "random", "json", "itertools", "np"}
SANDBOX_TIMEOUT = 10  # seconds


def _build_sandbox_wrapper(optimiser_code: str) -> str:
    """Wrap user code in a subprocess-safe script that reads args from stdin."""
    return f'''
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")  # prevent warnings from corrupting stdout JSON

# ── User-provided optimiser code ──────────────────────────────────────────
{optimiser_code}
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_obs = np.array(json.loads(sys.argv[1]))
    y_obs = np.array(json.loads(sys.argv[2]))
    bounds = np.array(json.loads(sys.argv[3]))
    result = suggest(X_obs, y_obs, bounds)
    if result is None:
        raise ValueError("suggest() returned None — function body is likely truncated (missing return statement)")
    print(json.dumps([float(x) for x in result]))
'''


def _run_in_sandbox(
    optimiser_code: str,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
) -> Optional[np.ndarray]:
    """Execute optimiser_code in a subprocess and return the suggested point."""
    wrapper = _build_sandbox_wrapper(optimiser_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path,
             json.dumps(X_obs.tolist()),
             json.dumps(y_obs.tolist()),
             json.dumps(bounds.tolist())],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT,
        )
        if result.returncode == 0:
            stdout = result.stdout.strip()
            if not stdout:
                logger.warning(f"Sandbox produced no output. stderr: {result.stderr[:200]}")
                return None
            x_next = np.array(json.loads(stdout))
            if x_next.shape == (bounds.shape[0],) and np.all(np.isfinite(x_next)):
                return x_next
            else:
                logger.warning(f"Sandbox output wrong shape or non-finite: {x_next}")
        #else:
        if result.returncode != 0:
            # Save failing code for inspection
            with open("./sdl_failed_code.py", "w") as f:
                f.write(optimiser_code)
            with open("./sdl_failed_stderr.txt", "w") as f:
                f.write(result.stderr)
            logger.warning(f"Sandbox error: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("Sandbox timed out")
    except json.JSONDecodeError as e:
        logger.warning(f"Sandbox JSON parse failed: {e}. stdout: {result.stdout[:100]!r}")
    except Exception as e:
        logger.warning(f"Sandbox exception: {e}")
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    return None


class ApproachCController(OllamaController):
    """
    Approach C: LLM rewrites the optimiser Python file.

    The LLM returns a complete Python module with a suggest() function.
    The code is executed in a subprocess sandbox.
    """

    _needs_gp_uncertainty = True

    def __init__(self, model: str = "qwen2.5-coder:7b", **kwargs):
        system = _SYSTEM_PROMPT.read_text() if _SYSTEM_PROMPT.exists() else _DEFAULT_SYSTEM
        kwargs.setdefault("max_retries", 3)
        kwargs.setdefault("timeout", 120)
        super().__init__(model=model, system_prompt=system, **kwargs)
        self._current_code = _TEMPLATE.read_text() if _TEMPLATE.exists() else _DEFAULT_TEMPLATE

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        progress = context["step"] / context["budget"]
        return (
            f"Campaign status:\n"
            f"  Step: {context['step']} / {context['budget']} ({100*progress:.0f}% complete)\n"
            f"  Best found (normalised): {context['best_normalised']:.3f}\n"
            f"  Improvement rate (last 10 steps): {context['improvement_rate']:.2f}\n"
            f"  GP uncertainty: {context['gp_uncertainty']:.4f}\n"
            f"  Dimensions: {context['bounds'].shape[0]}\n"
            f"  Observations so far: {context['n_obs']}\n\n"
            f"Current optimiser code:\n```python\n{self._current_code}\n```\n\n"
            f"Return a complete Python file with a suggest(X_obs, y_obs, bounds) function.\n"
            f"Allowed imports: numpy, scipy, sklearn, math, random.\n"
            f"Return ONLY the Python code, no markdown fences."
        )

    def _parse_response(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Strip markdown fences if present
        code = text.strip()
        for fence in ["```python", "```"]:
            if code.startswith(fence):
                code = code[len(fence):]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        if "def suggest" not in code:
            logger.warning("LLM response missing suggest() function — keeping current code")
            return {"strategy": "_custom", "params": {}, "x_next": None}

        # Try running in sandbox
        X_obs = context["X_obs"]
        y_obs = context["y_obs"]
        bounds = context["bounds"]

        x_next = _run_in_sandbox(code, X_obs, y_obs, bounds)
        if x_next is None:
            # New code failed — try last known-good code before falling back
            x_next = _run_in_sandbox(self._current_code, X_obs, y_obs, bounds)
        if x_next is not None:
            # Clip to bounds
            x_next = np.clip(x_next, bounds[:, 0], bounds[:, 1])
            self._current_code = code
            return {"strategy": "_custom", "params": {}, "x_next": x_next}

        # Sandbox failed — keep current code, return random
        logger.warning("Sandbox failed — falling back to random")
        return {"strategy": "random", "params": {}}


_DEFAULT_SYSTEM = """You are an autonomous optimisation controller for a self-driving laboratory.
Your task is to write a Python optimiser that suggests the next experiment to run.
You will receive the current optimiser code and campaign state, and must return an improved version.
The function signature must be: suggest(X_obs, y_obs, bounds) -> np.ndarray (shape: (d,))
Return ONLY valid Python code."""

_DEFAULT_TEMPLATE = """import numpy as np

def suggest(X_obs, y_obs, bounds):
    \"\"\"Default: uniform random sampling.\"\"\"
    d = bounds.shape[0]
    return np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])
"""
