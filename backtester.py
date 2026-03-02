"""
═══════════════════════════════════════════════════════════════════════════════
  Backtest Engine — Safely executes generated backtest code
  ─────────────────────────────────────────────────────────────────────────
  Takes generated Python code string, executes it in a controlled
  namespace, and returns the results dict.
═══════════════════════════════════════════════════════════════════════════════
"""

import traceback
from typing import Dict, Any, Optional


def run_backtest(code: str) -> Dict[str, Any]:
    """
    Execute generated backtest code and return results.

    Args:
        code: Complete Python code string (from code_generator.py)

    Returns:
        dict with keys: total_return, sharpe_ratio, max_drawdown,
             win_rate, total_trades, profit_factor, final_value,
             equity, bh_equity, close, buy_signals, sell_signals,
             trades_readable, ticker, and any indicator series.

    Raises:
        BacktestError: If execution fails
    """
    namespace = {}

    try:
        exec(code, namespace)
    except Exception as e:
        raise BacktestError(
            f"Backtest execution failed: {str(e)}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

    results = namespace.get("results")
    if results is None:
        raise BacktestError(
            "Backtest code did not produce a 'results' dictionary. "
            "This is likely a code generation bug."
        )

    return results


class BacktestError(Exception):
    """Raised when backtest execution fails."""
    pass
