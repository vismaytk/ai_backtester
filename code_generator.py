"""
═══════════════════════════════════════════════════════════════════════════════
  Code Generator — Converts ParsedStrategy → Runnable Python Code
  ─────────────────────────────────────────────────────────────────────────
  Takes the structured output from parser.py and generates complete,
  executable Python code using vectorbt for backtesting.
═══════════════════════════════════════════════════════════════════════════════
"""

from parser import (
    ParsedStrategy, Rule, Indicator, IndicatorType, Condition
)
from typing import List


def _generate_indicator_code(indicator: Indicator, var_prefix: str = "ind") -> tuple:
    """
    Generate Python code to compute an indicator.
    Returns: (setup_code, series_variable_name)
    """
    t = indicator.type
    p = indicator.params

    if t == IndicatorType.PRICE:
        return ("", "close")

    elif t == IndicatorType.VALUE:
        val = p.get("value", 0)
        return ("", str(val))

    elif t == IndicatorType.SMA:
        period = p.get("period", 50)
        var = f"sma_{period}"
        code = f"{var} = close.rolling(window={period}).mean()"
        return (code, var)

    elif t == IndicatorType.EMA:
        period = p.get("period", 20)
        var = f"ema_{period}"
        code = f"{var} = close.ewm(span={period}, adjust=False).mean()"
        return (code, var)

    elif t == IndicatorType.RSI:
        period = p.get("period", 14)
        var = f"rsi_{period}"
        code = f"""# RSI ({period}-period) — Wilder's smoothing
delta = close.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1/{period}, min_periods={period}, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/{period}, min_periods={period}, adjust=False).mean()
rs = avg_gain / avg_loss
{var} = 100 - (100 / (1 + rs))"""
        return (code, var)

    elif t == IndicatorType.MACD_LINE:
        fast = p.get("fast", 12)
        slow = p.get("slow", 26)
        var = "macd_line"
        code = f"""# MACD Line ({fast},{slow})
ema_fast = close.ewm(span={fast}, adjust=False).mean()
ema_slow = close.ewm(span={slow}, adjust=False).mean()
{var} = ema_fast - ema_slow"""
        return (code, var)

    elif t == IndicatorType.MACD_SIGNAL:
        fast = p.get("fast", 12)
        slow = p.get("slow", 26)
        signal = p.get("signal", 9)
        var = "macd_signal"
        code = f"""# MACD Signal ({fast},{slow},{signal})
ema_fast = close.ewm(span={fast}, adjust=False).mean()
ema_slow = close.ewm(span={slow}, adjust=False).mean()
macd_line = ema_fast - ema_slow
{var} = macd_line.ewm(span={signal}, adjust=False).mean()"""
        return (code, var)

    elif t == IndicatorType.MACD_HISTOGRAM:
        fast = p.get("fast", 12)
        slow = p.get("slow", 26)
        signal = p.get("signal", 9)
        var = "macd_histogram"
        code = f"""# MACD Histogram ({fast},{slow},{signal})
ema_fast = close.ewm(span={fast}, adjust=False).mean()
ema_slow = close.ewm(span={slow}, adjust=False).mean()
macd_line = ema_fast - ema_slow
macd_signal = macd_line.ewm(span={signal}, adjust=False).mean()
{var} = macd_line - macd_signal"""
        return (code, var)

    elif t == IndicatorType.BB_UPPER:
        period = p.get("period", 20)
        std = p.get("std", 2)
        var = "bb_upper"
        code = f"""# Bollinger Bands ({period},{std})
bb_middle = close.rolling(window={period}).mean()
bb_std = close.rolling(window={period}).std()
{var} = bb_middle + {std} * bb_std"""
        return (code, var)

    elif t == IndicatorType.BB_LOWER:
        period = p.get("period", 20)
        std = p.get("std", 2)
        var = "bb_lower"
        code = f"""# Bollinger Bands ({period},{std})
bb_middle = close.rolling(window={period}).mean()
bb_std = close.rolling(window={period}).std()
{var} = bb_middle - {std} * bb_std"""
        return (code, var)

    elif t == IndicatorType.BB_MIDDLE:
        period = p.get("period", 20)
        var = "bb_middle"
        code = f"bb_middle = close.rolling(window={period}).mean()"
        return (code, var)

    return ("", "close")


def _generate_condition_code(left_var: str, condition: Condition, right_var: str) -> str:
    """Generate the boolean condition expression."""
    if condition == Condition.CROSSES_ABOVE:
        return f"({left_var} > {right_var}) & ({left_var}.shift(1) <= {right_var}.shift(1))"
    elif condition == Condition.CROSSES_BELOW:
        return f"({left_var} < {right_var}) & ({left_var}.shift(1) >= {right_var}.shift(1))"
    elif condition == Condition.IS_ABOVE:
        return f"({left_var} > {right_var}) & ({left_var}.shift(1) <= {right_var}.shift(1))"
    elif condition == Condition.IS_BELOW:
        return f"({left_var} < {right_var}) & ({left_var}.shift(1) >= {right_var}.shift(1))"
    return "False"


def _generate_condition_code_for_right_value(left_var: str, condition: Condition, value: str) -> str:
    """Generate condition for comparisons against a constant value."""
    if condition == Condition.CROSSES_ABOVE:
        return f"({left_var} > {value}) & ({left_var}.shift(1) <= {value})"
    elif condition == Condition.CROSSES_BELOW:
        return f"({left_var} < {value}) & ({left_var}.shift(1) >= {value})"
    elif condition == Condition.IS_ABOVE:
        return f"({left_var} > {value}) & ({left_var}.shift(1) <= {value})"
    elif condition == Condition.IS_BELOW:
        return f"({left_var} < {value}) & ({left_var}.shift(1) >= {value})"
    return "False"


def generate_backtest_code(
    strategy: ParsedStrategy,
    ticker: str = "^NSEI",
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    init_cash: int = 1_000_000,
    fees: float = 0.001,
) -> str:
    """
    Generate complete, executable Python backtest code from a ParsedStrategy.

    Returns a string of Python code that can be exec()'d.
    """
    setup_codes = set()
    setup_code_list = []
    indicator_vars = {}

    def _ensure_indicator(ind: Indicator) -> str:
        """Generate code for an indicator and return its variable name."""
        code, var_name = _generate_indicator_code(ind)
        if code and code not in setup_codes:
            setup_codes.add(code)
            setup_code_list.append(code)
        return var_name

    # Process all indicators from all rules
    entry_conditions = []
    exit_conditions = []

    for rule in strategy.entry_rules:
        left_var = _ensure_indicator(rule.left)
        right_var = _ensure_indicator(rule.right)

        if rule.right.type == IndicatorType.VALUE:
            cond = _generate_condition_code_for_right_value(
                left_var, rule.condition, right_var
            )
        else:
            cond = _generate_condition_code(left_var, rule.condition, right_var)
        entry_conditions.append(cond)

    for rule in strategy.exit_rules:
        left_var = _ensure_indicator(rule.left)
        right_var = _ensure_indicator(rule.right)

        if rule.right.type == IndicatorType.VALUE:
            cond = _generate_condition_code_for_right_value(
                left_var, rule.condition, right_var
            )
        else:
            cond = _generate_condition_code(left_var, rule.condition, right_var)
        exit_conditions.append(cond)

    # Combine conditions
    if entry_conditions:
        entry_expr = " | ".join(f"({c})" for c in entry_conditions)
    else:
        entry_expr = "pd.Series(False, index=close.index)"

    if exit_conditions:
        exit_expr = " | ".join(f"({c})" for c in exit_conditions)
    else:
        exit_expr = "pd.Series(False, index=close.index)"

    # Build the indicator computation variable list for the chart
    indicator_chart_traces = _generate_chart_traces(strategy)

    # Assemble full code
    code = f'''import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════
ticker = "{ticker}"
data = yf.download(ticker, start="{start_date}", end="{end_date}")
close = data["Close"].squeeze()

# ═══════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════
{chr(10).join(setup_code_list)}

# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════
entries = ({entry_expr}).fillna(False)
exits = ({exit_expr}).fillna(False)

buy_signals = close[entries]
sell_signals = close[exits]

# ═══════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════
portfolio = vbt.Portfolio.from_signals(
    close,
    entries=entries.values,
    exits=exits.values,
    init_cash={init_cash},
    fees={fees},
    freq="1D",
)

# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════
total_return = portfolio.total_return()
sharpe_ratio = portfolio.sharpe_ratio()
max_drawdown = portfolio.max_drawdown()
win_rate = portfolio.trades.win_rate()
total_trades = portfolio.trades.count()
profit_factor = portfolio.trades.profit_factor()
final_value = portfolio.final_value()
equity = portfolio.value()
trades_readable = portfolio.trades.records_readable

# Buy & Hold Benchmark
bh_portfolio = vbt.Portfolio.from_holding(close, init_cash={init_cash}, freq="1D")
bh_return = bh_portfolio.total_return()
bh_sharpe = bh_portfolio.sharpe_ratio()
bh_drawdown = bh_portfolio.max_drawdown()
bh_equity = bh_portfolio.value()

# ═══════════════════════════════════════════════════════════════════
# RESULTS DICT (for the dashboard)
# ═══════════════════════════════════════════════════════════════════
results = {{
    "total_return": total_return,
    "sharpe_ratio": sharpe_ratio,
    "max_drawdown": max_drawdown,
    "win_rate": win_rate,
    "total_trades": total_trades,
    "profit_factor": profit_factor,
    "final_value": final_value,
    "equity": equity,
    "bh_return": bh_return,
    "bh_sharpe": bh_sharpe,
    "bh_drawdown": bh_drawdown,
    "bh_equity": bh_equity,
    "close": close,
    "buy_signals": buy_signals,
    "sell_signals": sell_signals,
    "trades_readable": trades_readable,
    "ticker": ticker,
{indicator_chart_traces}
}}
'''
    return code


def _generate_chart_traces(strategy: ParsedStrategy) -> str:
    """Generate code to include indicator series in the results dict for charting."""
    lines = []
    seen = set()

    for rule in strategy.entry_rules + strategy.exit_rules:
        for ind in [rule.left, rule.right]:
            if ind.type in (IndicatorType.VALUE, IndicatorType.PRICE):
                continue

            _, var_name = _generate_indicator_code(ind)
            if var_name not in seen:
                seen.add(var_name)
                label = repr(ind)
                lines.append(f'    "{label}": {var_name},')

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from parser import parse_strategy

    test = "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200"
    parsed = parse_strategy(test)
    code = generate_backtest_code(parsed)
    print(code)
