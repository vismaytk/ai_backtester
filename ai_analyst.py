"""
═══════════════════════════════════════════════════════════════════════════════
  AI Research Analyst — Level 2
  ─────────────────────────────────────────────────────────────────────────
  After a backtest completes, feeds the metrics to the AI model and
  receives a professional research note with:
    • Performance summary
    • Risk analysis
    • Market context
    • Comparison to benchmark
    • Concrete improvement suggestions
═══════════════════════════════════════════════════════════════════════════════
"""

from ai_client import call_ai, AIError
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — research analyst persona
# ══════════════════════════════════════════════════════════════════════════════

ANALYST_SYSTEM_PROMPT = """You are a senior quantitative analyst at a top hedge fund.

You are writing a concise research note for a portfolio manager. Your analysis
must be data-driven, insightful, and actionable.

FORMAT YOUR RESPONSE IN MARKDOWN with these sections:

## 📊 Performance Summary
Brief overview of the strategy's returns, risk metrics, and trade activity.

## ⚠️ Risk Analysis
What the max drawdown, Sharpe ratio, and win rate reveal about the strategy's
risk profile. Is the risk-adjusted return acceptable?

## 🌍 Market Context
Based on the time period and asset, what market conditions (bull/bear trends,
volatility regimes, events like COVID-19 crash of 2020) likely affected
performance? Be specific about the asset.

## 📐 Strategy vs. Benchmark
How does the strategy compare to simple buy-and-hold? Calculate and comment
on the alpha. Was the complexity worth it?

## 💡 Improvement Suggestions
Provide EXACTLY 3 concrete, actionable suggestions to improve the strategy.
Each should be specific (e.g., "Add a 2% trailing stop-loss" rather than
"manage risk better"). Format as a numbered list.

RULES:
- Keep the entire note under 400 words
- Use actual numbers from the metrics provided
- Be honest about poor performance — don't sugarcoat
- Use professional but accessible language
- Use bullet points for clarity where appropriate"""


def _format_metrics(
    strategy_description: str,
    results: dict,
    ticker: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Format backtest metrics into a clean prompt for the AI."""

    def safe_fmt(val, fmt=".2%"):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "N/A"
        try:
            return f"{val:{fmt}}"
        except (ValueError, TypeError):
            return str(val)

    total_return = results.get("total_return", 0)
    bh_return = results.get("bh_return", 0)

    # Calculate alpha safely
    try:
        alpha = total_return - bh_return
    except:
        alpha = 0

    metrics_text = f"""STRATEGY: {strategy_description}
ASSET: {ticker or results.get('ticker', 'Unknown')}
PERIOD: {start_date} to {end_date}

STRATEGY METRICS:
- Total Return: {safe_fmt(total_return)}
- Sharpe Ratio: {safe_fmt(results.get('sharpe_ratio'), '.3f')}
- Max Drawdown: {safe_fmt(results.get('max_drawdown'))}
- Win Rate: {safe_fmt(results.get('win_rate'))}
- Total Trades: {results.get('total_trades', 'N/A')}
- Profit Factor: {safe_fmt(results.get('profit_factor'), '.2f')}
- Final Portfolio Value: {safe_fmt(results.get('final_value'), ',.0f')}

BENCHMARK (Buy & Hold):
- Buy & Hold Return: {safe_fmt(bh_return)}
- Buy & Hold Sharpe: {safe_fmt(results.get('bh_sharpe'), '.3f')}
- Buy & Hold Max Drawdown: {safe_fmt(results.get('bh_drawdown'))}

COMPARISON:
- Alpha (Strategy - Benchmark): {safe_fmt(alpha, '+.2%')}
- Strategy {"outperformed" if alpha > 0 else "underperformed"} buy-and-hold by {safe_fmt(abs(alpha))}"""

    return metrics_text


def generate_research_note(
    strategy_description: str,
    results: dict,
    ticker: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    """
    Generate a professional research note from backtest results.

    Args:
        strategy_description: The original strategy in plain English
        results:              dict from backtester.run_backtest()
        ticker:               Ticker symbol used
        start_date:           Backtest start date
        end_date:             Backtest end date

    Returns:
        Markdown-formatted research note string

    Raises:
        AIError: If API call fails
    """
    metrics_prompt = _format_metrics(
        strategy_description, results, ticker, start_date, end_date
    )

    user_prompt = f"""Analyze the following backtest results and write a research note:

{metrics_prompt}

Write your analysis now. Be specific, use the actual numbers, and provide
actionable improvement suggestions."""

    note = call_ai(
        prompt=user_prompt,
        system_instruction=ANALYST_SYSTEM_PROMPT,
        temperature=0.7,  # slightly creative for natural writing
        max_output_tokens=2048,
    )

    return note
