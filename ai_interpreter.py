"""
═══════════════════════════════════════════════════════════════════════════════
  AI Strategy Interpreter — Level 1
  ─────────────────────────────────────────────────────────────────────────
  Sends the user's plain-English strategy to the AI model and receives
  back complete, executable Python/vectorbt backtest code.

  Handles ambiguous descriptions like:
    "buy when stock looks oversold and volume is spiking"

  Auto-retries once if the generated code fails execution.
═══════════════════════════════════════════════════════════════════════════════
"""

from ai_client import call_ai, AIError, AILogicError
import re
import json


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — constrains the AI to generate safe, executable code
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert quantitative trading strategy code generator.

Given a user's trading strategy described in plain English, generate COMPLETE, 
EXECUTABLE Python code that implements a backtest of that strategy.

YOUR CODE MUST FOLLOW THIS EXACT STRUCTURE:

1. Import ONLY these libraries: yfinance, vectorbt, pandas, numpy
2. Download OHLCV data using yfinance with the parameters provided
3. Compute any required technical indicators using pandas/numpy
4. Generate boolean entry (buy) and exit (sell) signal Series
5. Run the backtest using vectorbt.Portfolio.from_signals()
6. Compute all metrics and store them in a `results` dict

CRITICAL RULES:
- Do NOT import os, sys, subprocess, pathlib, shutil, or any file/network I/O
- Do NOT use print() statements
- Do NOT read or write any files
- Do NOT make any network calls except the yfinance download
- The code MUST create a variable called `results` which is a dict
- The `results` dict MUST contain EXACTLY these keys:
    - "total_return": portfolio.total_return()
    - "sharpe_ratio": portfolio.sharpe_ratio()
    - "max_drawdown": portfolio.max_drawdown()
    - "win_rate": portfolio.trades.win_rate()
    - "total_trades": portfolio.trades.count()
    - "profit_factor": portfolio.trades.profit_factor()
    - "final_value": portfolio.final_value()
    - "equity": portfolio.value()
    - "bh_return": buy_hold_portfolio.total_return()
    - "bh_sharpe": buy_hold_portfolio.sharpe_ratio()
    - "bh_drawdown": buy_hold_portfolio.max_drawdown()
    - "bh_equity": buy_hold_portfolio.value()
    - "close": the close price Series
    - "buy_signals": close[entries] (price at buy points)
    - "sell_signals": close[exits] (price at sell points)
    - "trades_readable": portfolio.trades.records_readable
    - "ticker": the ticker string
- Also include any computed indicator Series in results with descriptive keys
  (e.g., "SMA(50)": sma_50, "RSI(14)": rsi_14) so they can be plotted
- Use .squeeze() on downloaded data columns to get 1D Series
- Use .fillna(False) on entry/exit signals
- EXPLICIT SIZING: Pass `size=1.0` to Portfolio.from_signals() to ensure 100% equity sizing.
- SHIFT SIGNALS: To prevent lookahead bias, entry orders must execute on the next day's open or close. We'll simulate this by shifting signals: `entries = entries.shift(1).fillna(False)` and `exits = exits.shift(1).fillna(False)`.
- Pass entries.values and exits.values (numpy arrays) to Portfolio.from_signals()

RESPONSE FORMAT:
You MUST output your response in TWO parts:
Part 1: A JSON Intermediate Representation (IR) wrapped in ```json tags.
Part 2: The executable Python code wrapped in ```python tags.

If the strategy is contradictory, ambiguous, or nonsensical, set status to "error" and explain why in "message". Do not output code in this case.

Example Output (Success):
```json
{
  "status": "success",
  "indicators": ["SMA 50", "SMA 200"],
  "entry_logic": "SMA 50 > SMA 200",
  "exit_logic": "SMA 50 < SMA 200",
  "warnings": []
}
```
```python
import yfinance as yf...
```

Example Output (Error):
```json
{
  "status": "error",
  "message": "The strategy says to buy and sell when RSI > 70, which is contradictory."
}
```
"""


def _build_user_prompt(
    strategy: str,
    ticker: str,
    start_date: str,
    end_date: str,
    init_cash: int,
    fees: float,
) -> str:
    """Build the user prompt with parameters embedded."""
    return f"""Generate Python backtest code for this strategy:

STRATEGY: {strategy}

USE THESE EXACT PARAMETERS:
- Ticker: "{ticker}"
- Start date: "{start_date}"
- End date: "{end_date}"
- Initial cash: {init_cash}
- Transaction fees: {fees}
- Frequency: "1D"

Remember: You MUST output the ```json block first, then the ```python block. If the strategy is invalid, output only the JSON with status="error"."""


def _parse_ai_response(raw: str) -> tuple[dict, str]:
    """
    Extract the JSON IR and Python code from the AI's response.
    Returns (ir_dict, python_code).
    """
    ir_data = {"status": "error", "message": "Failed to parse AI response."}
    code = ""
    
    # Try finding JSON block
    json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            ir_data = json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            ir_data["message"] = "AI returned invalid JSON."
    
    # Try finding Python code block
    py_match = re.search(r'```python\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if py_match:
        code = py_match.group(1).strip()
    else:
        # Fallback if AI didn't use markdown fences for code
        # Remove JSON block and assume the rest is code if it starts with import
        rest = re.sub(r'```json\s*.*?\s*```', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
        lines = rest.split('\n')
        start_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import '):
                start_idx = i
                break
        if start_idx != -1:
            code = '\n'.join(lines[start_idx:]).strip()
            
    return ir_data, code


def _validate_code_safety(code: str) -> list:
    """
    Light safety check — flag dangerous imports/calls.
    Returns a list of warnings (empty = safe).
    """
    warnings = []
    dangerous_patterns = [
        (r'\bimport\s+os\b', "Imports 'os' module"),
        (r'\bimport\s+sys\b', "Imports 'sys' module"),
        (r'\bimport\s+subprocess\b', "Imports 'subprocess' module"),
        (r'\bimport\s+shutil\b', "Imports 'shutil' module"),
        (r'\bimport\s+pathlib\b', "Imports 'pathlib' module"),
        (r'\bopen\s*\(', "Uses file open()"),
        (r'\bexec\s*\(', "Uses exec()"),
        (r'\beval\s*\(', "Uses eval()"),
        (r'\b__import__\s*\(', "Uses __import__()"),
        (r'\brequests\b', "Uses requests library"),
        (r'\burllib\b', "Uses urllib library"),
    ]
    for pattern, desc in dangerous_patterns:
        if re.search(pattern, code):
            warnings.append(desc)

    return warnings


def generate_strategy_code(
    strategy: str,
    ticker: str = "^NSEI",
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    init_cash: int = 1_000_000,
    fees: float = 0.001,
    max_retries: int = 1,
) -> tuple[dict, str]:
    """
    Use AI to interpret a natural language strategy and generate
    executable Python backtest code.

    Args:
        strategy:    Plain English strategy description
        ticker:      Yahoo Finance ticker symbol
        start_date:  Backtest start date
        end_date:    Backtest end date
        init_cash:   Initial portfolio cash
        fees:        Transaction fee rate
        max_retries: Number of retries if code fails validation

    Returns:
        tuple containing (ir_dict, executable Python code string)

    Raises:
        AILogicError: If the user's strategy is contradictory/invalid.
        AIError: If AI API fails or code is unsafe
    """
    user_prompt = _build_user_prompt(
        strategy, ticker, start_date, end_date, init_cash, fees
    )

    # First attempt
    raw_response = call_ai(
        prompt=user_prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=4096,
    )

    ir_dict, code = _parse_ai_response(raw_response)
    
    if ir_dict.get("status") == "error":
        raise AILogicError(ir_dict.get("message", "The AI rejected the strategy logic."))
    
    if not code:
        raise AIError("AI did not return any Python code.")

    # Safety check
    safety_issues = _validate_code_safety(code)
    if safety_issues:
        if max_retries > 0:
            # Ask AI to fix
            fix_prompt = (
                f"Your previous code had safety issues: {', '.join(safety_issues)}.\n"
                f"Please regenerate the code WITHOUT any of these. "
                f"Only use yfinance, vectorbt, pandas, numpy.\n\n"
                f"Original strategy: {strategy}\n"
                f"Parameters: ticker={ticker}, start={start_date}, end={end_date}, "
                f"cash={init_cash}, fees={fees}"
            )
            raw_response = call_ai(
                prompt=fix_prompt,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=4096,
            )
            ir_dict, code = _parse_ai_response(raw_response)
            
            if ir_dict.get("status") == "error":
                raise AILogicError(ir_dict.get("message", "The AI rejected the strategy logic."))
            
            if not code:
                raise AIError("AI did not return any Python code on retry.")

            # Re-check
            safety_issues = _validate_code_safety(code)
            if safety_issues:
                raise AIError(
                    f"AI generated unsafe code even after retry. "
                    f"Issues: {', '.join(safety_issues)}"
                )
        else:
            raise AIError(
                f"AI generated unsafe code. Issues: {', '.join(safety_issues)}"
            )

    return ir_dict, code


def retry_with_error(
    strategy: str,
    previous_code: str,
    error_message: str,
    ticker: str = "^NSEI",
    start_date: str = "2019-01-01",
    end_date: str = "2024-01-01",
    init_cash: int = 1_000_000,
    fees: float = 0.001,
) -> str:
    """
    When generated code fails execution, feed the error back to the AI
    for a corrected version.

    Returns:
        tuple containing (ir_dict, Fixed Python code string)

    Raises:
        AIError: If retry also fails
    """
    retry_prompt = f"""Your previously generated Python code failed with this error:

ERROR:
{error_message}

ORIGINAL STRATEGY: {strategy}
PARAMETERS: ticker="{ticker}", start="{start_date}", end="{end_date}", cash={init_cash}, fees={fees}

Please fix the code and return the COMPLETE corrected version.
Common fixes:
- Use .squeeze() on yfinance columns to get 1D Series
- Use .fillna(False) on boolean signals
- Pass .values (numpy arrays) to vectorbt.Portfolio.from_signals()
- Make sure the `results` dict has ALL required keys

Return ONLY the ````json```` block with the IR, followed by the ````python```` block with the fixed code."""

    raw = call_ai(
        prompt=retry_prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,
        max_output_tokens=4096,
    )

    ir_dict, code = _parse_ai_response(raw)
    
    if not code:
        raise AIError("AI did not return any Python code on execution retry.")

    safety_issues = _validate_code_safety(code)
    if safety_issues:
        raise AIError(
            f"Retry code has safety issues: {', '.join(safety_issues)}"
        )

    return ir_dict, code
