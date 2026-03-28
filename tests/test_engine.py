import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_generator import generate_backtest_code
from backtester import run_backtest, BacktestError
from parser import parse_strategy

def test_engine_zero_trades():
    """Test what happens if the strategy logic generates exactly 0 signals."""
    strategy = "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200"
    parsed = parse_strategy(strategy)
    
    # We will pass a very short timeframe where crossover cannot possibly happen
    code = generate_backtest_code(
        strategy=parsed,
        ticker="Reliance.ns",
        start_date="2023-01-01",
        end_date="2023-01-15",
        init_cash=1000000,
        fees=0.001
    )
    
    # Run backtest
    results = run_backtest(code)
    
    # Ensure variables handled 0 trade gracefully
    assert results["total_trades"] == 0
    assert results["win_rate"] == 0.0
    assert results["profit_factor"] == 0.0
    assert isinstance(results["trades_readable"], pd.DataFrame)
    
def test_engine_determinism():
    """Ensure running the same code twice yields identical results."""
    strategy = "Buy when SMA 20 crosses above SMA 50, sell when SMA 20 crosses below SMA 50"
    parsed = parse_strategy(strategy)
    
    code = generate_backtest_code(
        strategy=parsed,
        ticker="^NSEI",
        start_date="2022-01-01",
        end_date="2023-01-01",
        init_cash=100_000,
        fees=0.0
    )
    
    results1 = run_backtest(code)
    results2 = run_backtest(code)
    
    assert np.isclose(results1["total_return"], results2["total_return"])
    assert results1["total_trades"] == results2["total_trades"]

def test_engine_missing_results():
    """Test isolation to ensure an exec() without 'results' throws properly."""
    code = "import yfinance as yf\nclose = [1,2,3]\ntotal_return = 5"
    with pytest.raises(BacktestError, match="did not produce a 'results' dictionary"):
        run_backtest(code)

def test_engine_syntax_error():
    """Test isolation handles syntax errors gracefully via BacktestError."""
    code = "import yfinance\nthis is not valid python code\n"
    with pytest.raises(BacktestError, match="execution failed"):
        run_backtest(code)
