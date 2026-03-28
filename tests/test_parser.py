import sys
import os
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_strategy, IndicatorType, Condition

def test_parse_easy():
    """Test a basic SMA crossover strategy."""
    strategy = "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200"
    parsed = parse_strategy(strategy)
    
    assert parsed.is_valid is True
    assert len(parsed.entry_rules) == 1
    assert len(parsed.exit_rules) == 1
    
    # Check entry rule
    assert parsed.entry_rules[0].left.type == IndicatorType.SMA
    assert parsed.entry_rules[0].left.params["period"] == 50
    assert parsed.entry_rules[0].condition == Condition.CROSSES_ABOVE
    assert parsed.entry_rules[0].right.params["period"] == 200
    
    # Check exit rule
    assert parsed.exit_rules[0].condition == Condition.CROSSES_BELOW

def test_parse_medium():
    """Test multi-indicator strategy with explicit keywords."""
    strategy = "Buy when RSI is below 30, buy when MACD crosses above MACD signal"
    parsed = parse_strategy(strategy)
    
    assert parsed.is_valid is True
    assert len(parsed.entry_rules) == 2
    
    # Needs to handle value 30
    assert parsed.entry_rules[0].left.type == IndicatorType.RSI
    assert parsed.entry_rules[0].condition == Condition.IS_BELOW
    assert parsed.entry_rules[0].right.type == IndicatorType.VALUE
    assert parsed.entry_rules[0].right.params["value"] == 30.0
    
    assert parsed.entry_rules[1].left.type == IndicatorType.MACD_LINE
    assert parsed.entry_rules[1].condition == Condition.CROSSES_ABOVE

def test_parse_hard_auto_exit():
    """Test a complex strategy missing an exit, forcing auto-generation."""
    strategy = "Go long when price crosses above upper Bollinger Band"
    parsed = parse_strategy(strategy)
    
    assert parsed.is_valid is True
    assert len(parsed.entry_rules) == 1
    assert len(parsed.exit_rules) == 1  # Should be auto-generated
    
    assert parsed.entry_rules[0].left.type == IndicatorType.PRICE
    assert parsed.entry_rules[0].condition == Condition.CROSSES_ABOVE
    assert parsed.entry_rules[0].right.type == IndicatorType.BB_UPPER
    
    # Auto-generated exit should inverse the condition
    assert parsed.exit_rules[0].condition == Condition.CROSSES_BELOW

def test_parse_extreme_contradiction():
    """Test contradictory logic which should invalidate the parsed rule."""
    strategy = "Buy when RSI goes above 70 and sell when RSI goes above 70"
    parsed = parse_strategy(strategy)
    
    # The new validation logic should see er == ex and clear the rules
    assert parsed.is_valid is False
    assert len(parsed.entry_rules) == 0
    assert len(parsed.exit_rules) == 0
    assert any("Contradictory" in w for w in parsed.warnings)

def test_parse_extreme_nonsense():
    """Test gibberish strings that should fail parsing cleanly without crashing."""
    strategy = "Buy when to the moon and sell if red"
    parsed = parse_strategy(strategy)
    
    assert parsed.is_valid is False
    assert len(parsed.entry_rules) == 0
    assert len(parsed.warnings) > 0
    assert "Could not parse clause" in parsed.warnings[0]
