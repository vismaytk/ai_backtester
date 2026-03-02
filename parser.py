"""
═══════════════════════════════════════════════════════════════════════════════
  Natural Language Strategy Parser
  ─────────────────────────────────────────────────────────────────────────
  Parses plain-English trading strategies into structured objects.

  Supports:
    • Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Price
    • Conditions: crosses above/below, is above/below
    • Actions: buy, sell, go long, go short

  Example:
    "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200"
    →  ParsedStrategy(
           entry = Rule(left=SMA(50), condition=CROSS_ABOVE, right=SMA(200)),
           exit  = Rule(left=SMA(50), condition=CROSS_BELOW, right=SMA(200)),
       )
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

class IndicatorType(Enum):
    SMA = "SMA"
    EMA = "EMA"
    RSI = "RSI"
    MACD_LINE = "MACD_LINE"
    MACD_SIGNAL = "MACD_SIGNAL"
    MACD_HISTOGRAM = "MACD_HISTOGRAM"
    BB_UPPER = "BB_UPPER"
    BB_LOWER = "BB_LOWER"
    BB_MIDDLE = "BB_MIDDLE"
    PRICE = "PRICE"
    VALUE = "VALUE"  # A numeric constant


class Condition(Enum):
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    IS_ABOVE = "is_above"
    IS_BELOW = "is_below"


class Action(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Indicator:
    type: IndicatorType
    params: dict = field(default_factory=dict)

    def __repr__(self):
        if self.type == IndicatorType.VALUE:
            return str(self.params.get("value", "?"))
        elif self.type == IndicatorType.PRICE:
            return "Price"
        elif self.type == IndicatorType.SMA:
            return f"SMA({self.params.get('period', '?')})"
        elif self.type == IndicatorType.EMA:
            return f"EMA({self.params.get('period', '?')})"
        elif self.type == IndicatorType.RSI:
            return f"RSI({self.params.get('period', 14)})"
        elif self.type == IndicatorType.MACD_LINE:
            return f"MACD_Line({self.params.get('fast', 12)},{self.params.get('slow', 26)})"
        elif self.type == IndicatorType.MACD_SIGNAL:
            return f"MACD_Signal({self.params.get('signal', 9)})"
        elif self.type == IndicatorType.BB_UPPER:
            return f"BB_Upper({self.params.get('period', 20)},{self.params.get('std', 2)})"
        elif self.type == IndicatorType.BB_LOWER:
            return f"BB_Lower({self.params.get('period', 20)},{self.params.get('std', 2)})"
        elif self.type == IndicatorType.BB_MIDDLE:
            return f"BB_Middle({self.params.get('period', 20)})"
        return f"{self.type.value}({self.params})"


@dataclass
class Rule:
    left: Indicator
    condition: Condition
    right: Indicator

    def __repr__(self):
        return f"{self.left} {self.condition.value} {self.right}"


@dataclass
class ParsedStrategy:
    entry_rules: List[Rule]
    exit_rules: List[Rule]
    raw_input: str = ""
    parse_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self):
        return len(self.entry_rules) > 0

    def summary(self):
        lines = []
        for i, rule in enumerate(self.entry_rules):
            lines.append(f"  Entry {i+1}: {rule}")
        for i, rule in enumerate(self.exit_rules):
            lines.append(f"  Exit  {i+1}: {rule}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & PARSER
# ══════════════════════════════════════════════════════════════════════════════

# ── Indicator Patterns ────────────────────────────────────────────────────────

INDICATOR_PATTERNS = [
    # SMA
    (r'\bsma\s*(\d+)\b', lambda m: Indicator(IndicatorType.SMA, {"period": int(m.group(1))})),
    (r'\b(\d+)\s*[-]?\s*(?:day|period)?\s*(?:simple\s*)?moving\s*average\b',
     lambda m: Indicator(IndicatorType.SMA, {"period": int(m.group(1))})),
    (r'\bsimple\s*moving\s*average\s*(?:of\s*)?(\d+)\b',
     lambda m: Indicator(IndicatorType.SMA, {"period": int(m.group(1))})),

    # EMA
    (r'\bema\s*(\d+)\b', lambda m: Indicator(IndicatorType.EMA, {"period": int(m.group(1))})),
    (r'\b(\d+)\s*[-]?\s*(?:day|period)?\s*exponential\s*moving\s*average\b',
     lambda m: Indicator(IndicatorType.EMA, {"period": int(m.group(1))})),
    (r'\bexponential\s*moving\s*average\s*(?:of\s*)?(\d+)\b',
     lambda m: Indicator(IndicatorType.EMA, {"period": int(m.group(1))})),

    # RSI with period
    (r'\brsi\s*[\(]?\s*(\d+)\s*[\)]?\b',
     lambda m: Indicator(IndicatorType.RSI, {"period": int(m.group(1))})),
    (r'\brelative\s*strength\s*index\s*(?:of\s*)?(\d+)?\b',
     lambda m: Indicator(IndicatorType.RSI, {"period": int(m.group(1)) if m.group(1) else 14})),
    # RSI without period (default 14)
    (r'\brsi\b', lambda m: Indicator(IndicatorType.RSI, {"period": 14})),

    # MACD
    (r'\bmacd\s*signal\b',
     lambda m: Indicator(IndicatorType.MACD_SIGNAL, {"fast": 12, "slow": 26, "signal": 9})),
    (r'\bmacd\s*histogram\b',
     lambda m: Indicator(IndicatorType.MACD_HISTOGRAM, {"fast": 12, "slow": 26, "signal": 9})),
    (r'\bmacd\s*line\b',
     lambda m: Indicator(IndicatorType.MACD_LINE, {"fast": 12, "slow": 26, "signal": 9})),
    (r'\bmacd\b',
     lambda m: Indicator(IndicatorType.MACD_LINE, {"fast": 12, "slow": 26, "signal": 9})),

    # Bollinger Bands
    (r'\bupper\s*(?:bollinger\s*)?band\b',
     lambda m: Indicator(IndicatorType.BB_UPPER, {"period": 20, "std": 2})),
    (r'\bollinger\s*upper\b',
     lambda m: Indicator(IndicatorType.BB_UPPER, {"period": 20, "std": 2})),
    (r'\blower\s*(?:bollinger\s*)?band\b',
     lambda m: Indicator(IndicatorType.BB_LOWER, {"period": 20, "std": 2})),
    (r'\bollinger\s*lower\b',
     lambda m: Indicator(IndicatorType.BB_LOWER, {"period": 20, "std": 2})),
    (r'\bmiddle\s*(?:bollinger\s*)?band\b',
     lambda m: Indicator(IndicatorType.BB_MIDDLE, {"period": 20})),

    # Price
    (r'\b(?:close|closing\s*price|price|close\s*price)\b',
     lambda m: Indicator(IndicatorType.PRICE)),
]

# ── Condition Patterns ────────────────────────────────────────────────────────

CONDITION_PATTERNS = [
    (r'\bcross(?:es)?\s*above\b', Condition.CROSSES_ABOVE),
    (r'\bbreak(?:s)?\s*above\b', Condition.CROSSES_ABOVE),
    (r'\bgoes?\s*above\b', Condition.CROSSES_ABOVE),
    (r'\bmoves?\s*above\b', Condition.CROSSES_ABOVE),
    (r'\bcross(?:es)?\s*over\b', Condition.CROSSES_ABOVE),

    (r'\bcross(?:es)?\s*below\b', Condition.CROSSES_BELOW),
    (r'\bbreak(?:s)?\s*below\b', Condition.CROSSES_BELOW),
    (r'\bgoes?\s*below\b', Condition.CROSSES_BELOW),
    (r'\bdrops?\s*below\b', Condition.CROSSES_BELOW),
    (r'\bfalls?\s*below\b', Condition.CROSSES_BELOW),
    (r'\bcross(?:es)?\s*under\b', Condition.CROSSES_BELOW),

    (r'\bis\s*above\b', Condition.IS_ABOVE),
    (r'\babove\b', Condition.IS_ABOVE),
    (r'\bgreater\s*than\b', Condition.IS_ABOVE),
    (r'\bover\b', Condition.IS_ABOVE),
    (r'\bhigher\s*than\b', Condition.IS_ABOVE),

    (r'\bis\s*below\b', Condition.IS_BELOW),
    (r'\bbelow\b', Condition.IS_BELOW),
    (r'\bless\s*than\b', Condition.IS_BELOW),
    (r'\bunder\b', Condition.IS_BELOW),
    (r'\blower\s*than\b', Condition.IS_BELOW),
]

# ── Action Patterns ───────────────────────────────────────────────────────────

BUY_PATTERNS = [
    r'\bbuy\b', r'\bgo\s*long\b', r'\benter\s*long\b',
    r'\blong\b', r'\benter\b', r'\bopen\b',
]

SELL_PATTERNS = [
    r'\bsell\b', r'\bgo\s*short\b', r'\bexit\b',
    r'\bclose\s*position\b', r'\bshort\b',
]


def _find_indicators_in_text(text: str) -> List[Tuple[int, int, Indicator]]:
    """Find all indicator mentions in text with their positions."""
    found = []
    text_lower = text.lower()

    for pattern, factory in INDICATOR_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            indicator = factory(match)
            found.append((match.start(), match.end(), indicator))

    # Remove overlapping matches (keep longest)
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, ind in found:
        if start >= last_end:
            filtered.append((start, end, ind))
            last_end = end

    return filtered


def _find_condition(text: str) -> Optional[Tuple[int, int, Condition]]:
    """Find the first condition in text."""
    text_lower = text.lower()
    best = None

    for pattern, condition in CONDITION_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if best is None or match.start() < best[0]:
                best = (match.start(), match.end(), condition)

    return best


def _find_numeric_values(text: str) -> List[Tuple[int, int, float]]:
    """Find standalone numeric values (not part of indicator params)."""
    found = []
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
        found.append((match.start(), match.end(), float(match.group(1))))
    return found


def _detect_action(text: str) -> Optional[Action]:
    """Detect whether this clause is a buy or sell action."""
    text_lower = text.lower()

    for pattern in BUY_PATTERNS:
        if re.search(pattern, text_lower):
            return Action.BUY

    for pattern in SELL_PATTERNS:
        if re.search(pattern, text_lower):
            return Action.SELL

    return None


def _parse_clause(clause: str) -> Optional[Tuple[Action, Rule]]:
    """Parse a single clause like 'buy when SMA 50 crosses above SMA 200'."""
    action = _detect_action(clause)
    if action is None:
        return None

    # Remove the action word and "when" for cleaner parsing
    clause_clean = re.sub(r'\b(?:buy|sell|go\s*long|go\s*short|enter|exit)\b', '', clause, flags=re.IGNORECASE)
    clause_clean = re.sub(r'\bwhen\b', '', clause_clean, flags=re.IGNORECASE).strip()

    # Find condition
    cond_match = _find_condition(clause_clean)
    if cond_match is None:
        return None

    cond_start, cond_end, condition = cond_match

    # Split text around condition
    left_text = clause_clean[:cond_start].strip()
    right_text = clause_clean[cond_end:].strip()

    # Find indicators in left and right parts
    left_indicators = _find_indicators_in_text(left_text)
    right_indicators = _find_indicators_in_text(right_text)

    # Determine left operand
    if left_indicators:
        left_ind = left_indicators[0][2]
    else:
        left_ind = Indicator(IndicatorType.PRICE)

    # Determine right operand
    if right_indicators:
        right_ind = right_indicators[0][2]
    else:
        # Check for numeric value
        nums = _find_numeric_values(right_text)
        if nums:
            right_ind = Indicator(IndicatorType.VALUE, {"value": nums[0][2]})
        else:
            return None

    rule = Rule(left=left_ind, condition=condition, right=right_ind)
    return (action, rule)


def parse_strategy(user_input: str) -> ParsedStrategy:
    """
    Parse a natural language strategy description into a structured ParsedStrategy.

    Examples:
        "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200"
        "Buy when RSI drops below 30, sell when RSI goes above 70"
        "Go long when price crosses above upper Bollinger Band"
    """
    entry_rules = []
    exit_rules = []
    warnings = []

    # Split on common delimiters
    clauses = re.split(r'[,;]\s*|\band\s+(?=sell|buy|exit|close)', user_input, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if c.strip()]

    for clause in clauses:
        result = _parse_clause(clause)
        if result:
            action, rule = result
            if action == Action.BUY:
                entry_rules.append(rule)
            else:
                exit_rules.append(rule)
        else:
            warnings.append(f"Could not parse clause: '{clause}'")

    # If we only have entry rules and no exit, try to infer exit
    if entry_rules and not exit_rules:
        for rule in entry_rules:
            # Invert the condition for exit
            if rule.condition == Condition.CROSSES_ABOVE:
                exit_cond = Condition.CROSSES_BELOW
            elif rule.condition == Condition.CROSSES_BELOW:
                exit_cond = Condition.CROSSES_ABOVE
            elif rule.condition == Condition.IS_ABOVE:
                exit_cond = Condition.IS_BELOW
            elif rule.condition == Condition.IS_BELOW:
                exit_cond = Condition.IS_ABOVE
            else:
                exit_cond = Condition.CROSSES_BELOW

            exit_rules.append(Rule(left=rule.left, condition=exit_cond, right=rule.right))
            warnings.append(f"Auto-generated exit rule (inverse of entry): {exit_rules[-1]}")

    # Calculate confidence
    confidence = 0.0
    if entry_rules:
        confidence += 0.5
    if exit_rules:
        confidence += 0.3
    if not warnings or all("Auto-generated" in w for w in warnings):
        confidence += 0.2

    return ParsedStrategy(
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        raw_input=user_input,
        parse_confidence=confidence,
        warnings=warnings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_strategies = [
        "Buy when SMA 50 crosses above SMA 200, sell when SMA 50 crosses below SMA 200",
        "Buy when RSI drops below 30, sell when RSI goes above 70",
        "Go long when price crosses above upper Bollinger Band",
        "Buy when EMA 20 crosses above EMA 50, sell when EMA 20 crosses below EMA 50",
        "Buy when MACD line crosses above MACD signal, sell when MACD line crosses below MACD signal",
        "Buy when price goes above SMA 200",
    ]

    for strat in test_strategies:
        print(f"\n{'─' * 60}")
        print(f"  Input: {strat}")
        parsed = parse_strategy(strat)
        print(f"  Valid: {parsed.is_valid} | Confidence: {parsed.parse_confidence:.0%}")
        print(parsed.summary())
        if parsed.warnings:
            for w in parsed.warnings:
                print(f"  ⚠️  {w}")
