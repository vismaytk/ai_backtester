import sys
import os
import random
import traceback
import string

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_strategy
from code_generator import generate_backtest_code

# Some base vocab for generating reasonable nonsense
INDICATORS = ["SMA 50", "RSI 30", "price", "lower Bollinger Band", "MACD line", "MACD signal", "EMA 200"]
CONDITIONS = ["crosses above", "goes below", "is higher than", "drops under"]
ACTIONS = ["buy", "sell", "go long", "close position"]
CONJUNCTIONS = ["when", "if", "and then", "but only after", "or"]

def generate_random_strategy():
    """Generates a random pseudo-English trading strategy string."""
    length = random.randint(1, 4)
    words = []
    
    for _ in range(length):
        if random.random() < 0.2:
            # Pure random string
            words.append(''.join(random.choices(string.ascii_letters + " ", k=10)))
        else:
            # Pseudo-strategy phrase
            words.append(random.choice(ACTIONS))
            words.append(random.choice(CONJUNCTIONS))
            words.append(random.choice(INDICATORS))
            words.append(random.choice(CONDITIONS))
            words.append(random.choice(INDICATORS))
            
    return " ".join(words)

def run_fuzz_test(iterations=500):
    """
    Run the parser and code generator against random strings to ensure
    there are no unhandled crashes (e.g. IndexError on an empty rule list).
    """
    print(f"[*] Starting {iterations} iterations of Fuzz Testing on rule-based parser...")
    
    crashes = 0
    valid_parses = 0
    logic_errors = 0
    
    for i in range(iterations):
        strategy = generate_random_strategy()
        
        try:
            # 1. Parse Strategy
            parsed = parse_strategy(strategy)
            
            # If valid, 2. Generate Code
            if parsed.is_valid:
                valid_parses += 1
                _ = generate_backtest_code(parsed)
            else:
                logic_errors += 1
                
        except Exception as e:
            crashes += 1
            print(f"\n[!] CRASH DETECTED ON ITERATION {i}")
            print(f"Strategy String: {strategy}")
            print(f"Exception Dump:\n{traceback.format_exc()}")
            
    print("\n--- Fuzzing Results ---")
    print(f"Total Iterations: {iterations}")
    print(f"Valid Parsed Strategies (No Crash): {valid_parses}")
    print(f"Invalid Logic Caught (No Crash): {logic_errors}")
    print(f"Unhandled Crashes (Failures): {crashes}")
    
    if crashes > 0:
        print("\n❌ Fuzz test failed!")
        sys.exit(1)
    else:
        print("\n✅ Fuzz test passed successfully! System is robust.")
        
if __name__ == "__main__":
    run_fuzz_test(500)
