"""Extension 3: Economics of Prompt Caching & Break-Even Analysis.

Analyzes token_usage.csv to determine the empirical cache hit rate,
calculates break-even read count against write surcharge, and validates
whether caching is net-positive for NimbusAI's workloads.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops.pricing import cache_is_worth_it
from missions.m2_inference_levers import MODEL_PRICES


def analyze_cache_economics():
    rows = load_csv("token_usage.csv")
    total_requests = len(rows)
    cache_eligible_requests = 0
    total_input = 0
    total_cached = 0

    for r in rows:
        inp = int(num(r["input_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        total_input += inp
        total_cached += cached
        if cached > 0:
            cache_eligible_requests += 1

    overall_cache_frac = total_cached / total_input if total_input else 0.0
    cache_hit_rate = cache_eligible_requests / total_requests if total_requests else 0.0
    
    # In real systems (e.g. Anthropic/Gemini), cache write costs ~1.25x base input price
    # Read discount is 90% off (10% of base price).
    # Break-even = 1.25 / (1.0 - 0.10) = 1.388 reads.
    write_price_large = 3.75
    read_price_large = 3.00 * 0.10
    savings_per_read_large = 3.00 - read_price_large
    break_even_large = write_price_large / savings_per_read_large

    print("============================================================")
    print("  EXTENSION 3: PROMPT CACHING ECONOMICS")
    print("============================================================")
    print(f"Total Requests Analyzed: {total_requests:,}")
    print(f"Requests with Cache Hits: {cache_eligible_requests:,} ({cache_hit_rate:.1%})")
    print(f"Total Input Tokens: {total_input:,}")
    print(f"Total Cached Input Tokens: {total_cached:,} ({overall_cache_frac:.1%} of input)")
    print(f"Break-even Read Multiplier: {break_even_large:.2f} reads")
    print(f"Is caching worth it with 3 reads avg? -> {cache_is_worth_it(3.0, write_price_large, 0.10, 3.00)}")
    print(f"Is caching worth it with 1 read avg?  -> {cache_is_worth_it(1.0, write_price_large, 0.10, 3.00)}")
    print("============================================================\n")


if __name__ == "__main__":
    analyze_cache_economics()
