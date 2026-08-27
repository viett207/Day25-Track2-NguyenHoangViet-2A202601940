"""Extension 4: Reasoning Traffic Budget & Energy Multiplier Analysis.

Separates cost and energy consumption (Wh / carbon) for is_reasoning=1 vs is_reasoning=0,
and proposes an optimal routing policy with budget caps.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability
from missions.m2_inference_levers import MODEL_PRICES


def analyze_reasoning_budget():
    rows = load_csv("token_usage.csv")
    stats = {
        "standard": {"count": 0, "inp": 0, "out": 0, "cost": 0.0, "wh": 0.0},
        "reasoning": {"count": 0, "inp": 0, "out": 0, "cost": 0.0, "wh": 0.0},
    }

    for r in rows:
        inp = int(num(r["input_tokens"]))
        out = int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)
        
        key = "reasoning" if is_reasoning else "standard"
        stats[key]["count"] += 1
        stats[key]["inp"] += inp
        stats[key]["out"] += out
        stats[key]["cost"] += cost
        stats[key]["wh"] += wh

    total_req = len(rows)
    total_cost = stats["standard"]["cost"] + stats["reasoning"]["cost"]
    total_wh = stats["standard"]["wh"] + stats["reasoning"]["wh"]

    print("============================================================")
    print("  EXTENSION 4: REASONING TRAFFIC & ENERGY BUDGET ANALYSIS")
    print("============================================================")
    print(f"Total Requests: {total_req:,}")
    print(f"  - Standard queries : {stats['standard']['count']:,} ({stats['standard']['count']/total_req:.1%})")
    print(f"  - Reasoning queries: {stats['reasoning']['count']:,} ({stats['reasoning']['count']/total_req:.1%})")
    print("\nFinancial Impact ($ / day):")
    print(f"  - Standard cost : ${stats['standard']['cost']:.2f} ({stats['standard']['cost']/total_cost:.1%})")
    print(f"  - Reasoning cost: ${stats['reasoning']['cost']:.2f} ({stats['reasoning']['cost']/total_cost:.1%})")
    print("\nEnergy Consumption (Wh / day):")
    print(f"  - Standard energy : {stats['standard']['wh']:,.1f} Wh ({stats['standard']['wh']/total_wh:.1%})")
    print(f"  - Reasoning energy: {stats['reasoning']['wh']:,.1f} Wh ({stats['reasoning']['wh']/total_wh:.1%})")
    print(f"\nEnergy per query:")
    print(f"  - Standard avg : {stats['standard']['wh']/stats['standard']['count']:.2f} Wh/query")
    print(f"  - Reasoning avg: {stats['reasoning']['wh']/stats['reasoning']['count']:.2f} Wh/query (Multiplier: ~{(stats['reasoning']['wh']/stats['reasoning']['count'])/(stats['standard']['wh']/stats['standard']['count']):.1f}x)")
    print("============================================================\n")


if __name__ == "__main__":
    analyze_reasoning_budget()
