"""Extension 5: Carbon-Aware Scheduling for Interruptible Training Workloads.

Evaluates carbon emission (gCO2e) and electricity costs across 5 global regions
for all interruptible workloads in workloads.csv.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30


def analyze_carbon_scheduling():
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    # Calculate total GPU energy (kWh) consumed by interruptible jobs in a month
    interruptible_jobs = [j for j in jobs if bool(int(num(j["interruptible"])))]
    
    total_gpu_kwh_month = 0.0
    for j in interruptible_jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        watts = num(cat[gtype]["watts"])
        # Energy (kWh) = (Watts * hours) / 1000
        gpu_kwh = (watts * ngpu * hpd * DAYS) / 1000.0
        total_gpu_kwh_month += gpu_kwh

    print("============================================================")
    print("  EXTENSION 5: CARBON-AWARE SCHEDULING ACROSS REGIONS")
    print("============================================================")
    print(f"Total Interruptible Training Workloads: {len(interruptible_jobs)} jobs")
    print(f"Total Monthly Electricity Required: {total_gpu_kwh_month:,.1f} kWh\n")
    print(f"{'Region':18}{'Grid Carbon (g/kWh)':>22}{'Elec Price ($/kWh)':>20}{'Carbon (kgCO2e)':>18}{'Power Cost ($)':>16}")
    print("-" * 96)

    baseline_carbon = 0.0
    cleanest_carbon = 0.0
    
    for region, carbon_rate in sustainability.REGION_CARBON.items():
        price_rate = sustainability.REGION_PRICE_KWH.get(region, 0.12)
        carbon_kg = (total_gpu_kwh_month * carbon_rate) / 1000.0
        power_cost = total_gpu_kwh_month * price_rate
        
        if region == "us-east-1":
            baseline_carbon = carbon_kg
        if region == "europe-north1":
            cleanest_carbon = carbon_kg

        print(f"{region:18}{carbon_rate:>22}{price_rate:>20.3f}{carbon_kg:>18,.1f}${power_cost:>15,.2f}")

    carbon_saved_kg = baseline_carbon - cleanest_carbon
    pct_reduction = (carbon_saved_kg / baseline_carbon) * 100.0 if baseline_carbon else 0.0

    print("-" * 96)
    print(f"Relocating interruptible workloads from us-east-1 to europe-north1 saves:")
    print(f"  -> {carbon_saved_kg:,.1f} kg CO2e / month ({pct_reduction:.1f}% reduction in emissions!)")
    print("============================================================\n")


if __name__ == "__main__":
    analyze_carbon_scheduling()
