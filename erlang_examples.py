"""Example workflows using the Erlang calculator utilities."""

import numpy as np
from erlang_calculator import X, CHAT, BL


def run_basic_workflow() -> None:
    """Run a basic Erlang calculation workflow and print the results."""
    traffic = 6
    agents = 8
    aht = 180
    target = 20

    b = X.erlang_b(traffic, agents)
    c = X.erlang_c(traffic, agents)
    sl = CHAT.service_level(traffic, agents, aht, target)
    asa = CHAT.asa(traffic, agents, aht)

    print("=== Basic Workflow ===")
    print(f"Erlang B Blocking: {b:.4f}")
    print(f"Erlang C Waiting: {c:.4f}")
    print(f"Service Level: {sl:.4f}")
    print(f"ASA: {asa:.2f} seconds")


def run_sensitivity_example() -> None:
    """Run a simple sensitivity analysis and output the resulting DataFrame."""
    traffic_range = np.linspace(4, 8, 5)
    df = BL.sensitivity(traffic_range, agents=8, aht=180, target=20)
    print("=== Sensitivity Analysis ===")
    print(df)


def run_monte_carlo_example() -> None:
    """Run Monte Carlo simulation and print summary statistics."""
    series = BL.monte_carlo(mean_traffic=6, agents=8, aht=180, target=20, iters=200)
    print("=== Monte Carlo Simulation ===")
    print(series.describe())


if __name__ == "__main__":
    run_basic_workflow()
    run_sensitivity_example()
    run_monte_carlo_example()
