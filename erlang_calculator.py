# coding: utf-8

"""Erlang calculator with Streamlit UI.

This module exposes a minimal set of Erlang based formulas that can be used in
scripts or through the included Streamlit application.  Additional convenience
classes are provided for common call centre KPIs such as service level or
occupancy.

Examples
--------
>>> from erlang_calculator import SLA, ASA, BLOCKING
>>> SLA.probability(traffic=6, agents=8, aht=180, target=20)
0.82  # approximate service level

>>> ASA.wait_time(traffic=6, agents=8, aht=180)
23.4  # average speed of answer in seconds

The :func:`run_app` function starts a Streamlit interface for interactive use.
"""

import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.optimize import minimize


class BLOCKING:
    """Blocking related calculations."""

    @staticmethod
    def probability(traffic: float, trunks: int) -> float:
        """Return Erlang B blocking probability.

        Parameters
        ----------
        traffic : float
            Offered traffic intensity in erlangs.
        trunks : int
            Number of available trunks/lines.

        Examples
        --------
        >>> BLOCKING.probability(traffic=4, trunks=5)
        0.10
        """

        traffic = float(traffic)
        trunks = int(trunks)
        if trunks <= 0:
            raise ValueError("trunks must be positive")
        inv_b = np.sum([(traffic ** k) / factorial(k) for k in range(trunks + 1)])
        b = ((traffic ** trunks) / factorial(trunks)) / inv_b
        return float(b)


class OCCUPANCY:
    """Agent occupancy calculations.

    Examples
    --------
    >>> OCCUPANCY.rate(traffic=5, agents=8)
    0.625
    """

    @staticmethod
    def rate(traffic: float, agents: int) -> float:
        """Return occupancy ratio given offered traffic and agents."""

        traffic = float(traffic)
        agents = int(agents)
        if agents <= 0:
            raise ValueError("agents must be positive")
        occ = traffic / agents
        return min(max(occ, 0.0), 1.0)


class SLA:
    """Service level calculations.

    Examples
    --------
    >>> SLA.probability(traffic=6, agents=8, aht=180, target=20)
    0.82
    """

    @staticmethod
    def probability(traffic: float, agents: int, aht: float, target: float) -> float:
        """Return service level probability."""

        if agents <= 0:
            raise ValueError("agents must be positive")
        if aht <= 0 or target < 0:
            raise ValueError("aht and target must be positive")
        c = X.erlang_c(traffic, agents)
        return 1 - c * np.exp(-(agents - traffic) * target / aht)


class ASA:
    """Average speed of answer related utilities.

    Examples
    --------
    >>> ASA.wait_time(traffic=6, agents=8, aht=180)
    23.4
    """

    @staticmethod
    def wait_time(traffic: float, agents: int, aht: float) -> float:
        """Return expected waiting time in seconds."""

        if agents <= 0:
            raise ValueError("agents must be positive")
        if aht <= 0:
            raise ValueError("aht must be positive")
        c = X.erlang_c(traffic, agents)
        if agents <= traffic:
            return float('inf')
        return (c / (agents - traffic)) * aht


class ABANDON:
    """Call abandonment calculations.

    Examples
    --------
    >>> ABANDON.probability(traffic=6, agents=8, aht=180, patience=30)
    0.45
    """

    @staticmethod
    def probability(
        traffic: float,
        agents: int,
        aht: float,
        patience: float = 30.0,
    ) -> float:
        """Estimate abandonment probability assuming exponential patience."""

        if patience <= 0:
            raise ValueError("patience must be positive")
        wait = ASA.wait_time(traffic, agents, aht)
        if np.isinf(wait):
            return 1.0
        return 1 - np.exp(-wait / patience)

class X:
    """Collection of Erlang formulas."""

    @staticmethod
    def erlang_b(traffic, trunks):
        """Compute Erlang B blocking probability.

        Parameters
        ----------
        traffic : float
            Offered traffic in erlangs.
        trunks : int
            Number of available trunks.
        """
        traffic = float(traffic)
        trunks = int(trunks)
        inv_b = np.sum([(traffic ** k) / factorial(k) for k in range(trunks + 1)])
        b = ((traffic ** trunks) / factorial(trunks)) / inv_b
        return b

    @staticmethod
    def erlang_c(traffic, agents):
        """Compute Erlang C waiting probability.

        Parameters
        ----------
        traffic : float
            Offered traffic in erlangs.
        agents : int
            Number of available agents.
        """
        traffic = float(traffic)
        agents = int(agents)
        rho = traffic / agents
        if rho >= 1:
            return 1.0
        numerator = ((traffic ** agents) / factorial(agents)) * (agents / (agents - traffic))
        denom = np.sum([(traffic ** k) / factorial(k) for k in range(agents)]) + numerator
        return numerator / denom

    class AGENTS:
        """Staffing utilities for Erlang B/C models."""

        @staticmethod
        def required_for_blocking(traffic: float, target_blocking: float) -> int:
            """Minimum trunks so blocking <= ``target_blocking``."""

            if traffic < 0:
                raise ValueError("traffic must be non-negative")
            if target_blocking <= 0:
                return int(np.ceil(traffic))

            if target_blocking >= 1:
                raise ValueError("target_blocking must be < 1")

            trunks = max(int(np.ceil(traffic)), 1)
            while BLOCKING.probability(traffic, trunks) > target_blocking:
                trunks += 1
            return trunks

class CHAT:
    """Call center metrics."""

    @staticmethod
    def service_level(traffic, agents, aht, target):
        """Return probability a call is answered within ``target`` seconds.

        Parameters
        ----------
        traffic : float
            Offered traffic in erlangs.
        agents : int
            Number of available agents.
        aht : float
            Average handle time (seconds).
        target : float
            Service level threshold in seconds.
        """

        c = X.erlang_c(traffic, agents)
        return 1 - c * np.exp(-(agents - traffic) * target / aht)

    @staticmethod
    def asa(traffic, agents, aht):
        """Return the average waiting time before answer in seconds."""

        c = X.erlang_c(traffic, agents)
        if agents <= traffic:
            return np.inf
        return (c / (agents - traffic)) * aht

    class AGENTS:
        """Agent requirement utilities for call centre KPIs."""

        @staticmethod
        def for_service_level(
            traffic: float, aht: float, target_service: float, target_time: float = 20
        ) -> int:
            """Wrapper around :func:`required_agents`."""

            return CHAT.required_agents(traffic, aht, target_service, target_time)

    @staticmethod
    def required_agents(traffic, aht, target_service, target_time=20):
        """Find minimum agents to hit target service level.

        Parameters
        ----------
        traffic : float
            Offered traffic in erlangs.
        aht : float
            Average handle time (seconds).
        target_service : float
            Desired service level (0-1).
        target_time : float, optional
            Threshold time in seconds, by default ``20``.
        """
        def objective(n):
            return abs(CHAT.service_level(traffic, int(n[0]), aht, target_time) - target_service)

        res = minimize(objective, x0=[traffic], bounds=[(traffic, traffic * 10)])
        return int(np.ceil(res.x[0]))

class BL:
    """Advanced analysis tools."""

    @staticmethod
    def sensitivity(traffic_range, agents, aht, target):
        rows = []
        for t in traffic_range:
            sl = CHAT.service_level(t, agents, aht, target)
            rows.append({'traffic': t, 'service_level': sl})
        return pd.DataFrame(rows)

    @staticmethod
    def monte_carlo(mean_traffic, agents, aht, target, iters=1000):
        samples = np.random.poisson(mean_traffic, size=iters)
        results = [CHAT.service_level(s, agents, aht, target) for s in samples]
        return pd.Series(results)

    class AGENTS:
        """Utility functions for staffing based on occupancy."""

        @staticmethod
        def for_occupancy(traffic: float, max_occupancy: float) -> int:
            """Return agents required so occupancy <= ``max_occupancy``."""

            if not 0 < max_occupancy <= 1:
                raise ValueError("max_occupancy must be in (0, 1]")
            return int(np.ceil(traffic / max_occupancy))



def run_app():
    """Launch Streamlit UI with multi-chat and analysis tools."""
    import streamlit as st
    st.title("Erlang Calculator")
    st.sidebar.header("Inputs")
    traffic = st.sidebar.number_input("Traffic Intensity (erlangs)", value=5.0)
    agents = st.sidebar.number_input("Agents", value=5, step=1)
    concurrency = st.sidebar.number_input(
        "Concurrent Chats per Agent", value=1, min_value=1, step=1
    )
    effective_agents = int(agents * concurrency)
    aht = st.sidebar.number_input("Average Handle Time (seconds)", value=180)
    target = st.sidebar.number_input("Service Level Target (seconds)", value=20)

    if st.sidebar.button("Compute"):
        b = X.erlang_b(traffic, effective_agents)
        c = X.erlang_c(traffic, effective_agents)
        sl = CHAT.service_level(traffic, effective_agents, aht, target)
        asa = CHAT.asa(traffic, effective_agents, aht)
        st.write("### Results")
        st.write(f"Erlang B Blocking: {b:.4f}")
        st.write(f"Erlang C Waiting: {c:.4f}")
        st.write(f"Service Level: {sl:.4f}")
        st.write(f"ASA: {asa:.2f} seconds")

    st.sidebar.header("Sensitivity Analysis")
    sens_start = st.sidebar.number_input(
        "Traffic Range Start", value=max(0.0, traffic - 3.0)
    )
    sens_end = st.sidebar.number_input("Traffic Range End", value=traffic + 3.0)
    sens_points = st.sidebar.number_input(
        "Points", min_value=5, value=20, step=1
    )
    if st.sidebar.button("Run Sensitivity"):
        tf_range = np.linspace(sens_start, sens_end, int(sens_points))
        df = BL.sensitivity(tf_range, effective_agents, aht, target)
        st.line_chart(df.set_index("traffic"))

    st.sidebar.header("Monte Carlo")
    mc_mean = st.sidebar.number_input("Mean Traffic", value=traffic)
    iters = st.sidebar.number_input("Iterations", value=500, step=100)
    if st.sidebar.button("Run Monte Carlo"):
        series = BL.monte_carlo(mc_mean, effective_agents, aht, target, int(iters))
        st.write(series.describe())
        counts, bins = np.histogram(series, bins=20)
        st.bar_chart(pd.DataFrame({"count": counts}, index=bins[:-1]))

    st.sidebar.header("Staffing Optimization")
    svc_target = st.sidebar.slider("Target Service Level", 0.5, 0.99, 0.8)
    if st.sidebar.button("Optimize Staffing"):
        required = CHAT.required_agents(traffic, aht, svc_target, target)
        physical_agents = int(np.ceil(required / concurrency))
        st.write(f"Required Agents (with concurrency): {physical_agents}")

if __name__ == "__main__":
    run_app()

