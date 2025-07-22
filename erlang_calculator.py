# coding: utf-8

"""Erlang calculator with Streamlit UI."""

import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.optimize import minimize
import streamlit as st

class X:
    """Collection of Erlang formulas."""

    @staticmethod
    def erlang_b(traffic, trunks):
        """Compute Erlang B blocking probability."""
        traffic = float(traffic)
        trunks = int(trunks)
        inv_b = np.sum([(traffic ** k) / factorial(k) for k in range(trunks + 1)])
        b = ((traffic ** trunks) / factorial(trunks)) / inv_b
        return b

    @staticmethod
    def erlang_c(traffic, agents):
        """Compute Erlang C waiting probability."""
        traffic = float(traffic)
        agents = int(agents)
        rho = traffic / agents
        if rho >= 1:
            return 1.0
        numerator = ((traffic ** agents) / factorial(agents)) * (agents / (agents - traffic))
        denom = np.sum([(traffic ** k) / factorial(k) for k in range(agents)]) + numerator
        return numerator / denom

class CHAT:
    """Call center metrics."""

    @staticmethod
    def service_level(traffic, agents, aht, target):
        c = X.erlang_c(traffic, agents)
        return 1 - c * np.exp(-(agents - traffic) * target / aht)

    @staticmethod
    def asa(traffic, agents, aht):
        c = X.erlang_c(traffic, agents)
        if agents <= traffic:
            return np.inf
        return (c / (agents - traffic)) * aht

    @staticmethod
    def required_agents(traffic, aht, target_service, target_time=20):
        """Find minimum agents to hit target service level."""
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



def run_app():
    st.title("Erlang Calculator")
    st.sidebar.header("Inputs")
    traffic = st.sidebar.number_input("Traffic Intensity (erlangs)", value=5.0)
    trunks = st.sidebar.number_input("Trunks/Agents", value=5, step=1)
    aht = st.sidebar.number_input("Average Handle Time (seconds)", value=180)
    target = st.sidebar.number_input("Service Level Target (seconds)", value=20)

    if st.sidebar.button("Compute"):
        b = X.erlang_b(traffic, trunks)
        c = X.erlang_c(traffic, trunks)
        sl = CHAT.service_level(traffic, trunks, aht, target)
        asa = CHAT.asa(traffic, trunks, aht)
        st.write("### Results")
        st.write(f"Erlang B Blocking: {b:.4f}")
        st.write(f"Erlang C Waiting: {c:.4f}")
        st.write(f"Service Level: {sl:.4f}")
        st.write(f"ASA: {asa:.2f} seconds")

    st.sidebar.header("Sensitivity Analysis")
    if st.sidebar.button("Run Sensitivity"):
        tf_range = np.linspace(max(0.1, traffic-3), traffic+3, 20)
        df = BL.sensitivity(tf_range, trunks, aht, target)
        st.line_chart(df.set_index('traffic'))

    st.sidebar.header("Monte Carlo")
    iters = st.sidebar.number_input("Iterations", value=500, step=100)
    if st.sidebar.button("Run Monte Carlo"):
        series = BL.monte_carlo(traffic, trunks, aht, target, int(iters))
        st.write(series.describe())
        counts, bins = np.histogram(series, bins=20)
        st.bar_chart(pd.DataFrame({'count': counts}, index=bins[:-1]))

    st.sidebar.header("Staffing Optimization")
    svc_target = st.sidebar.slider("Target Service Level", 0.5, 0.99, 0.8)
    if st.sidebar.button("Optimize Staffing"):
        agents = CHAT.required_agents(traffic, aht, svc_target, target)
        st.write(f"Required Agents: {agents}")

if __name__ == "__main__":
    run_app()

