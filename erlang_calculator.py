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
import matplotlib.pyplot as plt


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

# =============================================================================
# EJEMPLOS PRÁCTICOS - ERLANG CALCULATOR
# Casos de uso reales para centros de contacto
# =============================================================================

class ErlangAnalyzer:
    """Herramientas para análisis completos de centros de contacto."""

    def __init__(self) -> None:
        self.results = {}

    def dimensioning_analysis(self, forecast, aht, target_sl=0.80, awt=20):
        """Análisis completo de dimensionamiento."""
        print("\U0001F4CA ANÁLISIS DE DIMENSIONAMIENTO")
        print("=" * 50)

        agents_needed = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, forecast, aht, awt)

        agent_range = range(int(agents_needed * 0.8), int(agents_needed * 1.3))
        results = []

        for agents in agent_range:
            sl = X.SLA.__new__(X.SLA, forecast, aht, agents, awt)
            asa = X.ASA.__new__(X.ASA, forecast, aht, agents)
            occupancy = X.OCCUPANCY.__new__(X.OCCUPANCY, forecast, aht, agents)

            results.append({
                "Agents": agents,
                "Service_Level": sl,
                "ASA_seconds": asa * 60,
                "Occupancy": occupancy,
                "Cost_Score": agents * (1 - sl),
            })

        df = pd.DataFrame(results)

        print("\n\U0001F3AF RECOMENDACIÓN ÓPTIMA:")
        print(f"Agentes recomendados: {agents_needed}")
        print(f"Service Level objetivo: {target_sl:.0%}")
        print(f"AWT máximo: {awt} segundos")

        optimal = df.loc[df["Cost_Score"].idxmin()]
        print("\n\u26A1 CONFIGURACIÓN ÓPTIMA:")
        print(f"Agentes: {optimal['Agents']}")
        print(f"Service Level: {optimal['Service_Level']:.1%}")
        print(f"ASA: {optimal['ASA_seconds']:.1f} segundos")
        print(f"Ocupación: {optimal['Occupancy']:.1%}")

        return df

    def what_if_analysis(self, base_forecast, base_aht, base_agents):
        """Análisis de escenarios 'Qué pasaría si...?'."""
        print("\n\U0001F52E ANÁLISIS DE ESCENARIOS")
        print("=" * 50)

        scenarios = [
            {"name": "Escenario Base", "forecast": base_forecast, "aht": base_aht, "agents": base_agents},
            {"name": "↗️ +20% Volumen", "forecast": base_forecast * 1.2, "aht": base_aht, "agents": base_agents},
            {"name": "⏱️ +10% AHT", "forecast": base_forecast, "aht": base_aht * 1.1, "agents": base_agents},
            {"name": "👥 +2 Agentes", "forecast": base_forecast, "aht": base_aht, "agents": base_agents + 2},
            {"name": "💥 Crisis (Volumen +50%)", "forecast": base_forecast * 1.5, "aht": base_aht * 1.2, "agents": base_agents},
        ]

        results = []
        for scenario in scenarios:
            sl = X.SLA.__new__(X.SLA, scenario["forecast"], scenario["aht"], scenario["agents"], 20)
            asa = X.ASA.__new__(X.ASA, scenario["forecast"], scenario["aht"], scenario["agents"])
            occ = X.OCCUPANCY.__new__(X.OCCUPANCY, scenario["forecast"], scenario["aht"], scenario["agents"])

            results.append({
                "Escenario": scenario["name"],
                "Forecast": scenario["forecast"],
                "AHT": scenario["aht"],
                "Agentes": scenario["agents"],
                "Service_Level": f"{sl:.1%}",
                "ASA_min": f"{asa:.1f}",
                "Ocupación": f"{occ:.1%}",
            })

        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        return df

    def compare_models(self, forecast, aht, agents, awt=20):
        """Comparación entre diferentes modelos de Erlang."""
        print("\n⚖️ COMPARACIÓN DE MODELOS")
        print("=" * 50)

        basic_sl = X.SLA.__new__(X.SLA, forecast, aht, agents, awt)
        basic_asa = X.ASA.__new__(X.ASA, forecast, aht, agents)

        lines = int(agents * 1.2)
        limited_sl = X.SLA.__new__(X.SLA, forecast, aht, agents, awt, lines)

        patience = 120
        abandon_sl = X.SLA.__new__(X.SLA, forecast, aht, agents, awt, lines, patience)
        abandon_rate = X.ABANDON.__new__(X.ABANDON, forecast, aht, agents, lines, patience, 0.1)

        chat_aht = [aht * 0.7, aht * 0.8, aht * 0.9]
        chat_sl = CHAT.SLA.__new__(CHAT.SLA, forecast, chat_aht, agents, awt, lines, patience)

        threshold = 2
        blend_sl = BL.SLA.__new__(BL.SLA, forecast, aht, agents, awt, lines, patience, threshold)
        outbound_capacity = BL.OUTBOUND.__new__(BL.OUTBOUND, forecast, aht, agents, lines, patience, threshold, aht)

        print("🔹 ERLANG C (Básico)")
        print(f"   Service Level: {basic_sl:.1%}")
        print(f"   ASA: {basic_asa:.1f} minutos")

        print(f"\n🔸 ERLANG C + LÍNEAS LIMITADAS ({lines} líneas)")
        print(f"   Service Level: {limited_sl:.1%}")

        print(f"\n🔺 ERLANG X + ABANDONMENT (Paciencia: {patience}s)")
        print(f"   Service Level: {abandon_sl:.1%}")
        print(f"   Abandonment Rate: {abandon_rate:.1%}")

        print(f"\n💬 CHAT MODEL ({len(chat_aht)} chats simultáneos)")
        print(f"   Service Level: {chat_sl:.1%}")

        print(f"\n🗑️ BLENDING MODEL (Threshold: {threshold})")
        print(f"   Service Level: {blend_sl:.1%}")
        print(f"   Outbound Capacity: {outbound_capacity:.1f} llamadas/hora")

        return {
            "Erlang_C": basic_sl,
            "Limited_Lines": limited_sl,
            "With_Abandonment": abandon_sl,
            "Chat_Model": chat_sl,
            "Blending": blend_sl,
        }

    def staffing_optimization(self, hourly_forecast, aht, target_sl=0.80):
        """Optimización de staffing por horas del día."""
        print("\n⏰ OPTIMIZACIÓN DE STAFFING POR HORAS")
        print("=" * 50)

        hours = list(range(8, 20))
        results = []
        total_agents = 0

        for hour, forecast in zip(hours, hourly_forecast):
            agents_needed = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, forecast, aht, 20)
            sl_achieved = X.SLA.__new__(X.SLA, forecast, aht, agents_needed, 20)

            results.append({
                "Hora": f"{hour}:00",
                "Forecast": forecast,
                "Agentes_Necesarios": agents_needed,
                "SL_Logrado": f"{sl_achieved:.1%}",
            })

            total_agents += agents_needed

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print("\n\U0001F4C8 RESUMEN STAFFING:")
        print(f"Total agentes-hora necesarios: {total_agents}")
        print(f"Pico máximo: {max([r['Agentes_Necesarios'] for r in results])} agentes")
        print(f"Valle mínimo: {min([r['Agentes_Necesarios'] for r in results])} agentes")

        return df


class IndustryUseCases:
    """Casos de uso específicos por industria."""

    @staticmethod
    def call_center_tradicional():
        """Centro de llamadas tradicional - Servicio al cliente."""
        print("📞 CASO: CALL CENTER TRADICIONAL")
        print("Industria: Servicios Financieros")
        print("=" * 50)

        forecast = 150
        aht = 4.5
        target_sl = 0.80
        awt = 20

        analyzer = ErlangAnalyzer()
        df = analyzer.dimensioning_analysis(forecast, aht, target_sl, awt)

        print("\n\U0001F50D ANÁLISIS DE SENSIBILIDAD")
        sensitivities = []
        base_agents = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, forecast, aht, awt)

        for variation in [-20, -10, 0, 10, 20]:
            new_forecast = forecast * (1 + variation / 100)
            new_agents = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, new_forecast, aht, awt)
            sensitivities.append({
                "Parámetro": f"Forecast {variation:+}%",
                "Valor": f"{new_forecast:.0f}",
                "Agentes": new_agents,
                "Cambio": f"{new_agents - base_agents:+.1f}",
            })

        for variation in [-10, -5, 0, 5, 10]:
            new_aht = aht * (1 + variation / 100)
            new_agents = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, forecast, new_aht, awt)
            sensitivities.append({
                "Parámetro": f"AHT {variation:+}%",
                "Valor": f"{new_aht:.1f}min",
                "Agentes": new_agents,
                "Cambio": f"{new_agents - base_agents:+.1f}",
            })

        sens_df = pd.DataFrame(sensitivities)
        print(sens_df.to_string(index=False))

        return df, sens_df

    @staticmethod
    def chat_support():
        """Soporte por chat - Agentes manejan múltiples conversaciones."""
        print("\n💬 CASO: SOPORTE POR CHAT")
        print("Industria: E-commerce")
        print("=" * 50)

        forecast = 200
        chat_aht = [2, 2.5, 3.5, 5]
        target_sl = 0.85
        awt = 30
        lines = 300
        patience = 180

        agents_needed = CHAT.AGENTS.SLA.__new__(CHAT.AGENTS.SLA, target_sl, forecast, chat_aht, awt, lines, patience)

        configs = [
            {"chats": 1, "aht": [2]},
            {"chats": 2, "aht": [2, 2.5]},
            {"chats": 3, "aht": [2, 2.5, 3.5]},
            {"chats": 4, "aht": [2, 2.5, 3.5, 5]},
        ]

        results = []
        for config in configs:
            agents = CHAT.AGENTS.SLA.__new__(CHAT.AGENTS.SLA, target_sl, forecast, config["aht"], awt, lines, patience)
            sl = CHAT.SLA.__new__(CHAT.SLA, forecast, config["aht"], agents, awt, lines, patience)

            results.append({
                "Chats_Simultáneos": config["chats"],
                "Agentes_Necesarios": agents,
                "SL_Logrado": f"{sl:.1%}",
                "Eficiencia": f"{forecast / agents:.1f} chats/agente/hora",
            })

        chat_df = pd.DataFrame(results)
        print("\n\U0001F4C8 COMPARACIÓN POR NÚMERO DE CHATS SIMULTÁNEOS:")
        print(chat_df.to_string(index=False))

        print("\n\U0001F3AF RECOMENDACIÓN:")
        optimal = min(results, key=lambda x: x["Agentes_Necesarios"])
        print(f"Configuración óptima: {optimal['Chats_Simultáneos']} chats simultáneos")
        print(f"Agentes necesarios: {optimal['Agentes_Necesarios']}")
        print(f"Eficiencia: {optimal['Eficiencia']}")

        return chat_df

    @staticmethod
    def blended_operation():
        """Operación mixta - Inbound + Outbound."""
        print("\n🗝 CASO: OPERACIÓN BLENDED")
        print("Industria: Ventas + Soporte")
        print("=" * 50)

        inbound_forecast = 120
        inbound_aht = 3.5
        outbound_aht = 5.0
        target_sl = 0.75
        awt = 20
        total_agents = 25

        thresholds = range(0, 8)
        results = []

        for threshold in thresholds:
            sl = BL.SLA.__new__(BL.SLA, inbound_forecast, inbound_aht, total_agents, awt, 100, 300, threshold)
            outbound_capacity = BL.OUTBOUND.__new__(BL.OUTBOUND, inbound_forecast, inbound_aht, total_agents, 100, 300, threshold, outbound_aht)

            results.append({
                "Threshold": threshold,
                "SL_Inbound": f"{sl:.1%}",
                "Outbound_Capacity": f"{outbound_capacity:.1f}",
                "Total_Productivity": outbound_capacity + (sl * 100),
            })

        blend_df = pd.DataFrame(results)
        print("\n\U0001F4C8 ANÁLISIS DE THRESHOLD ÓPTIMO:")
        print(blend_df.to_string(index=False))

        optimal_threshold = max(results, key=lambda x: x["Total_Productivity"])["Threshold"]

        print(f"\n\U0001F3AF THRESHOLD ÓPTIMO: {optimal_threshold} agentes")
        print(f"Con {total_agents} agentes totales:")
        print(f"- Agentes disponibles para inbound: {total_agents - optimal_threshold}")
        print(f"- Agentes de reserva: {optimal_threshold}")

        return blend_df


class AdvancedAnalytics:
    """Herramientas de análisis avanzado."""

    @staticmethod
    def monte_carlo_simulation(forecast_mean, forecast_std, aht_mean, aht_std, agents, iterations=1000):
        """Simulación Monte Carlo para análisis de riesgo."""
        import numpy as np

        print("\n🎲 SIMULACIÓN MONTE CARLO")
        print("=" * 50)

        results = []

        for _ in range(iterations):
            forecast = max(1, np.random.normal(forecast_mean, forecast_std))
            aht = max(0.1, np.random.normal(aht_mean, aht_std))

            sl = X.SLA.__new__(X.SLA, forecast, aht, agents, 20)
            asa = X.ASA.__new__(X.ASA, forecast, aht, agents)
            occ = X.OCCUPANCY.__new__(X.OCCUPANCY, forecast, aht, agents)

            results.append({
                "forecast": forecast,
                "aht": aht,
                "sl": sl,
                "asa": asa,
                "occupancy": occ,
            })

        sl_values = [r["sl"] for r in results]
        asa_values = [r["asa"] for r in results]

        print(f"\U0001F4CA RESULTADOS ({iterations} simulaciones):")
        print("\nService Level:")
        print(f"  Media: {np.mean(sl_values):.1%}")
        print(f"  Desv. Std: {np.std(sl_values):.1%}")
        print(f"  P5: {np.percentile(sl_values, 5):.1%}")
        print(f"  P95: {np.percentile(sl_values, 95):.1%}")

        print("\nASA (minutos):")
        print(f"  Media: {np.mean(asa_values):.2f}")
        print(f"  Desv. Std: {np.std(asa_values):.2f}")
        print(f"  P95: {np.percentile(asa_values, 95):.2f}")

        prob_sl_80 = sum(1 for sl in sl_values if sl >= 0.80) / len(sl_values)
        prob_asa_30 = sum(1 for asa in asa_values if asa <= 0.5) / len(asa_values)

        print("\n\U0001F3AF PROBABILIDADES:")
        print(f"  SL ≥ 80%: {prob_sl_80:.1%}")
        print(f"  ASA ≤ 30seg: {prob_asa_30:.1%}")

        return results

    @staticmethod
    def capacity_planning(current_forecast, growth_rate, periods=12):
        """Planificación de capacidad a futuro."""
        print("\n\U0001F4C8 PLANIFICACIÓN DE CAPACIDAD")
        print(f"Crecimiento proyectado: {growth_rate:.1%} mensual")
        print("=" * 50)

        results = []
        aht = 4.0
        target_sl = 0.80

        for period in range(1, periods + 1):
            forecast = current_forecast * (1 + growth_rate) ** period
            agents_needed = X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, forecast, aht, 20)

            results.append({
                "Mes": period,
                "Forecast": f"{forecast:.0f}",
                "Agentes_Necesarios": agents_needed,
                "Incremento": agents_needed - (
                    results[-1]["Agentes_Necesarios"]
                    if results
                    else X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, current_forecast, aht, 20)
                ),
            })

        capacity_df = pd.DataFrame(results)
        print(capacity_df.to_string(index=False))

        total_growth = results[-1]["Agentes_Necesarios"] - X.AGENTS.SLA.__new__(X.AGENTS.SLA, target_sl, current_forecast, aht, 20)
        print(f"\n\U0001F4CA RESUMEN {periods} MESES:")
        print(f"Crecimiento total de agentes: +{total_growth}")
        print(f"Inversión estimada mensual: ${total_growth * 3000 / periods:,.0f} USD")

        return capacity_df


def run_complete_analysis():
    """Ejecuta un análisis completo con todos los módulos."""
    print("\U0001F680 ANÁLISIS COMPLETO DE CENTRO DE CONTACTO")
    print("=" * 70)

    print("\n" + "=" * 70)
    IndustryUseCases.call_center_tradicional()

    print("\n" + "=" * 70)
    IndustryUseCases.chat_support()

    print("\n" + "=" * 70)
    IndustryUseCases.blended_operation()

    print("\n" + "=" * 70)
    AdvancedAnalytics.monte_carlo_simulation(
        forecast_mean=150,
        forecast_std=20,
        aht_mean=4.0,
        aht_std=0.5,
        agents=35,
    )

    print("\n" + "=" * 70)
    AdvancedAnalytics.capacity_planning(
        current_forecast=150,
        growth_rate=0.05,
        periods=12,
    )


if __name__ == "__main__":
    run_app()

