# =============================================================================
# ERLANG CALCULATOR COMPLETO - STREAMLIT APP
# Implementación completa con X, CHAT, BL y ERLANG O
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import optimize
import math

# =============================================================================
# CONFIGURACIÓN DE STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Erlang Calculator Pro",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# VISUALIZACIÓN DINÁMICA DE AGENTES
# =============================================================================

def create_agent_visualization(forecast, aht, agents, awt, interval_seconds=3600):
    """Genera una visualización del estado de agentes y métricas en tiempo real"""

    arrival_rate = forecast / interval_seconds
    sl = service_level_erlang_c(arrival_rate, aht, agents, awt)
    asa = waiting_time_erlang_c(arrival_rate, aht, agents)
    occupancy = occupancy_erlang_c(arrival_rate, aht, agents)

    fig = go.Figure()

    agents_per_row = 10
    rows = math.ceil(agents / agents_per_row)

    colors = {
        "busy": "#FF6B6B",
        "available": "#4ECDC4",
        "break": "#45B7D1",
        "offline": "#95A5A6",
    }

    busy_agents = int(agents * occupancy)
    available_agents = agents - busy_agents

    agent_states = ["busy"] * busy_agents + ["available"] * available_agents

    agent_x, agent_y, agent_colors, agent_text = [], [], [], []

    for i in range(agents):
        row = i // agents_per_row
        col = i % agents_per_row

        x = col * 1.2
        y = rows - row - 1

        agent_x.append(x)
        agent_y.append(y)

        state = agent_states[i] if i < len(agent_states) else "offline"
        agent_colors.append(colors[state])

        status_emoji = {
            "busy": "🔴",
            "available": "🟢",
            "break": "🔵",
            "offline": "⚫",
        }
        agent_text.append(f"Agente {i+1}<br>{status_emoji.get(state, '⚫')}")

    fig.add_trace(
        go.Scatter(
            x=agent_x,
            y=agent_y,
            mode="markers+text",
            marker=dict(size=40, color=agent_colors, symbol="circle", line=dict(width=2, color="white")),
            text=["👤" for _ in range(agents)],
            textfont=dict(size=20),
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>Estado: %{customdata}<br><extra></extra>",
            customdata=[state.title() for state in agent_states],
            name="Agentes",
        )
    )

    if agents <= arrival_rate * aht:
        queue_length = 20
    else:
        pc = erlang_c(arrival_rate * aht, agents)
        queue_length = max(0, int(pc * arrival_rate * asa))

    # Posición inicial de la cola a la derecha de la sección de agentes
    queue_x = 15
    queue_start_y = rows - 1

    if queue_length > 0:
        queue_positions_x, queue_positions_y = [], []
        for q in range(min(queue_length, 15)):
            if q < 5:
                qx = queue_x
                qy = queue_start_y - (q * 0.3)
            elif q < 10:
                qx = queue_x + 0.8
                qy = queue_start_y - ((q - 5) * 0.3)
            else:
                qx = queue_x + 1.6
                qy = queue_start_y - ((q - 10) * 0.3)

            queue_positions_x.append(qx)
            queue_positions_y.append(qy)

        if asa <= 20:
            queue_color = "#4ECDC4"
        elif asa <= 60:
            queue_color = "#F7DC6F"
        else:
            queue_color = "#FF6B6B"

        fig.add_trace(
            go.Scatter(
                x=queue_positions_x,
                y=queue_positions_y,
                mode="markers+text",
                marker=dict(size=25, color=queue_color, symbol="circle", line=dict(width=1, color="gray")),
                text=["⏳" for _ in range(len(queue_positions_x))],
                textfont=dict(size=12),
                textposition="middle center",
                hovertemplate="<b>Persona en cola</b><br>Tiempo esperando: ~%{customdata:.1f} seg<br><extra></extra>",
                customdata=[asa * (q + 1) / queue_length for q in range(len(queue_positions_x))],
                name="Cola de Espera",
            )
        )

        if queue_length > 15:
            fig.add_trace(
                go.Scatter(
                    x=[queue_x + 0.8],
                    y=[queue_start_y - 5],
                    mode="text",
                    text=[f"+{queue_length - 15} más"],
                    textfont=dict(size=10, color="red"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Las métricas se sitúan más a la derecha para evitar superposición
    metrics_x = 25
    metrics_y = rows - 1

    sl_color = "#4ECDC4" if sl >= 0.8 else "#F7DC6F" if sl >= 0.7 else "#FF6B6B"
    fig.add_trace(
        go.Scatter(
            x=[metrics_x],
            y=[metrics_y],
            mode="markers+text",
            marker=dict(size=60, color=sl_color, symbol="circle"),
            text=[f"{sl:.0%}"],
            textfont=dict(size=12, color="white"),
            textposition="middle center",
            hovertemplate="<b>Service Level</b><br>%{text}<br><extra></extra>",
            name="Service Level",
        )
    )

    asa_color = "#4ECDC4" if asa <= 20 else "#F7DC6F" if asa <= 60 else "#FF6B6B"
    fig.add_trace(
        go.Scatter(
            x=[metrics_x],
            y=[metrics_y - 2.5],
            mode="markers+text",
            marker=dict(size=60, color=asa_color, symbol="circle"),
            text=[f"{asa:.0f}s"],
            textfont=dict(size=12, color="white"),
            textposition="middle center",
            hovertemplate="<b>ASA (Average Speed of Answer)</b><br>%{text} segundos<br><extra></extra>",
            name="ASA",
        )
    )

    occ_color = "#4ECDC4" if 0.7 <= occupancy <= 0.85 else "#F7DC6F" if occupancy <= 0.9 else "#FF6B6B"
    fig.add_trace(
        go.Scatter(
            x=[metrics_x],
            y=[metrics_y - 5],
            mode="markers+text",
            marker=dict(size=60, color=occ_color, symbol="circle"),
            text=[f"{occupancy:.0%}"],
            textfont=dict(size=12, color="white"),
            textposition="middle center",
            hovertemplate="<b>Ocupación</b><br>%{text}<br><extra></extra>",
            name="Ocupación",
        )
    )

    fig.add_annotation(
        x=agents_per_row * 1.2 / 2,
        y=rows + 0.5,
        text="<b>👥 AGENTES</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    if queue_length > 0:
        fig.add_annotation(
            x=queue_x + 0.8,
            y=rows + 0.5,
            text=f"<b>📞 COLA ({queue_length})</b>",
            showarrow=False,
            font=dict(size=16, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )

    fig.add_annotation(
        x=metrics_x,
        y=rows + 0.5,
        text="<b>📊 MÉTRICAS</b>",
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    fig_width = max(800, int((metrics_x + 5) * 30))

    fig.update_layout(
        title={
            "text": f"<b>🏢 Vista en Tiempo Real del Call Center</b><br><span style='font-size:14px'>Forecast: {forecast:.0f} llamadas/h | AHT: {aht:.0f}s | Agentes: {agents}</span>",
            "x": 0.5,
            "font": {"size": 18},
        },
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, metrics_x + 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, rows + 1]),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=400 + (rows * 50),
        width=fig_width,
        margin=dict(l=80, r=80, t=80, b=100),
    )

    return fig


def create_real_time_dashboard(forecast, aht, agents, awt, interval_seconds=3600):
    """Crea un dashboard con la visualización de agentes y su estado"""

    agent_viz = create_agent_visualization(forecast, aht, agents, awt, interval_seconds)

    arrival_rate = forecast / interval_seconds
    sl = service_level_erlang_c(arrival_rate, aht, agents, awt)
    asa = waiting_time_erlang_c(arrival_rate, aht, agents)
    occupancy = occupancy_erlang_c(arrival_rate, aht, agents)

    states_data = {
        "Estado": ["🔴 Ocupados", "🟢 Disponibles", "📞 En Cola"],
        "Cantidad": [
            int(agents * occupancy),
            agents - int(agents * occupancy),
            max(0, int(erlang_c(arrival_rate * aht, agents) * arrival_rate * asa)) if agents > arrival_rate * aht else 20,
        ],
    }

    status_bar = px.bar(
        x=states_data["Cantidad"],
        y=states_data["Estado"],
        orientation="h",
        color=states_data["Estado"],
        color_discrete_map={"🔴 Ocupados": "#FF6B6B", "🟢 Disponibles": "#4ECDC4", "📞 En Cola": "#F7DC6F"},
        title="📊 Estado Actual del Sistema",
    )

    status_bar.update_layout(showlegend=False, height=200, margin=dict(l=50, r=50, t=50, b=50))

    return agent_viz, status_bar

# =============================================================================
# FUNCIONES MATEMÁTICAS BASE
# =============================================================================

@st.cache_data
def factorial_approx(n):
    """Aproximación de factorial usando Stirling para números grandes"""
    if n < 170:
        return math.factorial(int(n))
    else:
        return math.sqrt(2 * math.pi * n) * (n / math.e) ** n

@st.cache_data
def erlang_b(traffic, agents):
    """Fórmula de Erlang B (blocking probability)"""
    agents = int(agents)
    if agents == 0:
        return 1.0
    if traffic == 0:
        return 0.0
    
    b = 1.0
    for i in range(1, agents + 1):
        b = (traffic * b) / (i + traffic * b)
    return b

@st.cache_data
def erlang_c(traffic, agents):
    """Fórmula de Erlang C (waiting probability)"""
    agents = int(agents)
    if agents <= traffic:
        return 1.0
    
    eb = erlang_b(traffic, agents)
    rho = traffic / agents
    
    if rho >= 1:
        return 1.0
    
    return eb / (1 - rho + rho * eb)

@st.cache_data
def service_level_erlang_c(forecast, aht, agents, awt):
    """Calcula el nivel de servicio usando Erlang C"""
    traffic = forecast * aht
    agents = int(agents)
    
    if agents <= traffic:
        return 0.0
    
    pc = erlang_c(traffic, agents)
    
    if pc == 0:
        return 1.0
    
    exp_factor = math.exp(-(agents - traffic) * awt / aht)
    return 1 - pc * exp_factor

@st.cache_data
def waiting_time_erlang_c(forecast, aht, agents):
    """Calcula el tiempo promedio de espera (ASA)"""
    traffic = forecast * aht
    agents = int(agents)

    if agents <= traffic:
        return float('inf')
    
    pc = erlang_c(traffic, agents)
    return (pc * aht) / (agents - traffic)

@st.cache_data
def occupancy_erlang_c(forecast, aht, agents):
    """Calcula la ocupación de los agentes"""
    traffic = forecast * aht
    agents = int(agents)
    return min(traffic / agents, 1.0)

@st.cache_data
def erlang_x_abandonment(forecast, aht, agents, lines, patience):
    """Calcula la probabilidad de abandono en modelo Erlang X"""
    traffic = forecast * aht
    agents = int(agents)
    
    if patience == 0:
        return erlang_b(traffic, lines)
    
    if agents >= traffic:
        pc = erlang_c(traffic, agents)
        avg_wait = waiting_time_erlang_c(forecast, aht, agents)
        return pc * (1 - math.exp(-avg_wait / patience))
    else:
        return min(1.0, traffic / lines)

# =============================================================================
# MÓDULO ERLANG O (OUTBOUND ONLY)
# =============================================================================

class ERLANG_O:
    """Módulo para cálculos de campañas outbound puras"""
    
    @staticmethod
    def productivity(agents, hours_per_day, calls_per_hour, success_rate=0.3):
        """
        Calcula la productividad de una campaña outbound
        
        Parameters:
        agents (int): Número de agentes
        hours_per_day (float): Horas de trabajo por día
        calls_per_hour (float): Llamadas por hora por agente
        success_rate (float): Tasa de éxito (contactos efectivos)
        
        Returns:
        dict: Métricas de productividad
        """
        total_calls = agents * hours_per_day * calls_per_hour
        successful_calls = total_calls * success_rate
        
        return {
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'success_rate': success_rate,
            'calls_per_agent_day': hours_per_day * calls_per_hour,
            'successful_per_agent_day': hours_per_day * calls_per_hour * success_rate
        }
    
    @staticmethod
    def agents_for_target(target_calls_day, hours_per_day, calls_per_hour, success_rate=0.3):
        """
        Calcula agentes necesarios para lograr objetivo de llamadas exitosas
        """
        calls_per_agent_day = hours_per_day * calls_per_hour * success_rate
        return math.ceil(target_calls_day / calls_per_agent_day)
    
    @staticmethod
    def dialer_ratio(answer_rate=0.25, agent_talk_time=5, wait_between_calls=2):
        """
        Calcula la ratio óptima del predictive dialer
        
        Parameters:
        answer_rate (float): Tasa de respuesta (0.2-0.3 típico)
        agent_talk_time (float): Tiempo promedio de conversación (minutos)
        wait_between_calls (float): Tiempo entre llamadas (minutos)
        
        Returns:
        float: Ratio de marcado (líneas por agente)
        """
        cycle_time = agent_talk_time + wait_between_calls
        ratio = cycle_time / (agent_talk_time * answer_rate)
        return max(1.0, ratio)

# =============================================================================
# MÓDULOS PRINCIPALES
# =============================================================================

class X:
    """Módulo Erlang C/X"""
    
    class SLA:
        @staticmethod
        def calculate(forecast, aht, agents, awt, lines=None, patience=None, retrials=None):
            if lines is None and patience is None:
                return service_level_erlang_c(forecast, aht, agents, awt)
            elif lines is not None and patience is None:
                traffic = forecast * aht
                blocking = erlang_b(traffic, lines)
                if blocking > 0.99:
                    return 0.0
                effective_forecast = forecast * (1 - blocking)
                return service_level_erlang_c(effective_forecast, aht, agents, awt)
            else:
                traffic = forecast * aht
                if agents <= traffic:
                    return 0.0
                base_sl = service_level_erlang_c(forecast, aht, agents, awt)
                abandon_rate = erlang_x_abandonment(forecast, aht, agents, lines or 999, patience or 999)
                return base_sl * (1 - abandon_rate * 0.5)
    
    class AGENTS:
        @staticmethod
        def for_sla(sl_target, forecast, aht, awt, lines=None, patience=None):
            traffic = forecast * aht
            
            def objective(agents):
                agents = int(round(agents))
                if agents <= 0:
                    return float('inf')
                sl = X.SLA.calculate(forecast, aht, agents, awt, lines, patience)
                return abs(sl - sl_target)
            
            result = optimize.minimize_scalar(objective, bounds=(traffic * 0.5, traffic * 3), method='bounded')
            return max(1, round(result.x, 1))
        
        @staticmethod
        def for_asa(asa_target, forecast, aht, lines=None, patience=None):
            traffic = forecast * aht
            
            def objective(agents):
                if agents <= traffic:
                    return float('inf')
                actual_asa = waiting_time_erlang_c(forecast, aht, agents)
                return abs(actual_asa - asa_target)
            
            result = optimize.minimize_scalar(objective, bounds=(traffic + 0.1, traffic * 2), method='bounded')
            return max(1, round(result.x, 1))
    
    @staticmethod
    def asa(forecast, aht, agents):
        return waiting_time_erlang_c(forecast, aht, agents)
    
    @staticmethod
    def occupancy(forecast, aht, agents):
        return occupancy_erlang_c(forecast, aht, agents)
    
    @staticmethod
    def abandonment(forecast, aht, agents, lines, patience):
        return erlang_x_abandonment(forecast, aht, agents, lines, patience)

    @staticmethod
    def erlang_b(traffic, agents):
        return erlang_b(traffic, agents)

    @staticmethod
    def erlang_c(traffic, agents):
        return erlang_c(traffic, agents)

class CHAT:
    """Módulo Chat Multi-canal"""
    
    @staticmethod
    def sla(forecast, aht_list, agents, awt, lines, patience):
        parallel_capacity = len(aht_list)
        avg_aht = sum(aht_list) / len(aht_list)
        effectiveness = 0.7 + (0.3 / parallel_capacity)
        effective_agents = agents * parallel_capacity * effectiveness
        return service_level_erlang_c(forecast, avg_aht, effective_agents, awt)
    
    @staticmethod
    def agents_for_sla(sl_target, forecast, aht_list, awt, lines, patience):
        parallel_capacity = len(aht_list)
        avg_aht = sum(aht_list) / len(aht_list)
        effectiveness = 0.7 + (0.3 / parallel_capacity)
        
        def objective(agents):
            if agents <= 0:
                return float('inf')
            effective_agents = agents * parallel_capacity * effectiveness
            sl = service_level_erlang_c(forecast, avg_aht, effective_agents, awt)
            return abs(sl - sl_target)
        
        traffic = forecast * avg_aht
        result = optimize.minimize_scalar(objective, bounds=(0.1, traffic), method='bounded')
        return max(1, round(result.x, 1))
    
    @staticmethod
    def asa(forecast, aht_list, agents, lines=None, patience=None):
        if isinstance(aht_list, list):
            parallel_capacity = len(aht_list)
            avg_aht = sum(aht_list) / len(aht_list)
            effectiveness = 0.7 + (0.3 / parallel_capacity)
            effective_agents = agents * parallel_capacity * effectiveness
            return waiting_time_erlang_c(forecast, avg_aht, effective_agents)
        else:
            traffic = forecast
            agents_val = aht_list
            aht = agents
            forecast_rate = traffic / aht if aht else 0
            return waiting_time_erlang_c(forecast_rate, aht, agents_val)

    @staticmethod
    def service_level(traffic, agents, aht, awt):
        forecast_rate = traffic / aht if aht else 0
        return service_level_erlang_c(forecast_rate, aht, agents, awt)

    @staticmethod
    def required_agents(traffic, aht, sl_target, awt):
        forecast_rate = traffic / aht if aht else 0
        return int(X.AGENTS.for_sla(sl_target, forecast_rate, aht, awt))
class BL:
    """Módulo Blending"""


    @staticmethod
    def sla(forecast, aht, agents, awt, lines, patience, threshold):
        available_agents = max(0, agents - threshold)
        if available_agents <= 0:
            return 0.0
        return service_level_erlang_c(forecast, aht, available_agents, awt)
    
    @staticmethod
    def outbound_capacity(forecast, aht, agents, lines, patience, threshold, outbound_aht):
        inbound_traffic = forecast * aht
        inbound_agents_needed = inbound_traffic + threshold
        outbound_agents = max(0, agents - inbound_agents_needed)
        return max(0, outbound_agents / outbound_aht)
    
    @staticmethod
    def optimal_threshold(forecast, aht, agents, awt, lines, patience, sl_target):
        def objective(threshold):
            if threshold < 0 or threshold > agents:
                return float('inf')
            sl = BL.sla(forecast, aht, agents, awt, lines, patience, threshold)
            return abs(sl - sl_target)
        
        result = optimize.minimize_scalar(objective, bounds=(0, agents), method='bounded')
        return max(0, round(result.x, 1))
    @staticmethod
    def sensitivity(traffic_range, agents, aht, target):
        data = []
        for t in traffic_range:
            sl = service_level_erlang_c(t, aht, agents, target)
            data.append({"traffic": t, "service_level": sl})
        return pd.DataFrame(data)

    @staticmethod
    def monte_carlo(traffic, agents, aht, target, iters=100):
        results = [service_level_erlang_c(traffic, aht, agents, target) for _ in range(iters)]
        return pd.Series(results)


# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">📞 Erlang Calculator Pro</h1>', unsafe_allow_html=True)
    st.markdown("**Calculadora completa de Erlang para Centros de Contacto**")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("🔧 Configuración")
    
    # Selección de módulo
    module = st.sidebar.selectbox(
        "📊 Seleccionar Módulo",
        [
            "Erlang C/X",
            "Erlang C/X Visual",
            "Chat Multi-canal",
            "Blending",
            "Erlang O (Outbound)",
            "Análisis Comparativo",
            "Staffing Optimizer",
        ]
    )
    
    if module == "Erlang C/X":
        erlang_x_interface()
    elif module == "Erlang C/X Visual":
        enhanced_erlang_x_interface()
    elif module == "Chat Multi-canal":
        chat_interface()
    elif module == "Blending":
        blending_interface()
    elif module == "Erlang O (Outbound)":
        erlang_o_interface()
    elif module == "Análisis Comparativo":
        comparative_analysis()
    elif module == "Staffing Optimizer":
        staffing_optimizer()

    # Display methodology and formulas
    show_methodology()

def erlang_x_interface():
    st.header("📈 Erlang C/X Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Parámetros de Entrada")
        forecast = st.number_input("Forecast (llamadas por intervalo)", min_value=1.0, value=100.0, step=1.0)
        interval_choice = st.selectbox("Intervalo del forecast", ["30 minutos", "1 hora"], index=1)
        interval_seconds = 1800 if interval_choice == "30 minutos" else 3600
        aht = st.number_input("AHT (segundos)", min_value=1.0, value=240.0, step=1.0)
        agents = int(st.number_input("Agentes", min_value=1, value=25, step=1))
        awt = st.number_input("AWT (segundos)", min_value=1.0, value=20.0, step=1.0)
        
        # Parámetros opcionales
        st.subheader("🔧 Parámetros Avanzados")
        use_advanced = st.checkbox("Usar Erlang X (con abandonment)")
        
        lines = None
        patience = None
        
        if use_advanced:
            lines = st.number_input("Líneas disponibles", min_value=int(agents), value=int(agents*1.2), step=1)
            patience = st.number_input("Patience (segundos)", min_value=1.0, value=120.0, step=1.0)
    
    with col2:
        st.subheader("📊 Resultados")
        
        # Calcular métricas
        arrival_rate = forecast / interval_seconds
        sl = X.SLA.calculate(arrival_rate, aht, agents, awt, lines, patience)
        asa = X.asa(arrival_rate, aht, agents)
        occ = X.occupancy(arrival_rate, aht, agents)
        
        # Mostrar métricas
        sl_class = "success-metric" if sl >= 0.8 else "warning-metric" if sl >= 0.7 else "danger-metric"
        asa_class = "success-metric" if asa <= 30 else "warning-metric" if asa <= 60 else "danger-metric"
        occ_class = "success-metric" if 0.7 <= occ <= 0.85 else "warning-metric"
        
        st.markdown(f"""
        <div class="metric-card {sl_class}">
            <h3>Service Level</h3>
            <h2>{sl:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card {asa_class}">
            <h3>ASA (Average Speed of Answer)</h3>
            <h2>{asa:.2f} seg</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card {occ_class}">
            <h3>Ocupación</h3>
            <h2>{occ:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if use_advanced and lines and patience:
            abandon_rate = X.abandonment(arrival_rate, aht, agents, lines, patience)
            abandon_class = "success-metric" if abandon_rate <= 0.05 else "warning-metric" if abandon_rate <= 0.1 else "danger-metric"
            
            st.markdown(f"""
            <div class="metric-card {abandon_class}">
                <h3>Abandonment Rate</h3>
                <h2>{abandon_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Análisis de dimensionamiento
    st.subheader("🎯 Análisis de Dimensionamiento")
    
    target_sl = st.slider("Service Level Objetivo", 0.7, 0.95, 0.8, 0.01)
    recommended_agents = X.AGENTS.for_sla(target_sl, arrival_rate, aht, awt, lines, patience)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Agentes Recomendados", f"{recommended_agents}")
    col2.metric("Agentes Actuales", f"{agents}")
    col3.metric("Diferencia", f"{recommended_agents - agents:+}")
    
    # Gráfico de sensibilidad
    st.subheader("📈 Análisis de Sensibilidad")
    
    agent_range = range(max(1, int(recommended_agents * 0.7)), int(recommended_agents * 1.5))
    sl_data = []
    asa_data = []
    occ_data = []
    
    for a in agent_range:
        sl_val = X.SLA.calculate(arrival_rate, aht, a, awt, lines, patience)
        asa_val = X.asa(arrival_rate, aht, a)
        occ_val = X.occupancy(arrival_rate, aht, a)
        
        sl_data.append(sl_val)
        asa_data.append(asa_val)
        occ_data.append(occ_val)
    
    # Crear gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(agent_range),
        y=sl_data,
        mode='lines+markers',
        name='Service Level',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(agent_range),
        y=asa_data,
        mode='lines+markers',
        name='ASA (seg)',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Service Level y ASA vs Número de Agentes",
        xaxis_title="Número de Agentes",
        yaxis=dict(title="Service Level", side="left", range=[0, 1]),
        yaxis2=dict(title="ASA (segundos)", side="right", overlaying="y"),
        hovermode='x unified'
    )

    # Líneas de referencia
    fig.add_vline(x=agents, line_dash="dash", line_color="red", annotation_text="Actual")
    fig.add_vline(x=recommended_agents, line_dash="dash", line_color="orange", annotation_text="Recomendado")
    
    st.plotly_chart(fig, use_container_width=True)


def enhanced_erlang_x_interface():
    """Versión con visualización dinámica de agentes"""

    st.header("📈 Erlang C/X Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Parámetros de Entrada")
        forecast = st.number_input("Forecast (llamadas por intervalo)", min_value=1.0, value=100.0, step=1.0)
        interval_choice = st.selectbox("Intervalo del forecast", ["30 minutos", "1 hora"], index=1)
        interval_seconds = 1800 if interval_choice == "30 minutos" else 3600
        aht = st.number_input("AHT (segundos)", min_value=1.0, value=240.0, step=1.0)
        agents = int(st.number_input("Agentes", min_value=1, value=25, step=1))
        awt = st.number_input("AWT (segundos)", min_value=1.0, value=20.0, step=1.0)

        st.subheader("🔧 Parámetros Avanzados")
        use_advanced = st.checkbox("Usar Erlang X (con abandonment)")

        lines = None
        patience = None

        if use_advanced:
            lines = st.number_input("Líneas disponibles", min_value=int(agents), value=int(agents * 1.2), step=1)
            patience = st.number_input("Patience (segundos)", min_value=1.0, value=120.0, step=1.0)

    with col2:
        st.subheader("📊 Resultados")

        arrival_rate = forecast / interval_seconds
        sl = X.SLA.calculate(arrival_rate, aht, agents, awt, lines, patience)
        asa = X.asa(arrival_rate, aht, agents)
        occ = X.occupancy(arrival_rate, aht, agents)

        st.metric("Service Level", f"{sl:.1%}")
        st.metric("ASA", f"{asa:.2f} seg")
        st.metric("Ocupación", f"{occ:.1%}")

    st.subheader("🎬 Visualización en Tiempo Real")

    agent_viz, status_bar = create_real_time_dashboard(forecast, aht, agents, awt, interval_seconds)
    st.plotly_chart(agent_viz, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(status_bar, use_container_width=True)
    with col2:
        st.markdown(
            """
            ### 🎨 Leyenda de Estados

            **Agentes:**
            - 🔴 **Ocupado**
            - 🟢 **Disponible**

            **Cola:**
            - ⏳ **Esperando**
            - 🟢 **Cola Corta**
            - 🟡 **Cola Moderada**
            - 🔴 **Cola Larga**

            **Métricas:**
            - 🟢 **Bueno**
            - 🟡 **Aceptable**
            - 🔴 **Crítico**
            """
        )

    st.subheader("🎮 Controles de Simulación")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Aumentar Demanda (+20%)"):
            forecast_sim = forecast * 1.2
            viz, _ = create_real_time_dashboard(forecast_sim, aht, agents, awt, interval_seconds)
            st.plotly_chart(viz, use_container_width=True)

    with col2:
        if st.button("👥 Agregar 5 Agentes"):
            agents_sim = agents + 5
            viz, _ = create_real_time_dashboard(forecast, aht, agents_sim, awt, interval_seconds)
            st.plotly_chart(viz, use_container_width=True)

    with col3:
        if st.button("⏱️ Reducir AHT (-30s)"):
            aht_sim = max(60, aht - 30)
            viz, _ = create_real_time_dashboard(forecast, aht_sim, agents, awt, interval_seconds)
            st.plotly_chart(viz, use_container_width=True)

def erlang_o_interface():
    st.header("📞 Erlang O - Outbound Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Parámetros Outbound")
        agents_out = st.number_input("Agentes Outbound", min_value=1, value=20, step=1)
        hours_per_day = st.number_input("Horas por día", min_value=1.0, value=8.0, step=0.5)
        calls_per_hour = st.number_input("Llamadas por hora por agente", min_value=1.0, value=25.0, step=1.0)
        success_rate = st.slider("Tasa de éxito (contactos efectivos)", 0.1, 0.8, 0.3, 0.01)
        
        st.subheader("🎯 Configuración de Objetivos")
        target_daily_calls = st.number_input("Objetivo llamadas exitosas/día", min_value=1, value=500, step=10)
        
        st.subheader("🤖 Predictive Dialer")
        answer_rate = st.slider("Tasa de respuesta", 0.1, 0.5, 0.25, 0.01)
        talk_time = st.number_input("Tiempo promedio conversación (min)", min_value=1.0, value=5.0, step=0.5)
        wait_between = st.number_input("Tiempo entre llamadas (min)", min_value=0.5, value=2.0, step=0.5)
    
    with col2:
        st.subheader("📊 Resultados Outbound")
        
        # Calcular productividad
        productivity = ERLANG_O.productivity(agents_out, hours_per_day, calls_per_hour, success_rate)
        agents_needed = ERLANG_O.agents_for_target(target_daily_calls, hours_per_day, calls_per_hour, success_rate)
        dialer_ratio = ERLANG_O.dialer_ratio(answer_rate, talk_time, wait_between)
        
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>Llamadas Totales/Día</h3>
            <h2>{productivity['total_calls']:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Llamadas Exitosas/Día</h3>
            <h2>{productivity['successful_calls']:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Productividad/Agente/Día</h3>
            <h2>{productivity['successful_per_agent_day']:.1f} exitosas</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h3>Agentes Necesarios (Objetivo)</h3>
            <h2>{agents_needed}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Ratio Predictive Dialer</h3>
            <h2>{dialer_ratio:.2f}:1</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Análisis de ROI
    st.subheader("💰 Análisis de ROI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cost_per_agent = st.number_input("Costo/agente/día ($)", min_value=1.0, value=150.0, step=10.0)
        revenue_per_success = st.number_input("Ingreso/llamada exitosa ($)", min_value=1.0, value=50.0, step=5.0)
    
    with col2:
        total_cost = agents_out * cost_per_agent
        total_revenue = productivity['successful_calls'] * revenue_per_success
        profit = total_revenue - total_cost
        roi = (profit / total_cost) * 100 if total_cost > 0 else 0
        
        st.metric("Costo Total/Día", f"${total_cost:,.0f}")
        st.metric("Ingresos/Día", f"${total_revenue:,.0f}")
    
    with col3:
        profit_color = "success-metric" if profit > 0 else "danger-metric"
        
        st.markdown(f"""
        <div class="metric-card {profit_color}">
            <h3>Ganancia/Día</h3>
            <h2>${profit:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>ROI</h3>
            <h2>{roi:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de productividad vs agentes
    st.subheader("📈 Productividad vs Número de Agentes")
    
    agent_range_out = range(1, 51)
    calls_data = []
    success_data = []
    profit_data = []
    
    for a in agent_range_out:
        prod = ERLANG_O.productivity(a, hours_per_day, calls_per_hour, success_rate)
        cost = a * cost_per_agent
        revenue = prod['successful_calls'] * revenue_per_success
        profit_val = revenue - cost
        
        calls_data.append(prod['total_calls'])
        success_data.append(prod['successful_calls'])
        profit_data.append(profit_val)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(agent_range_out),
        y=success_data,
        mode='lines+markers',
        name='Llamadas Exitosas/Día',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(agent_range_out),
        y=profit_data,
        mode='lines+markers',
        name='Ganancia/Día ($)',
        yaxis='y2',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="Productividad y Ganancia vs Número de Agentes",
        xaxis_title="Número de Agentes",
        yaxis=dict(title="Llamadas Exitosas/Día", side="left"),
        yaxis2=dict(title="Ganancia/Día ($)", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    fig.add_vline(x=agents_out, line_dash="dash", line_color="red", annotation_text="Actual")
    fig.add_vline(x=agents_needed, line_dash="dash", line_color="orange", annotation_text="Objetivo")
    
    st.plotly_chart(fig, use_container_width=True)

def comparative_analysis():
    st.header("⚖️ Análisis Comparativo de Modelos")
    
    # Parámetros comunes
    st.subheader("📝 Parámetros Base")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_comp = st.number_input("Forecast común (llamadas por intervalo)", min_value=1.0, value=150.0, step=1.0)
        interval_choice_comp = st.selectbox("Intervalo del forecast", ["30 minutos", "1 hora"], index=1, key="interval_comp")
        interval_seconds_comp = 1800 if interval_choice_comp == "30 minutos" else 3600
        aht_comp = st.number_input("AHT común (seg)", min_value=1.0, value=240.0, step=1.0)
    
    with col2:
        agents_comp = int(st.number_input("Agentes común", min_value=1, value=30, step=1))
        awt_comp = st.number_input("AWT común (seg)", min_value=1.0, value=20.0, step=1.0)
    
    with col3:
        lines_comp = int(agents_comp * 1.2)
        patience_comp = 180.0
        st.metric("Líneas", lines_comp)
        st.metric("Patience (seg)", patience_comp)
    
    # Comparación de resultados
    st.subheader("📊 Comparación de Resultados")
    
    # Erlang C básico
    arrival_rate_comp = forecast_comp / interval_seconds_comp
    sl_basic = X.SLA.calculate(arrival_rate_comp, aht_comp, agents_comp, awt_comp)
    asa_basic = X.asa(arrival_rate_comp, aht_comp, agents_comp)
    occ_basic = X.occupancy(arrival_rate_comp, aht_comp, agents_comp)
    
    # Erlang X con abandonment
    sl_abandon = X.SLA.calculate(arrival_rate_comp, aht_comp, agents_comp, awt_comp, lines_comp, patience_comp)
    abandon_rate = X.abandonment(arrival_rate_comp, aht_comp, agents_comp, lines_comp, patience_comp)
    
    # Chat modelo
    chat_aht_comp = [aht_comp * 0.7, aht_comp * 0.8, aht_comp * 0.9]
    sl_chat = CHAT.sla(arrival_rate_comp, chat_aht_comp, agents_comp, awt_comp, lines_comp, patience_comp)
    asa_chat = CHAT.asa(arrival_rate_comp, chat_aht_comp, agents_comp, lines_comp, patience_comp)
    
    # Blending modelo
    threshold_comp = 3
    sl_blend = BL.sla(arrival_rate_comp, aht_comp, agents_comp, awt_comp, lines_comp, patience_comp, threshold_comp)
    outbound_cap = BL.outbound_capacity(arrival_rate_comp, aht_comp, agents_comp, lines_comp, patience_comp, threshold_comp, aht_comp)
    
    # Crear tabla comparativa
    comparison_data = {
        'Modelo': ['Erlang C', 'Erlang X', 'Chat Multi-canal', 'Blending'],
        'Service Level': [f"{sl_basic:.1%}", f"{sl_abandon:.1%}", f"{sl_chat:.1%}", f"{sl_blend:.1%}"],
        'ASA (seg)': [f"{asa_basic:.2f}", f"{asa_basic:.2f}", f"{asa_chat:.2f}", f"{asa_basic:.2f}"],
        'Ocupación': [f"{occ_basic:.1%}", f"{occ_basic:.1%}", f"{occ_basic:.1%}", f"{occ_basic:.1%}"],
        'Características': [
            'Modelo básico, sin abandonment',
            f'Con abandonment ({abandon_rate:.1%})',
            f'Multi-chat ({len(chat_aht_comp)} simultáneos)',
            f'Outbound: {outbound_cap:.0f} llamadas/h'
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Gráfico comparativo
    models = comparison_data['Modelo']
    sl_values = [sl_basic, sl_abandon, sl_chat, sl_blend]
    
    fig = px.bar(
        x=models,
        y=sl_values,
        title="Comparación de Service Level por Modelo",
        labels={'x': 'Modelo', 'y': 'Service Level'},
        color=sl_values,
        color_continuous_scale='Blues'
    )
    
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Objetivo 80%")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones
    st.subheader("🎯 Recomendaciones")
    
    best_sl = max(sl_values)
    best_model = models[sl_values.index(best_sl)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Mejor Service Level:** {best_model} ({best_sl:.1%})")
        
        if sl_chat > sl_basic:
            st.info("💬 **Chat Multi-canal** mejora la eficiencia permitiendo múltiples conversaciones simultáneas")
        
        if sl_abandon < sl_basic:
            st.warning("⚠️ **Erlang X** muestra el impacto real del abandonment en el service level")
    
    with col2:
        st.info(f"🔄 **Blending** permite capacidad outbound adicional: {outbound_cap:.0f} llamadas/hora")
        
        if any(sl < 0.8 for sl in sl_values):
            st.error("❌ Algunos modelos no alcanzan el objetivo del 80%")
        else:
            st.success("✅ Todos los modelos superan el objetivo del 80%")

def staffing_optimizer():
    st.header("📅 Staffing Optimizer")
    
    # Configuración de horarios
    st.subheader("⏰ Configuración de Horarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_hour = st.selectbox("Hora inicio", range(0, 24), index=8)
        end_hour = st.selectbox("Hora fin", range(start_hour + 1, 25), index=20)
        aht_staff = st.number_input("AHT (segundos)", min_value=1.0, value=240.0, step=1.0, key="aht_staff")
        interval_seconds_staff = st.number_input("Duración del intervalo (segundos)", min_value=1.0, value=3600.0, step=1.0, key="int_staff")
        target_sl_staff = st.slider("Service Level objetivo", 0.7, 0.95, 0.8, 0.01, key="sl_staff")
    
    with col2:
        st.subheader("📈 Patrón de Demanda")
        pattern_type = st.selectbox("Tipo de patrón", ["Manual", "Típico Call Center", "E-commerce", "Soporte Técnico"])
    
    # Generar forecast por horas
    hours = list(range(start_hour, end_hour))
    
    if pattern_type == "Manual":
        st.subheader("📝 Ingreso Manual de Forecast")
        forecasts = []
        cols = st.columns(4)
        for i, hour in enumerate(hours):
            with cols[i % 4]:
                forecast_val = st.number_input(f"{hour:02d}:00", min_value=1.0, value=100.0, step=1.0, key=f"hour_{hour}")
                forecasts.append(forecast_val)
    
    else:
        # Patrones predefinidos
        if pattern_type == "Típico Call Center":
            # Pico en la mañana y tarde
            base_forecast = 80
            pattern = [0.6, 0.8, 1.0, 1.2, 1.1, 0.9, 0.8, 1.0, 1.3, 1.1, 0.9, 0.7]
        elif pattern_type == "E-commerce":
            # Más actividad en la tarde/noche
            base_forecast = 120
            pattern = [0.5, 0.7, 0.9, 1.1, 1.3, 1.2, 1.0, 0.8, 0.9, 1.1, 1.4, 1.2]
        else:  # Soporte Técnico
            # Distribución más uniforme
            base_forecast = 90
            pattern = [0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9]
        
        forecasts = [base_forecast * pattern[i % len(pattern)] for i in range(len(hours))]
        
        # Mostrar patrón
        fig_pattern = px.line(
            x=hours,
            y=forecasts,
            title=f"Patrón de Demanda - {pattern_type}",
            labels={'x': 'Hora', 'y': 'Forecast (llamadas/hora)'}
        )
        st.plotly_chart(fig_pattern, use_container_width=True)
    
    # Calcular staffing
    st.subheader("👥 Resultado de Staffing")
    
    staffing_results = []
    total_agent_hours = 0
    
    for hour, forecast in zip(hours, forecasts):
        arrival_rate_staff = forecast / interval_seconds_staff
        agents_needed = X.AGENTS.for_sla(target_sl_staff, arrival_rate_staff, aht_staff, 20)
        sl_achieved = X.SLA.calculate(arrival_rate_staff, aht_staff, agents_needed, 20)
        asa_achieved = X.asa(arrival_rate_staff, aht_staff, agents_needed)
        
        staffing_results.append({
            'Hora': f"{hour:02d}:00",
            'Forecast': f"{forecast:.0f}",
            'Agentes': agents_needed,
            'SL': f"{sl_achieved:.1%}",
            'ASA': f"{asa_achieved:.1f} seg"
        })
        
        total_agent_hours += agents_needed
    
    df_staffing = pd.DataFrame(staffing_results)
    st.dataframe(df_staffing, use_container_width=True)
    
    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    
    max_agents = max([r['Agentes'] for r in staffing_results])
    min_agents = min([r['Agentes'] for r in staffing_results])
    avg_agents = total_agent_hours / len(hours)
    
    col1.metric("Agentes Pico", max_agents)
    col2.metric("Agentes Valle", min_agents)
    col3.metric("Promedio", f"{avg_agents:.1f}")
    col4.metric("Total Agente-Horas", total_agent_hours)
    
    # Gráfico de staffing
    fig_staffing = go.Figure()
    
    fig_staffing.add_trace(go.Scatter(
        x=hours,
        y=forecasts,
        mode='lines+markers',
        name='Forecast',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    agents_values = [r['Agentes'] for r in staffing_results]
    fig_staffing.add_trace(go.Scatter(
        x=hours,
        y=agents_values,
        mode='lines+markers',
        name='Agentes Necesarios',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig_staffing.update_layout(
        title="Forecast vs Agentes Necesarios por Hora",
        xaxis_title="Hora del Día",
        yaxis=dict(title="Forecast (llamadas/hora)", side="left"),
        yaxis2=dict(title="Agentes Necesarios", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_staffing, use_container_width=True)
    
    # Análisis de turnos
    st.subheader("🔄 Análisis de Turnos")
    
    shift_analysis = st.checkbox("Realizar análisis de turnos")
    
    if shift_analysis:
        shift_duration = st.selectbox("Duración del turno (horas)", [4, 6, 8], index=2)
        
        # Calcular turnos óptimos
        shift_starts = []
        for start in range(start_hour, end_hour - shift_duration + 1):
            shift_end = start + shift_duration
            shift_hours = list(range(start, shift_end))
            shift_forecasts = [forecasts[h - start_hour] for h in shift_hours if h - start_hour < len(forecasts)]
            
            if shift_forecasts:
                max_forecast = max(shift_forecasts)
                arrival_rate_shift = max_forecast / interval_seconds_staff
                agents_for_shift = X.AGENTS.for_sla(target_sl_staff, arrival_rate_shift, aht_staff, 20)
                
                shift_starts.append({
                    'Turno': f"{start:02d}:00 - {shift_end:02d}:00",
                    'Agentes': agents_for_shift,
                    'Max_Forecast': f"{max_forecast:.0f}",
                    'Cobertura': len(shift_forecasts)
                })
        
        df_shifts = pd.DataFrame(shift_starts)
        st.dataframe(df_shifts, use_container_width=True)
        
        # Recomendación de turnos
        optimal_shifts = df_shifts.nsmallest(3, 'Agentes')
        st.subheader("🎯 Turnos Recomendados")
        st.dataframe(optimal_shifts, use_container_width=True)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def show_methodology():
    with st.expander("📚 Metodología y Fórmulas"):
        st.markdown("""
        ### 🧮 Fórmulas Utilizadas
        
        **Erlang B (Probabilidad de Bloqueo):**
        ```
        B(A,N) = (A^N / N!) / Σ(k=0 to N)[A^k / k!]
        ```
        
        **Erlang C (Probabilidad de Espera):**
        ```
        C(A,N) = [A^N / N!] / [Σ(k=0 to N-1)[A^k / k!] + (A^N / N!) * N/(N-A)]
        ```
        
        **Service Level:**
        ```
        SL = 1 - C * e^(-(N-A)*t/AHT)
        ```
        
        **ASA (Average Speed of Answer):**
        ```
        ASA = C * AHT / (N - A)
        ```
        
        ### 📊 Modelos Implementados
        
        - **Erlang C**: Modelo básico sin abandonment
        - **Erlang X**: Incluye abandonment y retrials
        - **Chat Multi-canal**: Agentes manejan múltiples conversaciones
        - **Blending**: Combinación inbound/outbound
        - **Erlang O**: Campañas outbound puras
        
        ### 🎯 Interpretación de Métricas
        
        - **Service Level**: % de llamadas atendidas dentro del AWT objetivo
        - **ASA**: Tiempo promedio de espera en cola
        - **Ocupación**: % del tiempo que los agentes están ocupados
        - **Abandonment**: % de clientes que cuelgan antes de ser atendidos
        """)


def chat_interface():
    st.header("💬 Chat Multi-canal Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Parámetros Chat")
        forecast = st.number_input("Chats por intervalo", min_value=1.0, value=200.0, step=1.0)
        interval_choice_chat = st.selectbox("Intervalo del forecast", ["30 minutos", "1 hora"], index=1, key="interval_chat")
        interval_seconds_chat = 1800 if interval_choice_chat == "30 minutos" else 3600
        
        st.subheader("⏱️ AHT por Número de Chats Simultáneos")
        max_chats = st.selectbox("Máximo chats simultáneos por agente", [1, 2, 3, 4, 5], index=2)
        
        aht_list = []
        for i in range(max_chats):
            aht = st.number_input(f"AHT para {i+1} chat(s) (seg)", min_value=1.0, value=120.0 + i*30.0, step=1.0, key=f"aht_{i}")
            aht_list.append(aht)
        
        agents = int(st.number_input("Agentes Chat", min_value=1, value=15, step=1))
        awt = st.number_input("AWT Chat (segundos)", min_value=1.0, value=30.0, step=1.0)
        lines = st.number_input("Líneas Chat", min_value=int(agents), value=300, step=1)
        patience = st.number_input("Patience Chat (segundos)", min_value=1.0, value=180.0, step=1.0)
    
    with col2:
        st.subheader("📊 Resultados Chat")
        
        # Calcular métricas chat
        arrival_rate_chat = forecast / interval_seconds_chat
        chat_sl = CHAT.sla(arrival_rate_chat, aht_list, agents, awt, lines, patience)
        chat_asa = CHAT.asa(arrival_rate_chat, aht_list, agents, lines, patience)
        
        # Métricas específicas del chat
        parallel_capacity = len(aht_list)
        avg_aht = sum(aht_list) / len(aht_list)
        effectiveness = 0.7 + (0.3 / parallel_capacity)
        chats_per_agent_hour = forecast / agents
        
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>Service Level Chat</h3>
            <h2>{chat_sl:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>ASA Chat</h3>
            <h2>{chat_asa:.2f} seg</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Chats Simultáneos Máx</h3>
            <h2>{parallel_capacity}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Eficiencia</h3>
            <h2>{chats_per_agent_hour:.1f} chats/agente/hora</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparación de configuraciones
    st.subheader("⚖️ Comparación de Configuraciones")
    
    configs = []
    for max_simultaneous in range(1, 6):
        test_aht = [120.0 + i*24.0 for i in range(max_simultaneous)]
        test_agents = CHAT.agents_for_sla(0.85, arrival_rate_chat, test_aht, awt, lines, patience)
        test_sl = CHAT.sla(arrival_rate_chat, test_aht, test_agents, awt, lines, patience)
        efficiency = forecast / test_agents
        
        configs.append({
            'Chats Simultáneos': max_simultaneous,
            'Agentes Necesarios': test_agents,
            'Service Level': f"{test_sl:.1%}",
            'Eficiencia': f"{efficiency:.1f} chats/agente/hora",
            'AHT Promedio': f"{sum(test_aht)/len(test_aht):.1f} seg"
        })
    
    df_configs = pd.DataFrame(configs)
    st.dataframe(df_configs, use_container_width=True)
    
    # Gráfico de eficiencia
    fig = px.bar(df_configs, x='Chats Simultáneos', y='Agentes Necesarios', 
                 title="Agentes Necesarios vs Chats Simultáneos",
                 color='Agentes Necesarios', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

def blending_interface():
    st.header("🔄 Blending Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Parámetros Blending")
        inbound_forecast = st.number_input("Forecast Inbound (llamadas por intervalo)", min_value=1.0, value=120.0, step=1.0)
        interval_choice_blend = st.selectbox("Intervalo del forecast", ["30 minutos", "1 hora"], index=1, key="interval_blend")
        interval_seconds_blend = 1800 if interval_choice_blend == "30 minutos" else 3600
        inbound_aht = st.number_input("AHT Inbound (segundos)", min_value=1.0, value=210.0, step=1.0)
        outbound_aht = st.number_input("AHT Outbound (segundos)", min_value=1.0, value=300.0, step=1.0)
        total_agents = int(st.number_input("Total Agentes", min_value=1, value=30, step=1))
        awt = st.number_input("AWT (segundos)", min_value=1.0, value=20.0, step=1.0)
        threshold = st.number_input("Threshold (agentes reservados)", min_value=0.0, value=3.0, step=1.0, max_value=total_agents)
        
        lines = st.number_input("Líneas", min_value=int(total_agents), value=int(total_agents*1.2), step=1)
        patience = st.number_input("Patience (segundos)", min_value=1.0, value=300.0, step=1.0)
    
    with col2:
        st.subheader("📊 Resultados Blending")
        
        # Calcular métricas blending
        arrival_rate_blend = inbound_forecast / interval_seconds_blend
        bl_sl = BL.sla(arrival_rate_blend, inbound_aht, total_agents, awt, lines, patience, threshold)
        outbound_capacity = BL.outbound_capacity(arrival_rate_blend, inbound_aht, total_agents, lines, patience, threshold, outbound_aht)

        available_for_inbound = total_agents - threshold
        inbound_occupancy = occupancy_erlang_c(arrival_rate_blend, inbound_aht, available_for_inbound)
        
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>Service Level Inbound</h3>
            <h2>{bl_sl:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Capacidad Outbound</h3>
            <h2>{outbound_capacity:.1f} llamadas/hora</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Agentes Disponibles Inbound</h3>
            <h2>{available_for_inbound:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Ocupación Inbound</h3>
            <h2>{inbound_occupancy:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Optimización de threshold
    st.subheader("🎯 Optimización de Threshold")
    
    target_sl_blend = st.slider("Service Level Objetivo Blending", 0.7, 0.95, 0.8, 0.01)
    optimal_threshold = BL.optimal_threshold(arrival_rate_blend, inbound_aht, total_agents, awt, lines, patience, target_sl_blend)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Threshold Óptimo", f"{optimal_threshold}")
    col2.metric("Threshold Actual", f"{threshold}")
    col3.metric("Diferencia", f"{optimal_threshold - threshold:+}")
    
    # Análisis de threshold
    st.subheader("📈 Análisis de Threshold")
    
    threshold_range = range(0, int(total_agents * 0.4))
    sl_blend_data = []
    outbound_data = []
    
    for t in threshold_range:
        sl_val = BL.sla(arrival_rate_blend, inbound_aht, total_agents, awt, lines, patience, t)
        out_val = BL.outbound_capacity(arrival_rate_blend, inbound_aht, total_agents, lines, patience, t, outbound_aht)
        
        sl_blend_data.append(sl_val)
        outbound_data.append(out_val)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(threshold_range),
        y=sl_blend_data,
        mode='lines+markers',
        name='Service Level Inbound',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(threshold_range),
        y=outbound_data,
        mode='lines+markers',
        name='Capacidad Outbound',
        yaxis='y2',
        line=dict(color='green')
    ))
    fig.update_layout(
        title="Service Level vs Capacidad Outbound por Threshold",
        xaxis_title="Threshold (Agentes Reservados)",
        yaxis=dict(title="Service Level Inbound", side="left", range=[0, 1]),
        yaxis2=dict(title="Capacidad Outbound (llamadas/hora)", side="right", overlaying="y"),
        hovermode='x unified'
    )

    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Actual")
    fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="orange", annotation_text="Óptimo")

    st.plotly_chart(fig, use_container_width=True)



def run_app():
    main()

if __name__ == "__main__":
    run_app()

