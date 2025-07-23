import numpy as np
import pandas as pd
from erlang_calculator import X, CHAT, BL


def run_app():
    import streamlit as st

    st.title("Multi-Chat Erlang Calculator")

    num_chats = int(st.sidebar.number_input("Number of Chat Queues", value=2, min_value=1, step=1))
    chats = []
    for i in range(num_chats):
        with st.sidebar.expander(f"Chat {i+1} Inputs"):
            traffic = st.number_input(f"Traffic Intensity {i+1}", value=5.0, key=f"traffic_{i}")
            agents = st.number_input(f"Agents {i+1}", value=5, step=1, key=f"agents_{i}")
            concurrency = st.number_input(f"Chat Concurrency {i+1}", value=1.0, min_value=1.0, step=1.0, key=f"conc_{i}")
            chats.append({'traffic': traffic, 'agents': agents, 'concurrency': concurrency})

    aht = st.sidebar.number_input("Average Handle Time (seconds)", value=180)
    target = st.sidebar.number_input("Service Level Target (seconds)", value=20)

    if st.sidebar.button("Compute"):
        rows = []
        for idx, p in enumerate(chats):
            b = X.erlang_b(p['traffic'] / p['concurrency'], p['agents'])
            c = X.erlang_c(p['traffic'] / p['concurrency'], p['agents'])
            sl = CHAT.service_level_multi(p['traffic'], p['agents'], aht, target, p['concurrency'])
            asa = CHAT.asa_multi(p['traffic'], p['agents'], aht, p['concurrency'])
            rows.append({
                'Chat': idx + 1,
                'Blocking': round(b, 4),
                'Waiting': round(c, 4),
                'Service Level': round(sl, 4),
                'ASA': round(asa, 2)
            })
        st.write("### Results")
        st.table(pd.DataFrame(rows).set_index('Chat'))

    st.sidebar.header("Sensitivity Analysis")
    if st.sidebar.button("Run Sensitivity"):
        for idx, p in enumerate(chats):
            tf_range = np.linspace(max(0.1, p['traffic'] - 3), p['traffic'] + 3, 20)
            df = BL.sensitivity_multi(tf_range, p['agents'], aht, target, p['concurrency'])
            st.subheader(f"Chat {idx+1}")
            st.line_chart(df.set_index('traffic'))

    st.sidebar.header("Monte Carlo")
    iters = st.sidebar.number_input("Iterations", value=500, step=100)
    if st.sidebar.button("Run Monte Carlo"):
        for idx, p in enumerate(chats):
            series = BL.monte_carlo_multi(p['traffic'], p['agents'], aht, target, p['concurrency'], int(iters))
            st.subheader(f"Chat {idx+1}")
            st.write(series.describe())
            counts, bins = np.histogram(series, bins=20)
            st.bar_chart(pd.DataFrame({'count': counts}, index=bins[:-1]))

    st.sidebar.header("Staffing Optimization")
    svc_target = st.sidebar.slider("Target Service Level", 0.5, 0.99, 0.8)
    if st.sidebar.button("Optimize Staffing"):
        for idx, p in enumerate(chats):
            agents = CHAT.required_agents_multi(p['traffic'], aht, svc_target, target, p['concurrency'])
            st.write(f"Chat {idx+1} Required Agents: {agents}")

if __name__ == "__main__":
    run_app()
