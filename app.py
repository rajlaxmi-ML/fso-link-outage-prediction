import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

import os
import joblib


# Load pre-trained model
model = joblib.load('fso_rf_model.pkl')

# Fallback dataset loading
if os.path.exists('LCD_sample_cvs.csv'):
    data = pd.read_csv('LCD_sample_cvs.csv')

# Page Configuration
st.set_page_config(page_title="FSO Live Telemetry EWS", page_icon="📡", layout="wide")

st.title("📡 Live Animated FSO Early Warning System")
st.markdown("### Real-Time Optical Link Telemetry & Predictive Outage Monitoring")

# Sidebar Controls
st.sidebar.header("⚙️ System Controls")
P_tx = st.sidebar.slider("Transmit Power (dBm)", 0.0, 30.0, 10.0, 0.5)
L = st.sidebar.slider("Link Distance (km)", 0.1, 5.0, 1.0, 0.1)
Rx_threshold = st.sidebar.slider("Sensitivity Threshold (dBm)", -30.0, 15.0, 9.88, 0.1)
alert_cutoff = st.sidebar.slider("Predictive Alert Cutoff (%)", 10, 90, 40, 5) / 100.0

sim_speed = st.sidebar.slider("Animation Speed (sec/frame)", 0.05, 0.5, 0.1, 0.05)
start_sim = st.sidebar.button("▶️ Start Live Telemetry Simulation", type="primary")

@st.cache_data
def prepare_data():
    df_raw = pd.read_csv('LCD_sample_csv.csv')
    required_cols = ['DATE', 'HourlyVisibility', 'HourlyDryBulbTemperatureC', 'HourlyWindSpeed']
    df_clean = df_raw[required_cols].dropna().reset_index(drop=True)

    df_clean['Visibility_km'] = df_clean['HourlyVisibility'] * 1.60934
    df_clean['Temperature_C'] = df_clean['HourlyDryBulbTemperatureC']
    df_clean['Wind_Speed_m_s'] = df_clean['HourlyWindSpeed'] * 0.44704
    return df_clean

df_clean = prepare_data().copy()

# Physics Computations
v = df_clean['Visibility_km']
conditions = [(v > 50), (v > 6) & (v <= 50), (v > 1) & (v <= 6), (v <= 1)]
choices = [1.6, 1.3, 0.585 * (v ** (1/3)), 0.0]

df_clean['q'] = np.select(conditions, choices)
df_clean['Attenuation_dB_km'] = (3.91 / v) * ((1550 / 550) ** (-df_clean['q']))
df_clean['RSSI_dBm'] = P_tx - (df_clean['Attenuation_dB_km'] * L)
df_clean['Outage_State'] = np.where(df_clean['RSSI_dBm'] < Rx_threshold, 1, 0)
df_clean['Target_Future_Outage'] = df_clean['Outage_State'].shift(-1)

df_model = df_clean.dropna(subset=['Target_Future_Outage']).reset_index(drop=True)

# Machine Learning Model
feature_cols = ['Visibility_km', 'Temperature_C', 'Wind_Speed_m_s', 'Attenuation_dB_km', 'RSSI_dBm']
X = df_model[feature_cols]
y = df_model['Target_Future_Outage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Calculate probabilities across dataset
df_model['Outage_Risk'] = rf_model.predict_proba(X)[:, 1]

# Display Layout Placeholders
alert_box = st.empty()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
metric_rssi = col_m1.empty()
metric_risk = col_m2.empty()
metric_vis = col_m3.empty()
metric_status = col_m4.empty()

plot_spot = st.empty()
heatmap_spot = st.empty()

st.divider()
st.subheader("📋 Warning Incident Log & Meantime Summary Table")
log_spot = st.empty()

# Static view before clicking Start
if not start_sim:
    alert_box.info("ℹ️ Click **'▶ Start Live Telemetry Simulation'** in the sidebar to run the live animated stream.")

    fig_rssi = go.Figure()
    fig_rssi.add_trace(go.Scatter(y=df_model['RSSI_dBm'], mode='lines', name='RSSI (dBm)', line=dict(color='navy')))
    fig_rssi.add_hline(y=Rx_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_rssi.update_layout(title="Full Dataset RSSI Signal Overview", height=350)
    plot_spot.plotly_chart(fig_rssi, use_container_width=True)

# Running Live Stream Loop
else:
    warning_logs = []
    window_size = 50

    for i in range(window_size, len(df_model), 2):
        window_df = df_model.iloc[i-window_size:i]
        curr_row = df_model.iloc[i-1]

        curr_rssi = curr_row['RSSI_dBm']
        curr_risk = curr_row['Outage_Risk']
        curr_vis = curr_row['Visibility_km']
        curr_date = curr_row['DATE'] if 'DATE' in curr_row else f"Step {i}"

        # 1. Alert Banner & Incident Logger
        if curr_risk > alert_cutoff or curr_rssi < Rx_threshold:
            alert_box.error(f"🚨 **CRITICAL WARNING:** Predicted Outage Risk at {curr_risk*100:.1f}%! Signal RSSI ({curr_rssi:.2f} dBm) crossing threshold!")
            status_text = "🚨 WARNING"

            # Log event into summary window
            warning_logs.append({
                "Timestamp / Index": curr_date,
                "Outage Risk (%)": f"{curr_risk*100:.1f}%",
                "Signal RSSI (dBm)": f"{curr_rssi:.2f}",
                "Visibility (km)": f"{curr_vis:.2f}",
                "Severity Status": "CRITICAL RISK"
            })
        else:
            alert_box.success("✅ **SYSTEM NOMINAL:** Optical link operating within normal safety limits.")
            status_text = "✅ NORMAL"

        # 2. Update Live Telemetry Metrics
        metric_rssi.metric("Current RSSI", f"{curr_rssi:.2f} dBm")
        metric_risk.metric("Outage Risk", f"{curr_risk*100:.1f}%")
        metric_vis.metric("Visibility", f"{curr_vis:.2f} km")
        metric_status.metric("Link Status", status_text)

        # 3. Dynamic Animated Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window_df.index, y=window_df['RSSI_dBm'], mode='lines+markers', name='Live RSSI (dBm)', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=window_df.index, y=window_df['Outage_Risk']*10, mode='lines', name='Risk Level (x10)', line=dict(color='orange', width=2, dash='dot')))
        fig.add_hline(y=Rx_threshold, line_dash="dash", line_color="red", annotation_text="Sensitivity Limit")

        fig.update_layout(
            title=f"Live Optical Telemetry Stream (Frame {i}/{len(df_model)})",
            xaxis_title="Time Index",
            yaxis_title="Signal / Risk Level",
            height=380,
            xaxis=dict(range=[i-window_size, i+5])
        )
        plot_spot.plotly_chart(fig, use_container_width=True)

        # 4. Dynamic Feature Heatmap
        h_df = window_df[feature_cols]
        h_norm = (h_df - h_df.min()) / (h_df.max() - h_df.min() + 1e-6)

        fig_h = px.imshow(
            h_norm.T,
            labels=dict(x="Time Step", y="Feature", color="Scale"),
            x=window_df.index,
            y=feature_cols,
            color_continuous_scale="Plasma",
            aspect="auto"
        )
        fig_h.update_layout(title="Live Feature Intensity Heatmap (Sliding Window)", height=300)
        heatmap_spot.plotly_chart(fig_h, use_container_width=True)

        # 5. Live Summary Table
        if warning_logs:
            summary_df = pd.DataFrame(warning_logs)
            log_spot.dataframe(summary_df.tail(10), use_container_width=True)
        else:
            log_spot.info("No critical warning incidents logged yet.")

        time.sleep(sim_speed)
