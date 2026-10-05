# Physics-Informed Machine Learning Architecture for Real-Time Optical Link Outage Prediction in Free Space Optics (FSO)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fso-link-outage-prediction-y45sr6hyzwup7cd5rbx2xm.streamlit.app)

## 📌 Executive Summary
Free Space Optics (FSO) links are vulnerable to weather impacts like atmospheric attenuation, Mie scattering, and signal fade caused by fog, rain, and cloud cover. Standard reactive mitigation protocols manage link downtime reactively, addressing failures only after a total signal outage has occurred. In this project a physics-informed approach is used to create a lookahead window of 45-60 seconds to provide an early warning based on link outage probability. 

This repository implements a **physics-informed machine learning pipeline** that integrates classical optical attenuation physics (Kim-Kruse Mie scattering models & Beer-Lambert law) with lookahead feature target shifting in Random Forest classifiers. The system provides a **45–60 second proactive lead-time warning** prior to link drops below receiver sensitivity thresholds (-38 dBm to -40 dBm).

---
📄 **[Read Full Research Paper Draft & Documentation](./FSO_early_warning_system.docx)**
## 🚀 Live Interactive Telemetry Dashboard
Access the live Streamlit early warning dashboard:  
👉 **[FSO Live Telemetry EWS Dashboard](https://fso-link-outage-prediction-y45sr6hyzwup7cd5rbx2xm.streamlit.app)**

---

## 🛠️ System Architecture

```text
[ NOAA Weather Dataset / Telemetry Stream ]
                  │
                  ▼
[ Physics Core: Kim-Kruse Mie Scattering Engine ]
       (Calculates spatial attenuation A_dB/km)
                  │
                  ▼
[ Time-Shifted Target Generator (t + k) ]
                  │
                  ▼
[ Random Forest Ensemble Classifier ]
                  │
                  ▼
[ Real-Time Streamlit Telemetry & Warning Alert ]
