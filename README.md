# Physics-Informed Machine Learning Architecture for Real-Time Optical Link Outage Prediction in FSO

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23154490.svg)](https://doi.org/10.5281/zenodo.23154490)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KbwLPenNy_RlMbEPWNDmY5bvEh8WIusj?usp=sharing)

**Authors:** Rajlaxmi Thakur, Saniya Sharma  
**Research Preprint:** [Zenodo DOI: 10.5281/zenodo.23154490](https://doi.org/10.5281/zenodo.23154490)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fso-link-outage-prediction-y45sr6hyzwup7cd5rlx2xm.streamlit.app/)

## 📌 Executive Summary
Free Space Optics (FSO) links are vulnerable to weather impacts like atmospheric attenuation, Mie scattering, and signal fade caused by fog, rain, and cloud cover. Standard reactive mitigation protocols manage link downtime reactively, addressing failures only after a total signal outage has occurred. In this project a physics-informed approach is used to create a lookahead window of 45-60 seconds to provide an early warning based on link outage probability. 

This repository implements a **physics-informed machine learning pipeline** that integrates classical optical attenuation physics (Kim-Kruse Mie scattering models & Beer-Lambert law) with lookahead feature target shifting in Random Forest classifiers. The system provides a **45–60 second proactive lead-time warning** prior to link drops below receiver sensitivity thresholds (-38 dBm to -40 dBm).

Data Source: NOAA Local Climatological Data (LCD) - Real hourly weather data | Physics: Kim & Kruse model for visibility to attenuation | ML: Time-shifted prediction (t -> t+k)

---
📄 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23154490.svg)](https://doi.org/10.5281/zenodo.23154490)

📄 **[Read Published Research Paper on Zenodo](https://doi.org/10.5281/zenodo.23154490)**

## 🚀 Live Interactive Telemetry Dashboard
Access the live Streamlit early warning dashboard:  
👉 **[FSO Live Telemetry EWS Dashboard](https://fso-link-outage-prediction-y45sr6hyzwup7cd5rbx2xm.streamlit.app)**
https://fso-link-outage-prediction-y45sr6hyzwup7cd5rlx2xm.streamlit.app/
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
