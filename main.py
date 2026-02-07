import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# Load trained ML model
# ----------------------------------
model = joblib.load("visibility_model.pkl")
FEATURES = list(model.feature_names_in_)

# ----------------------------------
# Ground station database
# ----------------------------------
GROUND_STATIONS = {
    "Bangalore": (12.9716, 77.5946),
    "Singapore": (1.3521, 103.8198),
    "Sriharikota": (13.733, 80.235),
    "Port Blair": (11.6234, 92.7265),
    "Mauritius": (-20.3484, 57.5522)
}

# ----------------------------------
# Station load (0 = free, 1 = saturated)
# ----------------------------------
STATION_LOAD = {
    "Bangalore": 0.30,
    "Singapore": 0.75,
    "Sriharikota": 0.40,
    "Port Blair": 0.20,
    "Mauritius": 0.60
}

# ----------------------------------
# Utility Functions (MATCH TRAINING)
# ----------------------------------
def fspl(distance_km, freq_ghz=12):
    return 92.45 + 20 * np.log10(distance_km) + 20 * np.log10(freq_ghz)

def estimate_doppler(distance_km):
    return 0.02 * distance_km  # proxy (same as dataset)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Satellite Handover Dashboard", layout="centered")
st.title("🛰️ Satellite–Ground Station Handover Dashboard")
st.caption("Physics-aware ML + load-aware decision engine")
st.divider()

# ----------------------------------
# Satellite Inputs
# ----------------------------------
orbit_type = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
orbit_type_enc = {"LEO": 0, "MEO": 1, "GEO": 2}[orbit_type]

sat_altitude_km = st.slider("Satellite Altitude (km)", 300, 36000, 550)
elevation = st.slider("Elevation Angle (degrees)", 0.0, 90.0, 45.0)
distance_km = st.slider("Slant Range Distance (km)", 200.0, 40000.0, 1200.0)

st.divider()

# ----------------------------------
# Prediction Logic
# ----------------------------------
rows = []

for station, (gs_lat, gs_lon) in GROUND_STATIONS.items():
    # Feature engineering (MATCH TRAINING)
    distance_log = np.log1p(distance_km)
    sin_elev = np.sin(np.radians(elevation))
    cos_elev = np.cos(np.radians(elevation))
    doppler_hz = estimate_doppler(distance_km)
    fspl_db = fspl(distance_km)
    atm_loss_db = 2 + (1 - sin_elev) * 3
    rx_power_dbm = -fspl_db - atm_loss_db
    rx_margin = rx_power_dbm + 120

    row = {
        "orbit_type_enc": orbit_type_enc,
        "sat_altitude_km": sat_altitude_km,
        "gs_lat": gs_lat,
        "gs_lon": gs_lon,
        "distance_km": distance_km,
        "elevation": elevation,
        "doppler_hz": doppler_hz,
        "fspl_db": fspl_db,
        "atm_loss_db": atm_loss_db,
        "rx_power_dbm": rx_power_dbm,
        "distance_log": distance_log,
        "sin_elevation": sin_elev,
        "cos_elevation": cos_elev,
        "rx_margin": rx_margin
    }

    X = pd.DataFrame([row])[FEATURES]

    prob = model.predict_proba(X)[0][1]
    load = STATION_LOAD.get(station, 0)
    final_score = prob * (1 - load)

    rows.append({
        "Station": station,
        "Raw Probability": round(prob, 4),
        "Load Factor": load,
        "Final Score": round(final_score, 4)
    })

df = pd.DataFrame(rows).sort_values("Final Score", ascending=False)

# ----------------------------------
# Display
# ----------------------------------
st.subheader("📊 Load-Aware Ground Station Ranking")
st.dataframe(df, use_container_width=True)

best = df.iloc[0]

st.success(
    f"""
 **Best Ground Station**

Station: **{best['Station']}**  
Final Score: **{best['Final Score']}**  
Raw Probability: **{best['Raw Probability']}**  
Load Factor: **{best['Load Factor']}**
"""
)

st.caption("Feature-locked | Physics-informed | Production-safe")

