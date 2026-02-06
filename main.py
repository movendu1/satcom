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
# Streamlit Page Setup
# ----------------------------------
st.set_page_config(
    page_title="Satellite Handover Dashboard",
    layout="centered"
)

st.title("🛰️ Satellite–Ground Station Handover Dashboard")
st.caption("ML-powered real-time visibility prediction")

st.divider()

# ----------------------------------
# Satellite Input Controls
# ----------------------------------
st.subheader("📡 Satellite Parameters")

orbit_type = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
orbit_type_enc = {"LEO": 0, "MEO": 1, "GEO": 2}[orbit_type]

elevation = st.slider(
    "Elevation Angle (degrees)",
    min_value=0.0,
    max_value=90.0,
    value=45.0
)

distance_km = st.slider(
    "Satellite Distance (km)",
    min_value=200.0,
    max_value=40000.0,
    value=1200.0
)

st.divider()

# ----------------------------------
# ML Prediction
# ----------------------------------
results = []

for station, (gs_lat, gs_lon) in GROUND_STATIONS.items():
    row = {
        "orbit_type_enc": orbit_type_enc,
        "elevation": elevation,
        "distance_km": distance_km,
        "gs_lat": gs_lat,
        "gs_lon": gs_lon
    }

    X = pd.DataFrame([row])
    X = X[FEATURES]  # force exact feature order

    prob = model.predict_proba(X)[0][1]

    results.append({
        "Station": station,
        "Visibility Probability": round(prob, 4),
        "Elevation (deg)": round(elevation, 2),
        "Distance (km)": round(distance_km, 2)
    })

df = pd.DataFrame(results).sort_values(
    by="Visibility Probability",
    ascending=False
)

# ----------------------------------
# Display Output
# ----------------------------------
st.subheader("📊 Ground Station Ranking")
st.dataframe(df, use_container_width=True)

best = df.iloc[0]

st.success(
    f"""
 **Best Ground Station for Handover**

**Station:** {best['Station']}  
**Probability:** {best['Visibility Probability']}  
**Elevation:** {best['Elevation (deg)']}°  
**Distance:** {best['Distance (km)']} km
"""
)

st.caption("Model: Random Forest | Feature-locked inference")
