import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("visibility_model.pkl")

# -------------------------------
# Example ground stations
# (lat, lon)
# -------------------------------
ground_stations = {
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
}

# -------------------------------
# Example satellite parameters
# (you can later replace with real-time data)
# -------------------------------
sat_lat = 10.0
sat_lon = 75.0
sat_alt = 550  # km (LEO)

# -------------------------------
# Utility functions
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def estimate_elevation(distance_km, sat_alt):
    # Simple approximation (good enough for ML input)
    return max(0, 90 - (distance_km / 100))

# -------------------------------
# Build prediction dataset
# -------------------------------
rows = []

for name, (gs_lat, gs_lon) in ground_stations.items():
    distance_km = haversine(gs_lat, gs_lon, sat_lat, sat_lon)
    elevation = estimate_elevation(distance_km, sat_alt)

    rows.append({
        "orbit_type_enc": 0,      # 0 = LEO (same encoding as training)
        "elevation": elevation,
        "distance_km": distance_km,
        "gs_lat": gs_lat,
        "gs_lon": gs_lon,
        "station": name
    })

df = pd.DataFrame(rows)

# -------------------------------
# Prepare features EXACTLY as model expects
# -------------------------------
X = df[model.feature_names_in_]

# -------------------------------
# Predict visibility probability
# -------------------------------
df["visibility_prob"] = model.predict_proba(X)[:, 1]

# -------------------------------
# Select best station
# -------------------------------
best = df.sort_values("visibility_prob", ascending=False).iloc[0]

print("\n Best Ground Station for Handover")
print("----------------------------------")
print(f"Station      : {best['station']}")
print(f"Probability  : {best['visibility_prob']:.3f}")
print(f"Elevation    : {best['elevation']:.2f}°")
print(f"Distance     : {best['distance_km']:.2f} km")


