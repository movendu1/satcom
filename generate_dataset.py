from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
import random
from datetime import timedelta

# Load timescale
ts = load.timescale()
t0 = ts.now()

# Load satellites
def load_tle(file):
    sats = []
    with open(file) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            sats.append(EarthSatellite(
                lines[i+1].strip(),
                lines[i+2].strip(),
                lines[i].strip(),
                ts
            ))
    return sats

leo_sats = load_tle("data/tle/leo_starlink.tle") + load_tle("data/tle/leo_iridium.tle")
meo_sats = load_tle("data/tle/meo_gps.tle") + load_tle("data/tle/meo_galileo.tle")


# Ground stations
stations = {
    "DELHI": wgs84.latlon(28.6, 77.2),
    "SINGAPORE": wgs84.latlon(1.3, 103.8),
    "LONDON": wgs84.latlon(51.5, -0.1),
    "CALIFORNIA": wgs84.latlon(37.3, -121.9),
}

rows = []

for sat in leo_sats + meo_sats:
    orbit_type = "LEO" if sat in leo_sats else "MEO"

    for station_name, gs in stations.items():
        for i in range(5):
            t = t0 + timedelta(minutes=random.randint(0, 1440))
            sat_pos = sat.at(t)
            difference = sat_pos - gs.at(t)

            alt, az, dist = difference.altaz()

            elevation = alt.degrees
            distance = dist.km

            # REALISTIC LINK CONDITION
            good_link = int(
                elevation > 10 and
                (distance < 25000 if orbit_type == "MEO" else distance < 3000)
            )

            rows.append([
                orbit_type,
                elevation,
                distance,
                gs.latitude.degrees,
                gs.longitude.degrees,
                good_link
            ])

df = pd.DataFrame(rows, columns=[
    "orbit_type",
    "elevation",
    "distance_km",
    "gs_lat",
    "gs_lon",
    "good_link"
])

df.to_csv("satcom_dataset.csv", index=False)
print("Dataset generated: satcom_dataset.csv")
