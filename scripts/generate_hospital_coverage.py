
from pathlib import Path
import json
import os
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "trauma_hospitals.csv"
OUTPUT = ROOT / "data" / "hospital_coverage.geojson"

API_KEY = os.environ.get("GEOAPIFY_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEOAPIFY_API_KEY GitHub secret")

hospitals = pd.read_csv(INPUT)
required = {"hospital", "lat", "lon"}
if not required.issubset(hospitals.columns):
    raise ValueError(f"Hospital CSV must contain {required}")

hospitals = hospitals.dropna(subset=["lat", "lon"]).copy()
hospitals = hospitals.drop_duplicates(subset=["hospital"])

# 5, 10, 15, 20, 25, and 30 minutes, in seconds.
ranges = [300, 600, 900, 1200, 1500, 1800]
features = []

for _, hospital in hospitals.iterrows():
    name = str(hospital["hospital"])
    print(f"Calculating coverage for {name}", flush=True)

    params = {
        "lat": float(hospital["lat"]),
        "lon": float(hospital["lon"]),
        "type": "time",
        "mode": "drive",
        "range": ",".join(map(str, ranges)),
        "traffic": "free_flow",
        "apiKey": API_KEY,
    }

    response = requests.get(
        "https://api.geoapify.com/v1/isoline",
        params=params,
        timeout=120,
    )

    # Some calculations may take longer and return a job ID.
    if response.status_code == 202:
        job_id = response.json().get("properties", {}).get("id")
        if not job_id:
            raise RuntimeError(f"No pending job ID for {name}")

        for attempt in range(20):
            time.sleep(10)
            response = requests.get(
                "https://api.geoapify.com/v1/isoline",
                params={"id": job_id, "apiKey": API_KEY},
                timeout=120,
            )
            if response.status_code == 200:
                break
            if response.status_code != 202:
                response.raise_for_status()

    response.raise_for_status()
    if response.status_code != 200:
        raise RuntimeError(f"Coverage calculation timed out for {name}")

    data = response.json()
    returned = data.get("features", [])

    actual_ranges = {
        int(f.get("properties", {}).get("range", -1))
        for f in returned
    }
    if not set(ranges).issubset(actual_ranges):
        raise RuntimeError(f"Incomplete coverage ranges for {name}")

    for feature in returned:
        seconds = int(feature["properties"]["range"])
        if feature.get("geometry", {}).get("type") not in (
            "Polygon", "MultiPolygon"
        ):
            raise RuntimeError(f"Invalid geometry for {name}")

        feature["properties"] = {
            "hospital": name,
            "minutes": seconds // 60,
            "mode": "drive",
            "traffic": "free_flow",
        }
        features.append(feature)

if not features:
    raise RuntimeError("No coverage areas generated")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps({"type": "FeatureCollection", "features": features}),
    encoding="utf-8",
)

print(f"Saved {len(features)} coverage polygons to {OUTPUT}")
