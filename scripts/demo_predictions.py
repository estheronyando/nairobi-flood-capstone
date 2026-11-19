#!/usr/bin/env python3
"""Drive the running API as a client and record what it returned.

This is the "a client can send data and get predictions back" requirement,
made repeatable: run the service, run this, and the transcript plus
`outputs/tables/api_demo.csv` are regenerated from live HTTP calls.

    uvicorn floodrisk.api:app --app-dir src &
    python scripts/demo_predictions.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

# Named places, so the output can be sanity-checked against local knowledge
# rather than only against the model.
SITES = [
    ("Nairobi CBD",            -1.2921, 36.8219),
    ("Mathare valley",         -1.2630, 36.8600),
    ("Kibera",                 -1.3130, 36.7800),
    ("Embakasi / Nairobi R.",  -1.3190, 36.9250),
    ("Kasarani",               -1.2270, 36.8870),
    ("Karen",                  -1.3350, 36.7100),
    ("Athi plains, south edge",-1.3733, 36.8569),
    ("Mombasa (outside grid)", -4.0435, 39.6682),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--scenario", default="median")
    args = ap.parse_args()

    try:
        client = httpx.Client(base_url=args.base_url, timeout=60.0)
        health = client.get("/health").raise_for_status().json()
    except Exception as exc:
        print(f"Cannot reach the API at {args.base_url}: {exc}\n"
              "Start it with:  uvicorn floodrisk.api:app --app-dir src", file=sys.stderr)
        return 1

    card = client.get("/model").raise_for_status().json()
    print(f"Service   : {args.base_url}  ({health['status']})")
    print(f"Model     : {card['model_name']}  features={card['features']}")
    print(f"Held out  : {len(card['test_events'])} events  "
          f"ROC AUC {card['metrics']['roc_auc']:.4f}  F1 {card['metrics']['f1']:.4f}")
    rr = card.get("rainfall_response") or {}
    if rr:
        print(f"Rainfall  : monotonic={rr['monotonic']} "
              f"({rr['monotone_steps']}/{rr['total_steps']} steps rising)")
    print()

    body = {"points": [{"lat": lat, "lon": lon} for _, lat, lon in SITES],
            "scenario": args.scenario}
    resp = client.post("/predict", json=body).raise_for_status().json()

    print(f"POST /predict  scenario={resp['scenario']}  model={resp['model_name']}")
    print(f"  {'location':26} {'status':16} {'P(flood)':>9} {'band':9} {'elev m':>7}")
    rows = []
    for (name, _, _), p in zip(SITES, resp["predictions"]):
        feats = p.get("features") or {}
        prob = f"{p['probability']:.4f}" if p["probability"] is not None else "-"
        print(f"  {name:26} {p['status']:16} {prob:>9} "
              f"{str(p['risk_band'] or '-'):9} {feats.get('dem', '-'):>7}")
        rows.append({
            "location": name, "lat": p["lat"], "lon": p["lon"], "status": p["status"],
            "probability": p["probability"], "risk_band": p["risk_band"],
            **{k: feats.get(k) for k in ("dem", "slope", "builtup", "rain7", "rain_sum")},
        })

    print(f"\n  caveat: {resp['caveat']}")

    # Scenario comparison on one pixel: observed rainfall fields only.
    print("\nSame pixel (Mathare) under observed rainfall scenarios:")
    for scen in ["median", "max"] + client.get("/scenarios").json()["events"][:3]:
        r = client.post("/predict", json={
            "points": [{"lat": -1.2630, "lon": 36.8600}], "scenario": scen,
        }).json()
        print(f"  {scen:16} P={r['predictions'][0]['probability']:.4f}")

    # The service must refuse what it cannot support.
    print("\nRefusals:")
    bad = client.post("/predict", json={"points": [{"lat": -1.26, "lon": 36.86}],
                                        "rain7": 300, "rain_sum": 500})
    print(f"  arbitrary rainfall override -> HTTP {bad.status_code} (not supported by this model)")
    oob = client.post("/predict", json={"points": [{"lat": -4.0435, "lon": 39.6682}]}).json()
    print(f"  outside the trained extent  -> status={oob['predictions'][0]['status']}, "
          f"probability={oob['predictions'][0]['probability']}")

    out = REPO_ROOT / "outputs" / "tables" / "api_demo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
