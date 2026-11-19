# How to validate this system

Every check below is a command you run, an output you compare against, and a
statement of what a failure would mean. Nothing here asks you to trust the
README. Run them in order; the whole file takes about five minutes.

Two different questions are being answered, and they should not be confused:

- **§1–4 — does the software do what it claims?** Engineering validation.
- **§5–8 — is the model scientifically credible?** Methodological validation.
- **§9 — what is still not validated.** Read this before defending the work.

---

## 0. The whole thing starts from scratch

```
START_HERE.bat                     Windows
./start_here.sh                    macOS / Linux
```

**Expect:** five numbered steps, ending with the app open in your browser and a
probability near 0.113 for Mathare valley (lat −1.2630, lon 36.8600).

Verified here from a clean copy with no data and no model: the script found the
archive, extracted 50 rasters, trained (16.2 s), served the API, started the UI,
and the page answered. Verified on Linux — **not** on Windows.

**If it stops:** it prints which of the five steps failed and what to do. The
usual cause is an interpreter older than 3.12.

## 1. The pipeline reproduces

```bash
python scripts/run_pipeline.py
```

**Expect:** completion in roughly 15–20 seconds, and a model-comparison table
matching the one in `README.md` to four decimal places — XGBoost at ROC AUC
0.9058, F1 0.7936. Every stage is seeded (`seed: 42` in `config.yaml`).

**If the numbers move:** either the data in `data/nairobi_flood_data/` is not
the 50 files the audit expects, or a library version changed the RNG. Both are
findings worth reporting, not things to hide. Check
`outputs/tables/raster_audit.csv` first.

## 2. The tests pass

```bash
pytest                              # 41 passed
pytest -o addopts="" --verbose      # ...and the name of every one
```

**Expect:** 41 passed. `pytest.ini` sets `-q`, so the second form is what prints
the individual test names. Read the test names rather than just the count — they
state the properties being enforced, including
`test_train_and_test_events_are_disjoint`,
`test_no_pixel_appears_on_both_sides` and
`test_point_outside_the_grid_is_reported_not_guessed`.

**If one fails:** the property it names is no longer true. Do not submit around
a failing test.

## 3. The API's numbers come from the trained model

Two independent code paths compute the same probability: the point scorer the
API uses, and the chunked raster sweep that produced the GeoTIFF. They share
the model but nothing else. Start the service, then compare.

```bash
uvicorn floodrisk.api:app --app-dir src &
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"points":[{"lat":-1.2630,"lon":36.8600}],"scenario":"median"}'
```

**Expect** `"probability": 0.1133` for that point, and the same value when the
pixel is read out of `outputs/flood_susceptibility_nairobi_30m.tif`. Measured
agreement across four sites:

| Site | API | Raster sweep |
|---|---|---|
| Mathare valley | 0.1133 | 0.1133 |
| Nairobi CBD | 0.0015 | 0.0015 |
| Kibera | 0.0048 | 0.0048 |
| Kasarani | 0.0048 | 0.0048 |

**One trap.** Do not try to re-derive the probability by feeding the `features`
block from the response back into the model. Those values are rounded to three
decimals for readability, and tree split thresholds sit close enough to them
that rounding alone moves Mathare from 0.1133 to 0.1508. The rounding is a
display choice; a mismatch found that way is an artefact of your check, not a
defect in the service.

## 4. The app cannot invent a prediction

This is worth validating explicitly, because the earlier standalone version of
this app did invent them — with no model file present it returned
`0.3 + 0.4*exp(-distance)` plus random noise, which is indistinguishable from a
prediction on screen.

```bash
grep -nE "^\s*(import|from)\s+(sklearn|joblib|xgboost|rasterio|numpy)" app/streamlit_app.py
```

**Expect:** no matches. The app's entire import list is `os`, `httpx`, `pandas`
and `streamlit` — no model machinery, so there is no code path by which it could
produce a number of its own. (Anchor the pattern on the import lines. A bare
search for `random` also hits the comment that explains what the old app did.)

Then stop the API and reload the page.

**Expect:** an error panel naming the unreachable URL, and **no** metric and no
table. Verified: with the API down the app renders 0 metrics and 0 dataframes.

**If a probability still appears with the API down,** something is estimating
locally and must be removed.

## 5. The validation design is not leaking

This is the single most important methodological check, and the repository
measures it rather than asserting it:

```bash
python scripts/check_split_leakage.py
```

| Split | Test rows | ROC AUC | F1 |
|---|---|---|---|
| Event holdout — 4 unseen storms | 1,600 | 0.9058 | 0.7936 |
| Random 80/20 — all 15 events on both sides | 1,200 | 0.9356 | 0.8528 |

**Random-split inflation: +0.0297 ROC AUC, +0.0592 F1.**

That gap is the thing to be able to explain in a viva. Pixels from one storm
are spatially autocorrelated and share a single rainfall field, so a random
split puts near-duplicate rows on both sides and the score partly measures
memorisation. The reported 0.9058 is the defensible number precisely because it
is the lower one. If someone asks why the figures are not higher, this is the
answer.

## 6. The model's rainfall response is backwards, and you must be able to say so

```bash
cat outputs/tables/rainfall_response.csv
```

**Expect** a non-monotonic sequence. Holding terrain fixed over 500 real pixels
and sweeping rainfall 0–1,000 mm, mean probability rises on only **4 of 8**
steps; between a light storm (50/120 mm) and a heavy one (400/800 mm) it
*falls* for **282 of 500** pixels, mean change −0.0394.

Confirm the service refuses to pretend otherwise:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"points":[{"lat":-1.26,"lon":36.86}],"rain7":300,"rain_sum":500}'
```

**Expect:** `422`. The endpoint rejects invented rainfall rather than returning
a confident answer pointing the wrong way.

**Why it happens:** CHIRPS rainfall is near-uniform within an event and sampling
is balanced per event, so `rain_sum` acts as an event identifier, not a driver.
Permutation importance agrees — `dem` 0.338 against `rain_sum` 0.006. This is a
terrain susceptibility model, and describing it as anything else is the main way
this project could be criticised.

## 7. The threshold is presentational, not calibrated

```bash
open outputs/figures/reliability_curve.png
```

Training is balanced 50/50 while true event prevalence is about 0.08%, so 0.5 is
a display default. Any operational use needs a threshold chosen against a real
cost ratio for missed floods versus false alarms, on prevalence-corrected
probabilities. Check that nothing in your write-up calls 0.5 optimal.

## 8. The labels are not ground truth

Sentinel-1 water masks under-detect flooding beneath dense urban fabric, and a
satellite overpass can miss peak extent. So recall is measured against *SAR-
visible* water, not against flooding. Confirm your write-up never calls these
observations verified.

---

## 9. What is **not** validated

State these as limitations; do not let a reader assume otherwise.

| Claim | Status |
|---|---|
| Pipeline reproduces, tests pass, API serves real model output | **Verified** by the commands above |
| Event-holdout scores | **Verified**, and shown to be conservative (§5) |
| `docker compose up --build` | **Never executed.** No Docker daemon was available where the code was written. Pins were checked against PyPI for Python 3.13 wheels only — run it yourself |
| The UI in a real browser | **Verified in Chromium** on Linux: the page was loaded, coordinates entered, Predict clicked, and the rendered probability (0.113 for Mathare) read back off the screen. **Not** verified on Windows, which is where you will run it |
| Agreement with observed flood impact | **Not tested.** No independent ground-truth dataset — no insurance claims, no field survey, no reported inundation depths — was used at any point |
| Transferability to another city | **Not tested.** Nairobi only |
| Performance on a future storm | **Not tested.** The four held-out events are historical; nothing here is a forecast |
| Rainfall as a causal driver | **Refuted by §6.** Do not claim it |

---

*Validate before you defend. A number you cannot reproduce on your own machine
is a number you cannot be questioned about.*
