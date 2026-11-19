# Urban Flood Susceptibility for Nairobi
## Final report — DSA 8401 Applied Machine Learning Capstone

Esther Onyando · MSc Data Science & Analytics, Strathmore University

---

## 1. Problem framing

Nairobi floods along the Nairobi, Mathare and Ngong river corridors every rainy season,
and the burden falls hardest on low-income and informally built settlements where
drainage is thinnest and buildings sit closest to the channel. Mapping of where this
happens is done project by project, so county planners have no standing screening layer
to consult before the rains when deciding where to clear drains or pre-position response.

This system predicts, for every 30 m pixel in Nairobi City County, the probability that
the pixel floods during a heavy-rainfall event. The output is a continuous susceptibility
surface rather than a flooded/not-flooded map, because the operational question is one
of triage — *which parts of the city warrant attention first* — and a ranking answers
that better than a binary label.

Two things make this a system rather than a model. First, susceptibility is conditional
on rainfall, so the surface has to be regenerable under a different storm rather than
fixed once. Second, a map that exists only inside a notebook cannot be consulted by
anyone, so the model is served behind an API that answers point queries.

### Why this is a supervised binary classification problem

Flood extent is observable after the fact: Sentinel-1 SAR sees standing water through
cloud, which is what makes a flood inventory possible at all in a tropical city. Fifteen
such events between 2016 and 2025 give labelled pixels, so the problem is supervised.
The label is binary at the pixel level — flooded in this event, or not — and the useful
output is the probability behind that label.

---

## 2. Data

| Layer | Source | Role |
|---|---|---|
| Elevation (`dem`) | SRTM, 30 m | Terrain predictor |
| Slope (`slope`) | Derived from the DEM | Terrain predictor |
| Built-up cover (`builtup`) | ESA WorldCover / OpenStreetMap | Anthropogenic predictor |
| 7-day antecedent rainfall (`rain7`) | CHIRPS | Hydro-climatic predictor, per event |
| Event rainfall total (`rain_sum`) | CHIRPS | Hydro-climatic predictor, per event |
| Land cover | ESA WorldCover | Audited, not modelled |
| Permanent water | Surface-water occurrence ≥ 30% | Masking only |
| Flood labels | Sentinel-1 SAR event water masks, 15 events 2016–2025 | Target |

All 50 rasters sit on one grid: 1,858 × 1,108 cells, EPSG:32737, 30 m pixels,
2,058,664 cells per layer. The audit stage asserts this and halts if any layer differs,
rather than resampling silently — a half-pixel offset between a predictor and a label
would be invisible in every output while corrupting every sample drawn.

### Real-world messiness

This is not a cleaned benchmark, and the complications are inherent rather than
introduced:

- **Label noise.** SAR flood detection in a dense city is degraded by speckle, layover,
  shadow and double-bounce off buildings. Satellite overpass timing may also miss peak
  extent. The labels are event-water proxies, not ground truth, which caps how well any
  model can score.
- **Extreme class imbalance.** Flood pixels are 0.070–0.094% of the grid per event
  (1,433–1,939 cells). Prevalence across all fifteen events is roughly 0.08%.
- **Missingness.** The DEM carries 0.2% invalid cells and slope 0.5%, encoded as NaN
  rather than a nodata tag; rainfall layers carry 0.2%. These are dropped at sampling.
- **Resolution mismatch.** CHIRPS rainfall is natively around 5 km, resampled to the
  30 m analysis grid. Rainfall is therefore nearly constant across the city within any
  one event — a fact with large consequences, taken up in §6.
- **Temporal structure.** Events span nine years of changing urban form, but the
  built-up and terrain layers are single-date and static.

### An inconsistency worth recording

The pipeline audits the relationship between the two rainfall layers
(`outputs/tables/rainfall_consistency.csv`). They are not required to sit in a fixed
ratio — a short, intense storm following a wet week legitimately produces more
seven-day antecedent rainfall than event total — but the spread across events is wide:

| Event | Median `rain7` (mm) | Median `rain_sum` (mm) | Ratio |
|---|---|---|---|
| `E07_2021_05b` | 192.9 | 8.2 | **23.6** |
| `E08_2021_11` | 131.3 | 86.3 | 1.52 |
| `E10_2023_11` | 177.5 | 135.4 | 1.31 |
| `E05_2020_04` | 184.1 | 150.2 | 1.23 |
| … nine events between 0.14 and 0.86 … | | | |
| `E11_2024_03` | 60.4 | 614.7 | 0.10 |
| `E01_2016_04` | 0.0 | 225.4 | **0.00** |

Two events sit outside any reading I can defend. `E07_2021_05b` records 23× more
antecedent rainfall than event rainfall, and `E01_2016_04` records a median antecedent
accumulation of exactly zero against a 225 mm event total. Both suggest the two layers
were built over mismatched windows for those events. Four further events
(`E05`, `E08`, `E10`, `E15`) have antecedent rainfall modestly exceeding the event
total, which is unusual but physically possible.

Both clearly anomalous events fall in the training set, not the held-out four, so the
reported scores are not contaminated. Given rainfall contributes almost nothing to the
model in any case (§6), the practical effect is small — but it is a defect in the
inventory, and it is recorded here rather than left in the data unremarked.

---

## 3. Method

### Sampling

For each event, 200 flood and 200 non-flood pixels are drawn under a fixed seed, giving
6,000 rows across fifteen events. Two choices matter more than they appear:

**Non-flood pixels exclude permanent water.** Rivers and dams are wet in every scene.
Leaving them in the dry class would let the model separate the classes by answering "is
this a river", which is not the question being asked. The flood labels have permanent
water removed for the same reason, so the target is transient flooding.

**Balancing is per event, not global.** Three or four large events would otherwise
dominate. The cost is that the training distribution is 50/50 while reality is 0.08%
positive, which is why the operating threshold is a presentational default rather than
a calibrated decision boundary (§7).

### Validation: event-based holdout

Eleven events train, four are held out: `E12_2024_04a`, `E13_2024_04b`, `E14_2025_04`,
`E15_2025_05`. Splitting on events rather than pixels is the single most important
design decision in this project.

Pixels inside one flood footprint are strongly spatially autocorrelated. Under a random
pixel split, a cell and its immediate neighbour land on opposite sides of the boundary,
so the test set is effectively memorised and every metric becomes optimistic. Published
flood-susceptibility work frequently reports ROC AUC above 0.98 under exactly this
design. Holding out whole events asks the question that matters operationally: does the
model generalise to a flood it has never seen? A test is included
(`tests/test_pipeline.py::TestEventSplit`) asserting that no `(event, row, col)` triple
appears on both sides, because this assumption silently carries every number in the
report.

---

## 4. Technique justification

### What was used, and why

**Logistic regression** is the baseline. A boosted tree that cannot beat a linear model
on five predictors is not worth the operational cost, so the comparison is not
ceremonial. It is scaled inside a pipeline, because elevation is of order 10³ and slope
of order 10¹, and an unscaled linear model would be dominated by units.

**Random forest and XGBoost** are the substantive candidates. Flood susceptibility is
driven by threshold effects and interactions — water pools below a certain slope, *and*
near a channel, *and* where infiltration is blocked — which is exactly what axis-aligned
tree ensembles represent cheaply and linear models cannot.

**Two ensembles**, an unweighted probability average and a stacking model, test whether
combining the candidates buys anything beyond the best single learner. They are reported
honestly: it did not.

**SHAP and permutation importance** are both computed. Permutation importance is the one
trusted here, because it is measured on data the model never saw and, unlike
impurity-based importance, is not biased toward high-cardinality continuous features
such as elevation.

### What was deliberately ruled out

**Deep learning.** 6,000 samples and five tabular predictors do not support a CNN. A
convolutional model would be justified if the input were image patches, letting the
network learn spatial context — plausible future work with a larger inventory, but not
at this sample size, where it would overfit while being harder to explain to a regulator.

**Clustering and dimensionality reduction.** Labels exist, so the problem is supervised.
PCA on five interpretable predictors would destroy exactly the interpretability that a
public-sector hazard product needs, in exchange for no computational benefit at this
scale.

**SMOTE and synthetic oversampling.** Balanced sampling already addresses the imbalance,
and synthesising flood pixels by interpolating between real ones would invent locations
that never flooded — a bad idea generally and worse when the output is a map of specific
places.

**Hyperparameter search.** Parameters are taken from the established configuration
rather than tuned. With four held-out events as the only honest evaluation set, an
extensive search would be selecting on 1,600 rows, and the winner's-curse optimism would
exceed the gains on offer. The 0.001 gap between the top two models shows how little
headroom there is to tune into.

### The selection rule

The pipeline does not deploy the argmax. It takes the best ROC AUC, keeps every model
within a 0.005 tolerance, and deploys the simplest of those under an explicit ordering.

Here that means XGBoost (0.9058) is served rather than the stacking ensemble (0.9068).
A 0.001 difference on 1,600 test rows is far inside run-to-run variation; paying 43× the
fit time and a meta-model that is harder to explain for it would be buying noise. The
rationale is recorded in `models/model_card.json` so the decision travels with the
artefact.

---

## 5. Results

### Held-out event performance

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Brier |
|---|---|---|---|---|---|---|
| Stacking ensemble | 0.8062 | 0.8614 | 0.7300 | 0.7903 | **0.9068** | 0.1317 |
| **XGBoost (deployed)** | 0.8069 | 0.8522 | 0.7425 | **0.7936** | 0.9058 | **0.1304** |
| Random forest | 0.7931 | 0.8558 | 0.7050 | 0.7731 | 0.8999 | 0.1313 |
| Average ensemble | 0.7988 | 0.8444 | 0.7325 | 0.7845 | 0.8940 | 0.1341 |
| Logistic regression | 0.6625 | 0.6395 | 0.7450 | 0.6882 | 0.7899 | 0.1931 |

The tree ensembles clear the linear baseline decisively — 0.90 against 0.79 ROC AUC —
which confirms that the flood/dry boundary in this feature space is genuinely non-linear
and that the modelling choice was not ceremonial. The three best models sit within
0.007 of each other, so the ranking among them is not meaningful; what is meaningful is
the gap to logistic regression.

At the 0.5 threshold the deployed model returns 594 true positives, 697 true negatives,
103 false positives and 206 false negatives on 1,600 held-out pixels. Recall (0.74) sits
below precision (0.85): the model misses about a quarter of flooded pixels in events it
has never seen. For a screening product intended to direct attention, that asymmetry is
the wrong way round — a missed flooded settlement costs more than an unnecessary
inspection — which is a threshold question, not a model question, and is taken up in §7.

### Probability quality

The Brier score of 0.130 and the reliability curve show probabilities that are usable
but not sharp. This matters more than it usually would: the product is a probability
surface people will threshold, so calibration is part of the deliverable rather than a
diagnostic afterthought.

### The susceptibility surface

Scoring all 2,044,682 valid cells under the median rainfall scenario puts 12.6% of the
city at or above 0.5, with a mean probability of 0.20. Visually the surface traces
drainage lines and valley bottoms and leaves the elevated central plateau low, which is
the pattern domain knowledge predicts.

One artefact is visible and should not be explained away: rectangular blocking in the
north-east of the map. That is the CHIRPS rainfall footprint, a ~5 km grid imprinting
itself on a 30 m output. It is a resolution mismatch made visible, and it is a reason to
read the surface as a regional indication rather than a parcel-level verdict.

---

## 6. What the model actually learned

Permutation importance on the held-out events — the drop in ROC AUC when each predictor
is shuffled:

| Predictor | Permutation importance | Mean \|SHAP\| |
|---|---|---|
| `dem` (elevation) | 0.338 | 1.981 |
| `slope` | 0.054 | 0.897 |
| `builtup` | 0.035 | 0.467 |
| `rain_sum` | 0.006 | 0.414 |
| `rain7` | 0.002 | 0.255 |

**This is a terrain model.** Elevation carries roughly six times the signal of the next
predictor and a hundred times that of rainfall. The two rainfall predictors are close to
inert.

The reason is structural rather than a defect in the data. CHIRPS rainfall is nearly
spatially uniform across a city 55 km across, and the sampling design is balanced *within
each event*. So within any event's 400 pixels, rainfall is almost constant and cannot
discriminate flooded cells from dry ones; all the within-event separation has to come
from terrain and built form. Rainfall would only earn importance in a design that asked
the model to discriminate *between* events, which this one does not.

Three consequences follow, and they constrain how the output may honestly be used:

1. **The map is closer to a terrain-susceptibility surface than a rainfall-driven flood
   forecast.** It says where water would collect, not when.
2. **The model's rainfall response is not monotonic, and the API therefore refuses
   rainfall what-ifs.** Holding terrain fixed across 500 real pixels and sweeping event
   rainfall from 0 to 1,000 mm, mean predicted probability rises on only 4 of 8 steps
   and falls sharply above 400 mm. Comparing a light storm (50/120 mm) against a heavy
   one (400/800 mm) per pixel, probability *falls* for 282 of 500 pixels and rises for
   218, mean change −0.039. Heavier rain predicting less flooding is hydrologically
   backwards, and it is a direct consequence of rainfall carrying no within-event
   contrast: `rain_sum` has effectively become an event identifier. A free-form
   "what if it rained X mm" endpoint built on this would return confident answers
   pointing the wrong way, so the service exposes only observed rainfall scenarios and
   attaches the caveat to every response. The check is reproduced by the pipeline into
   `outputs/tables/rainfall_response.csv` and `outputs/figures/rainfall_response.png`.
3. **Adding hydrological predictors is the highest-value next step.** Flow accumulation,
   distance to stream, drainage density and topographic wetness index are standard in
   this literature and absent here. They would give the model physical pathways rather
   than elevation as a proxy for all of them.

---

## 7. Limitations

**Labels are proxies, not truth.** Sentinel-1 water masks under-detect flooding beneath
dense urban canopy and built structures, exactly where the social consequences are
greatest — Mathare and Kibera are the hardest places for SAR to see and the most
important to get right. Reported performance is agreement with the available masks, not
with actual flood extent.

**The threshold is not calibrated for deployment.** Training is 50/50; reality is 0.08%
positive. A probability of 0.5 from this model does not mean a 50% chance that a given
pixel floods in the real prevalence regime. Before operational use the threshold must be
re-derived against true prevalence and against the asymmetric cost of a miss versus a
false alarm — and given that missing a flooded settlement is worse than inspecting a dry
one, the operating point should almost certainly sit below 0.5, trading precision for
recall.

**Rainfall resolution, and a non-monotonic rainfall response.** CHIRPS at ~5 km cannot
represent the convective, highly local storms that drive Nairobi flash flooding, and its
footprint is visible in the output as rectangular blocking. More seriously, because
rainfall is near-uniform within each event and sampling is balanced per event, the model
never saw rainfall vary independently of everything else, and its learned response to
rainfall is not monotonic (§6). Until the sampling design gives rainfall real
within-event contrast, this system cannot answer questions of the form "what happens if
it rains harder", and the deployed API declines to pretend otherwise.

**Predictor coverage.** No flow accumulation, distance-to-stream, drainage density, TWI,
soil or drainage-infrastructure data. Elevation is doing the work several more specific
variables should share, which limits physical interpretability.

**Static predictors over a nine-year window.** Built-up cover and terrain are single-date
while events span 2016–2025, so urban growth across the period is not represented.

**Spatial autocorrelation is only partly handled.** The event split prevents temporal and
event-level leakage, but pixels within one event remain spatially dependent. Spatial
block cross-validation would tighten this further.

**This is a hazard layer, not a risk map.** Risk is hazard × exposure × vulnerability.
Joining population, settlement type and critical infrastructure is required before the
output can support equity-based planning — and publishing susceptibility alone, without
that join, risks directing resources toward property rather than toward people.

---

## 8. Deployment

The trained model is serialized with its feature order, threshold, training events and
held-out metrics in one artefact, so a server cannot silently load a model and feed it
columns in the wrong order. A FastAPI service exposes health, the model card, the valid
extent, available scenarios, point prediction and the GeoTIFF download; it runs under
Docker via `docker compose up --build`.

Three deployment decisions are worth stating. The service **reads the rasters directly**
rather than only serving the precomputed surface, so a query can be scored under any
observed rainfall scenario. It **refuses to extrapolate**: a point outside the raster
extent returns `status: "outside_extent"` with a null probability rather than a number
that would look like a prediction. And it **refuses the query it cannot answer**:
arbitrary rainfall values are rejected with HTTP 422, because the rainfall response is
not monotonic (§6). Every `/predict` response carries that caveat, and `/model`
publishes the measured limitation alongside the scores, so a probability cannot travel
without the caveat that governs it.

A Streamlit front end (`app/streamlit_app.py`, `docker compose --profile ui up`) renders
what the API returns. It holds no model and computes nothing: if the API is unreachable
it says so rather than falling back to an estimate. This replaces an earlier standalone
version that, whenever its model file was absent, produced a flood probability from
distance to the city centre plus random noise — a number indistinguishable from a
prediction on screen and unrelated to any model.

### Reproducibility

One seed drives sampling and every model. `python scripts/run_pipeline.py` runs audit →
sample → train → evaluate → predict in about 15 seconds and regenerates every table and
figure in this report. 41 tests cover the data contract, the leakage-free split,
sampling determinism, the selection rule and all six API endpoints.

---

## 9. Conclusion

A gradient-boosted tree on five open-data predictors separates flooded from dry pixels
on unseen flood events at ROC AUC 0.906 and F1 0.794, well clear of a linear baseline at
0.790. The result is reproducible from raw GeoTIFFs by one command and queryable through
a containerised API.

The honest headline is narrower than the metrics suggest. The model is driven almost
entirely by elevation, because the sampling design gives rainfall no within-event
variation to work with, so what has been built is a terrain-susceptibility surface with
a rainfall input that barely moves it. That is a useful screening product for a city
that currently has none, and it is not a flood forecast. The route from one to the other
runs through hydrological predictors, a calibrated operating threshold, and exposure and
vulnerability layers — in that order.

---

### Figures

| Figure | File |
|---|---|
| Susceptibility map and value distribution | `outputs/figures/susceptibility_map.png` |
| ROC curves, all candidates | `outputs/figures/roc_curves.png` |
| Model comparison across metrics | `outputs/figures/model_comparison.png` |
| Confusion matrix, deployed model | `outputs/figures/confusion_matrix.png` |
| Reliability curve and Brier scores | `outputs/figures/reliability_curve.png` |
| Permutation and native importance | `outputs/figures/feature_importance.png` |
| SHAP summary | `outputs/figures/shap_summary.png` |
| Rainfall response (non-monotonic) | `outputs/figures/rainfall_response.png` |
| Predictor distributions by class | `outputs/figures/predictor_distributions.png` |

### Tables

`outputs/tables/` — raster audit, event audit, rainfall consistency, rainfall response,
training table, model comparison, permutation importance, SHAP importance, API demo
transcript. `models/model_card.json` carries the deployed
model's provenance and selection rationale.
