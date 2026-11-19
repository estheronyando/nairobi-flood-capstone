# Prepared data and attribution

`nairobi_flood_data/` contains the 51 prepared GeoTIFFs used by the scripted
pipeline and the live point-prediction API (about 19 MB total). They are cropped,
co-registered project derivatives on a 30 m Nairobi grid, not authoritative source
distributions. Including them makes a fresh clone, Docker build, and Render deployment
able to run the deployed model without silently changing the API into a cached map.

## Source acknowledgement and reuse

| Prepared files | Upstream source / transformation | Attribution and terms |
|---|---|---|
| `dem_srtm_*`, `slope_*` | NASA Shuttle Radar Topography Mission (SRTM); slope is derived from the DEM. | [NASA SRTM](https://data.nasa.gov/dataset/shuttle-radar-topography-mission-srtm-images) |
| `landcover_worldcover_*` | ESA WorldCover, cropped and resampled to the project grid. | [ESA WorldCover — CC BY 4.0 and required acknowledgement](https://esa-worldcover.org/en/data-access) |
| `builtup_mask_worldcover_*` | Project-built binary mask from WorldCover and OpenStreetMap built-up information. | [© OpenStreetMap contributors — ODbL](https://www.openstreetmap.org/copyright); ESA WorldCover acknowledgement above. |
| `rain7_*`, `rain_sum_*` | CHIRPS v2 rainfall fields, aggregated by event and resampled to the analysis grid. | [Climate Hazards Center: CHIRPS v2](https://www.chc.ucsb.edu/data/chirps) |
| `label_event_water_*` | Project-prepared transient-water labels derived from Sentinel-1 SAR scenes. They are a flood proxy, not field-verified ground truth. | [Copernicus Sentinel data legal notice](https://cds.climate.copernicus.eu/licences/ec-sentinel) |
| `perm_water_mask_occ30_*` | Project-prepared permanent-water mask (surface-water occurrence ≥ 30%), used only for masking and not as a model feature. | The original catalogue record was not retained with this project. Confirm its provenance before reusing this layer outside this academic submission. |

Users must comply with the upstream terms. In particular, OpenStreetMap requires visible
attribution and notice of the Open Database License where its data are reused. This
repository does not grant a separate licence to upstream data.
