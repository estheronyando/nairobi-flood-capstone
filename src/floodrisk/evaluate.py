"""Step 4: diagnostics — curves, calibration, importance and SHAP.

Six figures, each answering one question a reviewer will ask:

* which model discriminates best on unseen events (ROC);
* how the candidates compare across metrics (bar panel);
* what kind of errors the chosen model makes (confusion matrix);
* whether its probabilities mean anything (reliability curve);
* what it is keying on (permutation + native importance);
* and how each predictor pushes an individual pixel (SHAP).

Colour is used to carry meaning, not decoration: a single accent marks the
selected model, everything else is grey, and the flood/dry classes keep the
same two colours in every figure.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, brier_score_loss, roc_curve

from .config import Config

log = logging.getLogger(__name__)

ACCENT = "#1f5f5b"     # the selected model
MUTED = "#9aa3ae"      # everything else
FLOOD = "#b4342c"
DRY = "#4a7fb5"
PALETTE = ["#1f5f5b", "#b4762c", "#4a7fb5", "#7a5c99", "#5c8a3a"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.autolayout": False,
    }
)


def _save(fig, cfg: Config, name: str) -> None:
    path = cfg.figures_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  figure -> %s", path.name)


def plot_predictor_distributions(cfg: Config, table: pd.DataFrame) -> None:
    """Do the predictors separate flood from dry at all, before any model?"""
    features = cfg.feature_columns
    fig, axes = plt.subplots(1, len(features), figsize=(3.1 * len(features), 3.0))
    for ax, feat in zip(np.atleast_1d(axes), features):
        for label, colour, name in ((0, DRY, "Dry"), (1, FLOOD, "Flood")):
            vals = table.loc[table["label"] == label, feat].to_numpy()
            ax.hist(vals, bins=40, alpha=0.6, color=colour, label=name, density=True)
        ax.set_title(feat)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Predictor distributions, flood vs dry sample pixels",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, cfg, "predictor_distributions.png")


def plot_roc(cfg: Config, y_test: np.ndarray, probabilities: dict[str, np.ndarray],
             best: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    order = sorted(probabilities, key=lambda k: k != best)  # best drawn last, on top
    for name in order:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        is_best = name == best
        ax.plot(
            fpr, tpr,
            color=ACCENT if is_best else MUTED,
            lw=2.4 if is_best else 1.3,
            alpha=1.0 if is_best else 0.75,
            label=f"{name} ({auc:.3f})" + ("  ← selected" if is_best else ""),
            zorder=3 if is_best else 2,
        )
    ax.plot([0, 1], [0, 1], ls=":", color="#c0c6cc", lw=1, label="chance (0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Discrimination on four unseen flood events")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _save(fig, cfg, "roc_curves.png")


def plot_model_comparison(cfg: Config, comparison: pd.DataFrame, best: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    models = list(comparison.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        vals = [comparison.loc[name, m] for m in metrics]
        is_best = name == best
        bars = ax.bar(
            x + i * width - 0.4 + width / 2, vals, width * 0.92,
            color=ACCENT if is_best else MUTED,
            alpha=1.0 if is_best else 0.55,
            label=name + ("  ← selected" if is_best else ""),
            edgecolor="white", linewidth=0.6,
        )
        if is_best:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", fontsize=7.5, color=ACCENT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Held-out event performance: tree ensembles lead the linear baseline")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _save(fig, cfg, "model_comparison.png")


def plot_confusion(cfg: Config, y_test: np.ndarray, proba: np.ndarray,
                   threshold: float, best: str) -> None:
    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Dry", "Flood"], cmap="BuGn",
        colorbar=False, ax=ax, values_format="d",
    )
    ax.set_title(f"{best} at threshold {threshold:g}")
    ax.grid(False)
    _save(fig, cfg, "confusion_matrix.png")


def plot_reliability(cfg: Config, y_test: np.ndarray, probabilities: dict[str, np.ndarray],
                     best: str) -> None:
    """A susceptibility map is read as a probability, so calibration matters."""
    bins = int(cfg.section("evaluation")["calibration_bins"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls=":", color="#c0c6cc", lw=1, label="perfectly calibrated")
    for i, (name, proba) in enumerate(probabilities.items()):
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=bins, strategy="quantile")
        is_best = name == best
        ax.plot(
            mean_pred, frac_pos, marker="o", ms=4,
            color=ACCENT if is_best else MUTED,
            lw=2.2 if is_best else 1.1, alpha=1.0 if is_best else 0.6,
            label=f"{name} (Brier {brier_score_loss(y_test, proba):.3f})",
            zorder=3 if is_best else 2,
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed flood frequency")
    ax.set_title("Reliability on held-out events")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _save(fig, cfg, "reliability_curve.png")


def plot_importance(cfg: Config, result: dict, test: pd.DataFrame) -> None:
    """Permutation importance on the held-out events, plus native importances.

    Permutation importance is the one to trust here: it is measured on data
    the model never saw, and unlike impurity importance it is not biased
    toward high-cardinality continuous features such as elevation.
    """
    features = cfg.feature_columns
    best = result["best_name"]
    model = result["fitted"][best]
    X_test, y_test = result["X_test"], result["y_test"]

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=cfg.seed,
        scoring="roc_auc", n_jobs=1,
    )
    order = np.argsort(perm.importances_mean)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].barh(
        [features[i] for i in order], perm.importances_mean[order],
        xerr=perm.importances_std[order], color=ACCENT, alpha=0.85,
        error_kw={"ecolor": "#5c646f", "lw": 1},
    )
    axes[0].set_xlabel("Drop in ROC AUC when shuffled")
    axes[0].set_title(f"Permutation importance ({best}, held-out events)")

    native = {}
    for name in ("random_forest", "xgboost"):
        m = result["fitted"].get(name)
        if m is not None and hasattr(m, "feature_importances_"):
            native[name] = m.feature_importances_
    if native:
        x = np.arange(len(features))
        w = 0.8 / len(native)
        for i, (name, vals) in enumerate(native.items()):
            axes[1].bar(x + i * w - 0.4 + w / 2, vals, w * 0.9,
                        color=PALETTE[i], alpha=0.85, label=name)
        axes[1].set_xticks(x); axes[1].set_xticklabels(features, rotation=20, ha="right")
        axes[1].set_ylabel("native importance")
        axes[1].set_title("Impurity / gain importance, for contrast")
        axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, cfg, "feature_importance.png")

    pd.DataFrame(
        {"feature": features,
         "permutation_mean": perm.importances_mean,
         "permutation_std": perm.importances_std}
    ).sort_values("permutation_mean", ascending=False).to_csv(
        cfg.tables_dir / "feature_importance.csv", index=False
    )


def plot_shap(cfg: Config, result: dict) -> None:
    """SHAP beeswarm for the tree model, so individual pixels can be explained."""
    try:
        import shap
    except ImportError:  # pragma: no cover - shap is in requirements
        log.warning("shap not installed; skipping SHAP figure")
        return

    model = result["fitted"].get("xgboost")
    if model is None:
        log.warning("No tree model available for SHAP; skipping")
        return

    n = min(int(cfg.section("evaluation")["shap_sample"]), len(result["X_test"]))
    sample = result["X_test"][:n]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("SHAP failed (%s); skipping", exc)
        return

    fig = plt.figure(figsize=(7.2, 4.4))
    shap.summary_plot(values, sample, feature_names=cfg.feature_columns, show=False,
                      plot_size=None)
    plt.title("SHAP: how each predictor moves a pixel's flood probability",
              fontsize=11, fontweight="bold")
    _save(fig, cfg, "shap_summary.png")

    mean_abs = np.abs(values).mean(axis=0)
    pd.DataFrame({"feature": cfg.feature_columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(cfg.tables_dir / "shap_importance.csv", index=False)


def plot_susceptibility_map(cfg: Config, raster_path) -> None:
    """Render the probability surface, plus the distribution of its values."""
    import rasterio

    with rasterio.open(raster_path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        bounds = ds.bounds
    masked = np.ma.masked_where(arr == nodata, arr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eceff1")
    im = axes[0].imshow(
        masked, cmap=cmap, vmin=0, vmax=1,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )
    axes[0].set_title("Flood susceptibility, Nairobi (30 m, median rainfall scenario)")
    axes[0].set_xlabel("Easting (m, EPSG:32737)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].grid(False)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_label("P(flood)")

    vals = masked.compressed()
    axes[1].hist(vals, bins=50, color=ACCENT, alpha=0.85)
    axes[1].axvline(0.5, color=FLOOD, ls="--", lw=1.4, label="threshold 0.5")
    axes[1].set_xlabel("Predicted flood probability")
    axes[1].set_ylabel("Cells")
    axes[1].set_title(f"{100 * float((vals >= 0.5).mean()):.1f}% of cells at or above 0.5")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, cfg, "susceptibility_map.png")


def check_rainfall_response(cfg: Config, result: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Does the deployed model respond to rainfall the way hydrology requires?

    Holding every pixel's terrain fixed and sweeping the two rainfall
    predictors upward, flood probability should rise monotonically. This
    check exists because the answer here is *no*, and that finding governs
    how the service may be used: a model whose rainfall response is not
    monotonic cannot support a "what if it rained more" query, however
    natural that query is to ask.

    The cause is structural. CHIRPS rainfall is near-uniform across the city
    within any one event, and sampling is balanced per event, so rainfall
    carries no within-event contrast for the model to learn from and ends up
    acting as an event identifier instead of a driver.
    """
    model = result["fitted"][result["best_name"]]
    features = cfg.feature_columns
    sample = table.sample(min(500, len(table)), random_state=cfg.seed)

    grid = [(0, 0), (25, 60), (50, 120), (100, 240), (150, 300),
            (200, 400), (300, 600), (400, 800), (500, 1000)]
    rows = []
    for r7, rs in grid:
        X = sample[features].copy()
        X["rain7"], X["rain_sum"] = r7, rs
        p = model.predict_proba(X.to_numpy(dtype="float32"))[:, 1]
        rows.append({"rain7_mm": r7, "rain_sum_mm": rs,
                     "mean_probability": round(float(p.mean()), 4),
                     "median_probability": round(float(np.median(p)), 4)})
    df = pd.DataFrame(rows)

    means = df["mean_probability"].to_numpy()
    rising = int((np.diff(means) > 0).sum())
    steps = len(means) - 1

    dry, wet = sample[features].copy(), sample[features].copy()
    dry["rain7"], dry["rain_sum"] = 50, 120
    wet["rain7"], wet["rain_sum"] = 400, 800
    p_dry = model.predict_proba(dry.to_numpy(dtype="float32"))[:, 1]
    p_wet = model.predict_proba(wet.to_numpy(dtype="float32"))[:, 1]

    df.attrs["monotone_steps"] = rising
    df.attrs["total_steps"] = steps
    df.attrs["pixels_up"] = int((p_wet > p_dry).sum())
    df.attrs["pixels_down"] = int((p_wet < p_dry).sum())
    df.attrs["mean_change"] = float(np.mean(p_wet - p_dry))
    df.to_csv(cfg.tables_dir / "rainfall_response.csv", index=False)

    if rising < steps:
        log.warning(
            "Rainfall response is NOT monotonic (%d/%d steps rising; heavier rain lowers "
            "P on %d of %d pixels, mean change %+.4f). Rainfall overrides are therefore "
            "not exposed by the API - see report section 6.",
            rising, steps, df.attrs["pixels_down"], len(p_dry), df.attrs["mean_change"],
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(df["rain_sum_mm"], df["mean_probability"], marker="o",
                 color=ACCENT, lw=2, label="model response")
    axes[0].plot(df["rain_sum_mm"], np.linspace(means.min(), means.max(), len(means)),
                 ls=":", color=MUTED, lw=1.4, label="monotonic, for reference")
    axes[0].set_xlabel("Event rainfall total supplied (mm)")
    axes[0].set_ylabel("Mean P(flood) over 500 real pixels")
    axes[0].set_title(f"Rainfall response is not monotonic ({rising}/{steps} steps rising)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].hist(p_wet - p_dry, bins=40, color=ACCENT, alpha=0.85)
    axes[1].axvline(0, color=FLOOD, ls="--", lw=1.4)
    axes[1].set_xlabel("Change in P(flood): heavy rain minus light rain")
    axes[1].set_ylabel("Pixels")
    axes[1].set_title(f"{df.attrs['pixels_down']} of {len(p_dry)} pixels move the wrong way")
    fig.tight_layout()
    _save(fig, cfg, "rainfall_response.png")
    return df


def run(cfg: Config, table: pd.DataFrame, test: pd.DataFrame, result: dict) -> None:
    log.info("Diagnostics:")
    best = result["best_name"]
    threshold = float(cfg.section("evaluation")["threshold"])
    plot_predictor_distributions(cfg, table)
    plot_roc(cfg, result["y_test"], result["probabilities"], best)
    plot_model_comparison(cfg, result["comparison"], best)
    plot_confusion(cfg, result["y_test"], result["probabilities"][best], threshold, best)
    plot_reliability(cfg, result["y_test"], result["probabilities"], best)
    plot_importance(cfg, result, test)
    plot_shap(cfg, result)
    check_rainfall_response(cfg, result, table)
