"""Is a difference in forecast error real, or is it noise?

A model that scores better than persistence on 1,340 hourly observations has
not necessarily beaten it. Two things make the naive reading wrong:

Hourly forecast errors are not independent.
    Consecutive hours of a 24-hour rolling mean share almost all their
    observations, so a run of good hours is one event, not twenty-four
    independent successes. Any test that assumes independent samples will
    report a confidence it has not earned. That is why the interval here
    comes from a moving-block bootstrap: resampling whole blocks of
    consecutive hours keeps the dependence that treating hours as
    independent would throw away.

A mean difference hides its own distribution.
    A model can win on average while losing more often than it wins, if its
    wins are larger. Both numbers are reported, because they answer different
    questions: the interval says whether the average gain is real, the win
    rate says how often you would actually prefer it.

Nothing here decides anything. It gives an interval and a rate, and leaves
the reading to whoever is looking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Length of a bootstrap block, in hours.
#
# The AQI is a 24-hour rolling mean, so two errors less than 24 hours apart
# are computed from overlapping observations and cannot be treated as
# independent draws. Resampling in 24-hour blocks keeps that dependence
# inside the block, where it belongs.
DEFAULT_BLOCK_HOURS = 24

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PairedComparison:
    """One model measured against a baseline on the same observations."""

    model: str
    baseline: str
    n: int
    model_mae: float
    baseline_mae: float
    mae_difference: float
    ci_low: float
    ci_high: float
    win_rate: float
    blocks: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero in either direction."""

        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "indistinguishable from the baseline"

        return "better than the baseline" if self.mae_difference > 0 else "worse"


def moving_block_bootstrap(
    differences: np.ndarray,
    *,
    block_hours: int = DEFAULT_BLOCK_HOURS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for the mean of a dependent series.

    differences are per-observation, baseline error minus model error, so a
    positive mean means the model is better.

    Blocks of consecutive observations are resampled with replacement and
    concatenated to the original length. The interval that results is wider
    than an independent bootstrap would give, and that extra width is the
    honest part.
    """

    values = np.asarray(differences, dtype="float64")
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan")

    if values.size == 1:
        return float(values[0]), float(values[0])

    block = max(1, min(int(block_hours), values.size))
    starts = values.size - block + 1
    needed = int(np.ceil(values.size / block))

    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, starts, size=(resamples, needed))

    # Build every resample at once: each row is a set of block starts, and
    # adding 0..block-1 to each start expands them into index runs.
    index = (offsets[:, :, None] + np.arange(block)[None, None, :]).reshape(
        resamples, -1
    )[:, : values.size]

    means = values[index].mean(axis=1)

    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def compare_to_baseline(
    actual: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    model_name: str,
    baseline_name: str = "persistence",
    block_hours: int = DEFAULT_BLOCK_HOURS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedComparison | None:
    """Compare a model against a baseline on the observations both predicted.

    Rows where either prediction is missing are dropped from both, so the
    comparison stays paired. Scoring one model on rows the other could not
    predict is the arithmetic equivalent of changing the test set.
    """

    actual = np.asarray(actual, dtype="float64")
    model = np.asarray(model, dtype="float64")
    baseline = np.asarray(baseline, dtype="float64")

    usable = np.isfinite(actual) & np.isfinite(model) & np.isfinite(baseline)

    if not usable.any():
        return None

    actual, model, baseline = actual[usable], model[usable], baseline[usable]

    model_error = np.abs(actual - model)
    baseline_error = np.abs(actual - baseline)
    differences = baseline_error - model_error

    low, high = moving_block_bootstrap(
        differences, block_hours=block_hours, resamples=resamples, seed=seed
    )

    return PairedComparison(
        model=model_name,
        baseline=baseline_name,
        n=int(usable.sum()),
        model_mae=float(model_error.mean()),
        baseline_mae=float(baseline_error.mean()),
        mae_difference=float(differences.mean()),
        ci_low=low,
        ci_high=high,
        win_rate=float((differences > 0).mean()),
        blocks=int(max(1, min(block_hours, differences.size))),
    )


def format_comparison(comparison: PairedComparison, indent: str = "    ") -> str:
    """One line per model, in the terms the comparison actually supports."""

    return (
        f"{indent}{comparison.model:14s} MAE {comparison.model_mae:7.3f} vs "
        f"{comparison.baseline_mae:7.3f}  "
        f"gain {comparison.mae_difference:+6.3f} "
        f"[{comparison.ci_low:+6.3f}, {comparison.ci_high:+6.3f}]  "
        f"wins {comparison.win_rate:5.1%}  "
        f"{comparison.verdict}"
    )
