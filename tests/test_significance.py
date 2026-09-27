"""The paired comparison, and the dependence it is built to respect."""

import numpy as np
import pytest

from air_quality_intelligence.analysis.significance import (
    DEFAULT_BLOCK_HOURS,
    compare_to_baseline,
    format_comparison,
    moving_block_bootstrap,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


def test_a_clear_gain_gives_an_interval_above_zero():
    differences = np.full(500, 2.0) + rng().normal(0, 0.1, 500)

    low, high = moving_block_bootstrap(differences)

    assert low > 0
    assert low < 2.0 < high


def test_no_gain_gives_an_interval_spanning_zero():
    differences = rng().normal(0, 1.0, 500)

    low, high = moving_block_bootstrap(differences)

    assert low < 0 < high


def test_a_loss_gives_an_interval_below_zero():
    differences = np.full(500, -3.0) + rng().normal(0, 0.1, 500)

    _, high = moving_block_bootstrap(differences)

    assert high < 0


def test_block_resampling_is_wider_than_treating_hours_as_independent():
    """The reason blocks are used at all.

    A strongly autocorrelated series carries less information than its length
    suggests. Resampling single observations pretends otherwise and returns
    an interval narrower than the data earns.
    """
    noise = rng(1).normal(0, 1.0, 2000)
    autocorrelated = np.convolve(noise, np.ones(48) / 48, mode="same")

    blocked = moving_block_bootstrap(autocorrelated, block_hours=48)
    independent = moving_block_bootstrap(autocorrelated, block_hours=1)

    assert (blocked[1] - blocked[0]) > (independent[1] - independent[0])


def test_the_interval_is_deterministic_for_a_given_seed():
    differences = rng(2).normal(0.5, 1.0, 400)

    assert moving_block_bootstrap(differences, seed=7) == moving_block_bootstrap(
        differences, seed=7
    )


def test_non_finite_differences_are_dropped():
    differences = np.array([1.0, np.nan, 1.0, np.inf, 1.0] * 50)

    low, high = moving_block_bootstrap(differences)

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_degenerate_inputs_do_not_raise():
    assert np.isnan(moving_block_bootstrap(np.array([]))[0])
    assert moving_block_bootstrap(np.array([3.0])) == (3.0, 3.0)


def test_a_block_longer_than_the_series_is_clamped():
    differences = np.full(10, 1.0)

    low, high = moving_block_bootstrap(differences, block_hours=1000)

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def test_a_better_model_is_reported_as_better():
    actual = rng(3).normal(100, 20, 600)
    baseline = actual + rng(4).normal(0, 10, 600)
    model = actual + rng(5).normal(0, 3, 600)

    result = compare_to_baseline(actual, model, baseline, model_name="sarima")

    assert result.mae_difference > 0
    assert result.significant
    assert result.verdict == "better than the baseline"
    assert result.win_rate > 0.5


def test_an_identical_model_is_indistinguishable():
    actual = rng(6).normal(100, 20, 600)
    baseline = actual + rng(7).normal(0, 8, 600)

    result = compare_to_baseline(actual, baseline.copy(), baseline, model_name="copy")

    assert result.mae_difference == pytest.approx(0.0)
    assert not result.significant
    assert result.verdict == "indistinguishable from the baseline"


def test_a_worse_model_is_reported_as_worse():
    actual = rng(8).normal(100, 20, 600)
    baseline = actual + rng(9).normal(0, 3, 600)
    model = actual + rng(10).normal(0, 12, 600)

    result = compare_to_baseline(actual, model, baseline, model_name="noisy")

    assert result.mae_difference < 0
    assert result.verdict == "worse"


def test_winning_on_average_while_losing_more_often_is_visible():
    """A model can win on the mean and lose most hours, if its wins are big.

    Reporting only the mean would call that a win outright, which is why the
    win rate is carried alongside it.
    """
    # Nine hours in ten the model is slightly worse; the tenth it is much
    # better. Mean: 0.9 * -0.1 + 0.1 * +4.5 = +0.36, on a 10% win rate.
    baseline_error = np.full(400, 1.0)
    model_error = np.full(400, 1.1)
    baseline_error[::10] = 5.0
    model_error[::10] = 0.5

    actual = np.zeros(400)
    result = compare_to_baseline(
        actual, model_error, baseline_error, model_name="spiky"
    )

    assert result.win_rate < 0.5
    assert result.mae_difference > 0


def test_rows_either_model_could_not_predict_are_dropped_from_both():
    """Scoring one model on rows the other missed changes the test set."""
    actual = np.arange(100, dtype="float64")
    baseline = actual + 1.0
    model = actual + 0.5
    model[:50] = np.nan

    result = compare_to_baseline(actual, model, baseline, model_name="partial")

    assert result.n == 50
    assert result.baseline_mae == pytest.approx(1.0)
    assert result.model_mae == pytest.approx(0.5)


def test_nothing_usable_returns_none():
    actual = np.full(10, np.nan)

    assert compare_to_baseline(actual, actual, actual, model_name="empty") is None


def test_the_default_block_matches_the_averaging_window():
    """24 hours, because that is the window the AQI is averaged over."""
    assert DEFAULT_BLOCK_HOURS == 24


def test_the_formatted_line_carries_the_interval_and_the_win_rate():
    actual = rng(11).normal(100, 20, 300)
    baseline = actual + rng(12).normal(0, 10, 300)
    model = actual + rng(13).normal(0, 4, 300)

    line = format_comparison(
        compare_to_baseline(actual, model, baseline, model_name="sarima")
    )

    assert "sarima" in line
    assert "wins" in line
    assert "[" in line and "]" in line
