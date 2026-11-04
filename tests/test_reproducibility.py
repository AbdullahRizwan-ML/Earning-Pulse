"""
Tests for reproducibility and evaluation hygiene.

- Walk-forward splits: training data always precedes test data.
- Fitting never sees test labels or other test rows.
- Each feature for quarter i is unchanged when quarter i's reported EPS (or any
  later data) changes, i.e. it is knowable before the announcement.
- The committed dataset rebuilds from the committed API cache, offline.
- The README Results block is exactly what results/metrics.json renders to.

Known problems are marked ``xfail(strict=True)``: they document a real defect
and will start failing (XPASS) once it is fixed, so the marker gets removed.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import requests

import src.data_loader as dl
from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluate import (
    EMBARGO_DAYS,
    RESULTS_JSON,
    fit_predict,
    load_dataset,
    readme_results_block,
    render_results_section,
    walk_forward_splits,
)
from src.model import FeaturePreprocessor, group_shuffle_splits, prepare_feature_frame


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_dataset()


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")
    monkeypatch.setattr(requests, "get", refuse)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL ORDERING
# ═══════════════════════════════════════════════════════════════════════════════
class TestTrainingPrecedesTest:

    def test_walk_forward_training_data_always_precedes_test_data(self, dataset: pd.DataFrame) -> None:
        dates = dataset["quarter_end"]
        splits = walk_forward_splits(dates)
        assert splits, "no walk-forward splits produced"
        for year, train_idx, test_idx in splits:
            last_train = dates.iloc[train_idx].max()
            first_test = dates.iloc[test_idx].min()
            assert last_train < first_test, f"{year}: train ends {last_train}, test starts {first_test}"
            assert last_train + pd.Timedelta(days=EMBARGO_DAYS) < first_test, f"{year}: embargo violated"
            assert not set(train_idx) & set(test_idx)

    def test_walk_forward_ordering_holds_on_unsorted_rows(self) -> None:
        dates = pd.Series(pd.date_range("2000-03-31", periods=100, freq=pd.offsets.QuarterEnd())).sample(frac=1, random_state=1)
        dates = dates.reset_index(drop=True)
        for _, train_idx, test_idx in walk_forward_splits(dates, first_test_year=2010):
            assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min()

    def test_every_row_from_first_test_year_is_tested_exactly_once(self, dataset: pd.DataFrame) -> None:
        dates = dataset["quarter_end"]
        tested = np.concatenate([t for _, _, t in walk_forward_splits(dates)])
        assert len(tested) == len(set(tested))
        expected = np.flatnonzero((dates.dt.year >= 2010).to_numpy())
        assert sorted(tested) == sorted(expected)

    @pytest.mark.xfail(strict=True, reason=(
        "The README's 0.74 comes from GroupShuffleSplit, which trains on quarters "
        "later than the quarters it tests."))
    def test_readme_cv_protocol_training_precedes_test(self, dataset: pd.DataFrame) -> None:
        X = prepare_feature_frame(dataset, FEATURE_COLUMNS)
        dates = dataset["quarter_end"]
        for train_idx, val_idx in group_shuffle_splits(X, dataset[TARGET_COLUMN], dataset["ticker"]):
            assert dates.iloc[train_idx].max() < dates.iloc[val_idx].min()


# ═══════════════════════════════════════════════════════════════════════════════
# NO TEST INFORMATION IN FITTING
# ═══════════════════════════════════════════════════════════════════════════════
def _synthetic_rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df[TARGET_COLUMN] = (df["eps_consistency"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


class TestNoTestLeakageInFitting:

    def test_predictions_ignore_test_labels(self) -> None:
        train, test = _synthetic_rows(200, 0), _synthetic_rows(40, 1)
        flipped = test.copy()
        flipped[TARGET_COLUMN] = 1 - flipped[TARGET_COLUMN]
        np.testing.assert_array_equal(
            fit_predict(train, test, FEATURE_COLUMNS),
            fit_predict(train, flipped, FEATURE_COLUMNS),
        )

    def test_prediction_for_a_row_ignores_other_test_rows(self) -> None:
        train, test = _synthetic_rows(200, 0), _synthetic_rows(40, 1)
        perturbed = test.copy()
        perturbed.iloc[1:, :len(FEATURE_COLUMNS)] *= 100  # extreme values in every other test row
        a = fit_predict(train, test, FEATURE_COLUMNS)
        b = fit_predict(train, perturbed, FEATURE_COLUMNS)
        assert a[0] == b[0]

    def test_preprocessor_statistics_come_from_fit_rows_only(self) -> None:
        train = prepare_feature_frame(_synthetic_rows(200, 0), FEATURE_COLUMNS)
        extra = prepare_feature_frame(_synthetic_rows(40, 1), FEATURE_COLUMNS) * 1000
        a = FeaturePreprocessor().fit(train)
        b = FeaturePreprocessor().fit(train)
        b.transform(extra)
        assert a.clip_bounds == b.clip_bounds
        np.testing.assert_array_equal(a.scaler.center_, b.scaler.center_)
        np.testing.assert_array_equal(a.scaler.scale_, b.scaler.scale_)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE LOOK-AHEAD (synthetic API responses → real feature builder)
# ═══════════════════════════════════════════════════════════════════════════════
N_QUARTERS = 16
PROBE = 10  # quarter whose features are checked


def _write_fake_cache(cache: Path, eps_override: Dict[int, float], price_bump_after: bool = False) -> None:
    quarter_ends = pd.date_range("2015-03-31", periods=N_QUARTERS, freq=pd.offsets.QuarterEnd())
    quarterly = []
    for i, qe in enumerate(quarter_ends):
        estimate = 1.0 + 0.05 * i
        reported = eps_override.get(i, estimate + (0.1 if i % 3 else -0.1))
        quarterly.append({
            "fiscalDateEnding": qe.strftime("%Y-%m-%d"),
            "reportedEPS": f"{reported}",
            "estimatedEPS": f"{estimate}",
        })
    (cache / "TEST_EARNINGS.json").write_text(json.dumps({"quarterlyEarnings": quarterly[::-1]}))

    probe_cutoff = quarter_ends[PROBE] - pd.Timedelta(days=5)
    months = pd.date_range("2014-01-31", "2019-12-31", freq=pd.offsets.MonthEnd())
    series = {}
    for k, m in enumerate(months):
        close = 100 + k + (10 * np.sin(k))
        if price_bump_after and m > probe_cutoff:
            close *= 3
        series[m.strftime("%Y-%m-%d")] = {"4. close": f"{close}", "5. adjusted close": f"{close}"}
    (cache / "TEST_MONTHLY.json").write_text(json.dumps({"Monthly Adjusted Time Series": series}))

    days = pd.date_range("2014-01-01", "2019-12-31", freq="D")
    for sid, base in (("VIXCLS", 15.0), ("T10Y2Y", 1.0)):
        values = base + np.cos(np.arange(len(days)) / 30)
        pd.DataFrame({"observation_date": days.strftime("%Y-%m-%d"), sid: values}).to_csv(
            cache / f"fred_{sid}.csv", index=False)


def _build_probe_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, **kwargs: Any) -> pd.Series:
    cache = tmp_path / name
    cache.mkdir()
    _write_fake_cache(cache, **kwargs)
    monkeypatch.setattr(dl, "CACHE_DIR", cache)
    df = dl._build_training_data(output_csv=cache / "out.csv", cache_only=True, tickers=["TEST"])
    probe_date = pd.date_range("2015-03-31", periods=N_QUARTERS, freq=pd.offsets.QuarterEnd())[PROBE]
    return df.loc[df["quarter_end"] == probe_date].iloc[0]


@pytest.fixture
def probe_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network: None) -> Dict[str, pd.Series]:
    base = _build_probe_row(tmp_path, monkeypatch, "base", eps_override={})
    current = _build_probe_row(tmp_path, monkeypatch, "current", eps_override={PROBE: -9.99})
    future = _build_probe_row(
        tmp_path, monkeypatch, "future",
        eps_override={i: 9.99 for i in range(PROBE + 1, N_QUARTERS)}, price_bump_after=True,
    )
    return {"base": base, "current": current, "future": future}


def _feature_params(xfail: Dict[str, str]) -> List[Any]:
    return [
        pytest.param(f, marks=pytest.mark.xfail(strict=True, reason=xfail[f])) if f in xfail else f
        for f in FEATURE_COLUMNS
    ]


class TestFeaturesAreKnownBeforeAnnouncement:

    def test_probe_label_changes_with_reported_eps(self, probe_rows: Dict[str, pd.Series]) -> None:
        # sanity: the perturbation really changes the quarter being predicted
        assert probe_rows["base"][TARGET_COLUMN] == 1
        assert probe_rows["current"][TARGET_COLUMN] == 0

    @pytest.mark.parametrize("feature", _feature_params({
        "eps_growth_yoy": "Computed from the current quarter's reported EPS (src/data_loader.py).",
    }))
    def test_feature_ignores_current_quarter_reported_eps(
        self, feature: str, probe_rows: Dict[str, pd.Series],
    ) -> None:
        assert probe_rows["base"][feature] == probe_rows["current"][feature]

    @pytest.mark.parametrize("feature", FEATURE_COLUMNS)
    def test_feature_ignores_later_quarters_and_prices(
        self, feature: str, probe_rows: Dict[str, pd.Series],
    ) -> None:
        assert probe_rows["base"][feature] == probe_rows["future"][feature]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PROVENANCE
# ═══════════════════════════════════════════════════════════════════════════════
class TestDatasetRebuildsFromCache:

    def test_one_ticker_rebuilds_identically_offline(
        self, tmp_path: Path, dataset: pd.DataFrame, no_network: None,
    ) -> None:
        rebuilt = dl._build_training_data(
            output_csv=tmp_path / "aapl.csv", cache_only=True, tickers=["AAPL"])
        committed = pd.read_csv(dl.TRAINING_CSV)
        committed = committed[committed["ticker"] == "AAPL"].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "aapl.csv"), committed)
        assert len(rebuilt) == len(committed)

    def test_cache_only_raises_instead_of_fetching(self, tmp_path: Path, no_network: None) -> None:
        with pytest.raises(dl.CacheMissError):
            dl._build_training_data(output_csv=tmp_path / "x.csv", cache_only=True, tickers=["NOT_CACHED"])

    def test_committed_csv_hash_matches_results(self) -> None:
        committed = json.loads(RESULTS_JSON.read_text())
        assert dl.file_sha256(dl.TRAINING_CSV) == committed["dataset"]["csv_sha256"]


# ═══════════════════════════════════════════════════════════════════════════════
# README RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
class TestReadmeResults:

    def test_readme_results_block_is_rendered_from_metrics_json(self) -> None:
        committed = json.loads(RESULTS_JSON.read_text())
        assert readme_results_block() == render_results_section(committed), (
            "README Results block is stale — run `python -m src.evaluate`")
