from datetime import timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tdp_core.contracts import LABEL_COLUMNS
from tdp_ml.eval.harness import PROTOCOLS, ProtocolRun, Split, protocol_splits, run_protocol
from tdp_ml.eval.metrics import QUERY_COLUMNS, RANKING_METRICS
from tdp_ml.eval.stats import wilcoxon_holm
from tdp_ml.eval.synthetic import make_synthetic_dataset

H = 365


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(n_repos=3, n_snapshots=5, n_files=30, seed=4)


def hotspot(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return test["pct_churn_decayed"].astype("float64") * test["pct_cc_sum"]


def test_protocol_splits(data: pd.DataFrame) -> None:
    assert PROTOCOLS == ("temporal", "lopo")
    (temporal,) = protocol_splits(data, "temporal", horizon_days=H)
    assert isinstance(temporal, Split) and temporal.name == "temporal"
    assert (data.loc[temporal.test, "snapshot_idx"] == 4).all()
    lopo = protocol_splits(data, "lopo")
    assert [s.name for s in lopo] == sorted(data["repo"].unique().tolist())
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol_splits(data, "kfold")


def test_run_protocol_temporal(data: pd.DataFrame) -> None:
    seen: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        seen.append((train, test))
        for repo, rows in train.groupby("repo"):
            last = test.loc[test["repo"] == repo, "snapshot_time"].max()
            assert (rows["snapshot_time"] + timedelta(days=H) <= last).all()
        assert not set(LABEL_COLUMNS) & set(test.columns)  # labels are hidden from the model
        return hotspot(train, test)

    run = run_protocol(data, "temporal", fit_predict, horizon_days=H, model="hotspot")
    assert isinstance(run, ProtocolRun) and run.protocol == "temporal" and run.model == "hotspot"
    assert len(seen) == 1
    pq = run.per_query
    assert list(pq.columns) == ["model", "protocol", "split", *QUERY_COLUMNS, *RANKING_METRICS]
    assert len(pq) == 3 and (pq["split"] == "temporal").all() and (pq["model"] == "hotspot").all()
    assert pq["qid"].is_monotonic_increasing
    _, test = seen[0]
    assert run.scores.index.equals(test.index)
    assert run.scores.name == "score" and run.scores.dtype == np.float64
    assert run.scores.to_numpy() == pytest.approx(hotspot(data, test).to_numpy())


def test_run_protocol_lopo(data: pd.DataFrame) -> None:
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> npt.NDArray[np.float64]:
        held_out = set(test["repo"])
        assert len(held_out) == 1 and not held_out & set(train["repo"])
        return hotspot(train, test).to_numpy()

    run = run_protocol(data, "lopo", fit_predict, model="hotspot")
    pq = run.per_query
    assert (pq["split"] == pq["repo"]).all() and len(pq) == 3
    temporal = run_protocol(data, "temporal", hotspot, model="hotspot")
    # both protocols test each repository's last snapshot, so their qids line up
    assert pq["qid"].tolist() == temporal.per_query["qid"].tolist()
    assert run.scores.index.tolist() == sorted(run.scores.index.tolist())


def test_oracle_scores_are_perfect(data: pd.DataFrame) -> None:
    run = run_protocol(data, "lopo", lambda train, test: data.loc[test.index, "pain"], model="oracle")
    assert run.per_query["ndcg_10"].to_numpy() == pytest.approx(1.0)


def test_series_scores_are_aligned_by_label(data: pd.DataFrame) -> None:
    forward = run_protocol(data, "temporal", hotspot)
    backward = run_protocol(data, "temporal", lambda tr, te: hotspot(tr, te).iloc[::-1])
    pd.testing.assert_frame_equal(forward.per_query, backward.per_query)
    with pytest.raises(ValueError, match="indexed by the test rows"):
        run_protocol(data, "temporal", lambda tr, te: hotspot(tr, te).reset_index(drop=True))


def test_run_protocol_errors(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="expected 30 scores"):
        run_protocol(data, "lopo", lambda tr, te: np.zeros(len(te) - 1))
    with pytest.raises(ValueError, match="NaN"):
        run_protocol(data, "lopo", lambda tr, te: np.full(len(te), np.nan))
    one_repo = data[data["repo"] == data["repo"].iloc[0]]
    with pytest.raises(ValueError, match="no split"):
        run_protocol(one_repo, "lopo", hotspot)
    one_snapshot = data[data["snapshot_idx"] == 0]
    with pytest.raises(ValueError, match="no split"):
        run_protocol(one_snapshot, "temporal", hotspot)
    with pytest.raises(ValueError, match="unknown protocol"):
        run_protocol(data, "random", hotspot)


def test_hotspot_beats_random_under_lopo() -> None:
    df = make_synthetic_dataset(n_repos=10, n_snapshots=4, n_files=80, seed=8)
    rng = np.random.default_rng(0)
    hot = run_protocol(df, "lopo", hotspot, model="hotspot")
    rnd = run_protocol(df, "lopo", lambda tr, te: rng.uniform(size=len(te)), model="random")
    row = wilcoxon_holm(hot.per_query, {"random": rnd.per_query}, "ndcg_10").iloc[0]
    assert row["n"] == 10 and row["mean_diff"] > 0 and row["p_value"] < 0.05
