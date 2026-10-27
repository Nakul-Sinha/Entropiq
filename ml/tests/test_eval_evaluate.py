import math

import numpy as np
import pandas as pd
import pytest

from tdp_ml.eval.metrics import (
    PROBABILITY_METRICS,
    QUERY_COLUMNS,
    RANKING_METRICS,
    evaluate,
    ndcg_at_k,
    popt,
    recall_at_loc,
)
from tdp_ml.eval.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    df = make_synthetic_dataset(n_repos=3, n_snapshots=4, n_files=40, seed=5)
    df["hot"] = df["pct_churn_decayed"].astype("float64") * df["pct_cc_sum"]
    df["prob"] = np.clip(df["pct_prior_latent_bugs"].astype("float64"), 0.0, 1.0)
    return df


def _query(paths: list[str], scores: list[float], grades: list[int], qid: int = 7) -> pd.DataFrame:
    n = len(paths)
    return pd.DataFrame(
        {
            "qid": np.full(n, qid, dtype=np.int64),
            "repo": "own__repo",
            "snapshot_idx": np.full(n, 2, dtype=np.int16),
            "snapshot_time": pd.Timestamp("2021-05-01", tz="UTC"),
            "path": paths,
            "grade": np.asarray(grades, dtype=np.int8),
            "pain": np.asarray(grades, dtype=np.float32),
            "latent_defects": np.asarray(grades, dtype=np.int32),
            "sloc": np.full(n, 10.0, dtype=np.float32),
            "score": scores,
        }
    )


def test_one_row_per_query_with_columns_and_dtypes(data: pd.DataFrame) -> None:
    out = evaluate(data, "hot")
    assert list(out.columns) == list(QUERY_COLUMNS + RANKING_METRICS)
    assert out["qid"].tolist() == sorted(data["qid"].unique().tolist())
    assert isinstance(out.index, pd.RangeIndex)
    for col in ("qid", "snapshot_idx", "n_files", "n_defective", "n_defects"):
        assert str(out[col].dtype) == "int64", col
    assert out["repo"].dtype == "str"
    assert isinstance(out["snapshot_time"].dtype, pd.DatetimeTZDtype)
    assert str(out["snapshot_time"].dt.tz) == "UTC"
    for m in RANKING_METRICS:
        assert out[m].dtype == np.float64
    first = data[data["qid"] == 0]
    row = out.iloc[0]
    assert row["n_files"] == 40
    assert row["n_defective"] == int((first["latent_defects"] > 0).sum())
    assert row["n_defects"] == int(first["latent_defects"].sum())
    assert row["ndcg_10"] == pytest.approx(ndcg_at_k(first["grade"], first["hot"], 10))
    effort, defects = first["sloc"], first["latent_defects"]
    assert row["recall_20loc"] == pytest.approx(recall_at_loc(first["hot"], effort, defects))
    assert row["popt"] == pytest.approx(popt(first["hot"], effort, defects))
    assert row["snapshot_time"] == first["snapshot_time"].iloc[0]


def test_pain_as_score_is_a_perfect_ranking(data: pd.DataFrame) -> None:
    out = evaluate(data, "pain")
    for k in (5, 10, 20):
        assert out[f"ndcg_{k}"].to_numpy() == pytest.approx(1.0)
    assert (out["spearman"] == 1.0).all()


def test_row_order_does_not_matter(data: pd.DataFrame) -> None:
    expected = evaluate(data, "hot", prob_col="prob")
    shuffled = data.sample(frac=1.0, random_state=3)
    shuffled.index = pd.Index([f"row{i}" for i in range(len(shuffled))])
    pd.testing.assert_frame_equal(evaluate(shuffled, "hot", prob_col="prob"), expected)


def test_ties_are_broken_by_path() -> None:
    df = _query(["b.py", "a.py", "c.py"], [1.0, 1.0, 1.0], [0, 1, 3])
    out = evaluate(df, "score")
    # sorted by path the grades are a=1, b=0, c=3
    assert out.loc[0, "ndcg_5"] == pytest.approx(ndcg_at_k([1, 0, 3], [1.0, 1.0, 1.0], 5))
    assert out.loc[0, "ndcg_5"] != pytest.approx(ndcg_at_k([0, 1, 3], [1.0, 1.0, 1.0], 5))


def test_probability_metrics_only_with_prob_col(data: pd.DataFrame) -> None:
    without = evaluate(data, "hot")
    assert not set(PROBABILITY_METRICS) & set(without.columns)
    with_prob = evaluate(data, "hot", prob_col="prob")
    assert list(with_prob.columns) == list(QUERY_COLUMNS + RANKING_METRICS + PROBABILITY_METRICS)
    pd.testing.assert_frame_equal(with_prob[list(without.columns)], without)
    assert with_prob["brier"].between(0, 1).all() and with_prob["ece"].between(0, 1).all()


def test_undefined_metrics_are_nan() -> None:
    df = _query(["a.py", "b.py"], [2.0, 1.0], [0, 0])
    out = evaluate(df, "score")
    assert math.isnan(out.loc[0, "ndcg_10"]) and math.isnan(out.loc[0, "popt"])
    assert out.loc[0, "precision_10"] == 0.0


def test_invalid_input(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"missing columns \['nope'\]"):
        evaluate(data, "nope")
    with pytest.raises(ValueError, match=r"\['has_latent_defect', 'pain'\]"):
        evaluate(data.drop(columns=["pain", "has_latent_defect"]), "hot", prob_col="prob")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(pd.concat([data.head(3), data.head(1)]), "hot")
    mixed = data[data["qid"] == 0].copy()
    mixed.loc[mixed.index[0], "repo"] = "other__repo"
    with pytest.raises(ValueError, match="mixes"):
        evaluate(mixed, "hot")
    broken = data.copy()
    broken.loc[broken.index[50], "hot"] = np.nan
    with pytest.raises(ValueError, match=r"qid 1: .*NaN"):
        evaluate(broken, "hot")
    bad_prob = data.assign(prob=data["prob"] + 0.5)
    with pytest.raises(ValueError, match="qid"):
        evaluate(bad_prob, "hot", prob_col="prob")
    with pytest.raises(ValueError, match="integer dtype"):
        evaluate(data.assign(qid=data["qid"].astype("float64")), "hot")
    with pytest.raises(ValueError, match="nulls"):
        evaluate(data.assign(path=None), "hot")


def test_empty_input_gives_empty_frame(data: pd.DataFrame) -> None:
    out = evaluate(data.iloc[:0], "hot", prob_col="prob")
    assert out.empty
    assert list(out.columns) == list(QUERY_COLUMNS + RANKING_METRICS + PROBABILITY_METRICS)
    assert str(out["qid"].dtype) == "int64" and out["ndcg_10"].dtype == np.float64
