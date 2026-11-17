import numpy as np

from leadscore.evaluate import score, tree_rules
from leadscore.features import encode, split
from leadscore.pipeline import compare, run
from leadscore.train import default_forest, default_tree, tuned_tree


def test_score_on_known_predictions():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    s = score(y_true, y_pred)
    assert s.accuracy == 0.8
    assert round(s.recall, 4) == round(2 / 3, 4)
    assert round(s.precision, 4) == round(2 / 3, 4)


def test_unpruned_tree_memorizes_training_data(leads):
    x, y = encode(leads)
    x_train, _, y_train, _ = split(x, y)
    tree = default_tree(x_train, y_train)
    assert score(y_train, tree.predict(x_train)).accuracy == 1.0


def test_models_are_reproducible(leads):
    x, y = encode(leads)
    x_train, x_test, y_train, _ = split(x, y)
    a = default_forest(x_train, y_train).predict(x_test)
    b = default_forest(x_train, y_train).predict(x_test)
    assert (a == b).all()


def test_tuned_tree_rules_are_readable(leads, small_grids):
    x, y = encode(leads)
    x_train, _, y_train, _ = split(x, y)
    model, params = tuned_tree(x_train, y_train)
    assert params["max_depth"] in (3, 5)
    assert "time_spent_on_website" in tree_rules(model, list(x.columns))


def test_pipeline_runs_end_to_end(leads, small_grids):
    metrics = run(leads, n_jobs=1)
    assert metrics["rows"] == 500
    assert set(metrics["test_scores"]) == {
        "decision_tree_default",
        "decision_tree_tuned",
        "random_forest_default",
        "random_forest_tuned",
    }
    assert 0 <= metrics["test_scores"]["random_forest_tuned"]["recall"] <= 1


def test_compare_flags_differences():
    ref = {"_source": "x", "test_scores": {"m": {"recall": 0.85}}}
    assert compare({"test_scores": {"m": {"recall": 0.85}}}, ref) == []
    assert compare({"test_scores": {"m": {"recall": 0.80}}}, ref) != []
