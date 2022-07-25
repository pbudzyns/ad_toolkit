import math

import numpy as np
import pytest

from sops_anomaly.evaluation import Result


@pytest.mark.parametrize("predicted,targets,expected_tp", (
    (np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 0]), 0),
    (np.array([0, 0, 0, 1, 0]), np.array([0, 0, 1, 1, 0]), 1),
    (np.array([0, 0, 1, 1, 0]), np.array([0, 0, 1, 1, 0]), 2),
    (np.array([1, 1, 1, 1, 0]), np.array([1, 0, 1, 1, 0]), 3),
    (np.array([1, 1, 1, 1, 0]), np.array([1, 1, 1, 1, 0]), 4),
))
def test_get_tp(predicted, targets, expected_tp):
    result = Result(predicted, targets)
    assert result.tp == expected_tp


@pytest.mark.parametrize("predicted,targets,expected_fp", (
    (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), 0),
    (np.array([0, 0, 1, 0]), np.array([0, 0, 1, 1]), 0),
    (np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]), 1),
    (np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), 2),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 1]), 3),
))
def test_get_fp(predicted, targets, expected_fp):
    result = Result(predicted, targets)
    assert result.fp == expected_fp


@pytest.mark.parametrize("predicted,targets,expected_tn", (
    (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), 0),
    (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 4),
    (np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]), 1),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 2),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 1]), 0),
))
def test_get_tn(predicted, targets, expected_tn):
    result = Result(predicted, targets)
    assert result.tn == expected_tn


@pytest.mark.parametrize("predicted,targets,expected_fn", (
    (np.array([0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1]), 5),
    (np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0]), 0),
    (np.array([0, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0]), 1),
    (np.array([0, 0, 0, 1, 0]), np.array([0, 0, 1, 1, 1]), 2),
    (np.array([1, 1, 1, 1, 0]), np.array([0, 0, 0, 1, 0]), 0),
))
def test_get_fn(predicted, targets, expected_fn):
    result = Result(predicted, targets)
    assert result.fn == expected_fn


@pytest.mark.parametrize("predicted,targets,expected_acc", (
    (np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), 0),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 0.75),
    (np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]), 0.5),
    (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), 1.0),
))
def test_get_accuracy(predicted, targets, expected_acc):
    result = Result(predicted, targets)
    assert math.isclose(result.accuracy, expected_acc)


@pytest.mark.parametrize("predicted,targets,expected_recall", (
    (np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), 0),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 0.5),
    (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
))
def test_get_recall(predicted, targets, expected_recall):
    result = Result(predicted, targets)
    assert math.isclose(result.recall, expected_recall)


@pytest.mark.parametrize("predicted,targets,expected_precision", (
    (np.array([1, 0, 0, 0]), np.array([0, 0, 1, 1]), 0),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 1.0),
    (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 0.5),
))
def test_get_precision(predicted, targets, expected_precision):
    result = Result(predicted, targets)
    assert math.isclose(result.precision, expected_precision)


@pytest.mark.parametrize("predicted,targets,expected_f1", (
    (np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), 0),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 0.67),
    (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 0.67),
))
def test_get_f1(predicted, targets, expected_f1):
    result = Result(predicted, targets)
    assert math.isclose(result.f1, expected_f1)


@pytest.mark.parametrize("predicted,targets,expected_roc", (
    (np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), 0.5),
    (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 0.75),
    (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1.0),
    (np.array([1, 1, 1, 1]), np.array([0, 0, 1, 1]), 0.5),
))
def test_get_roc_auc(predicted, targets, expected_roc):
    result = Result(predicted, targets)
    assert math.isclose(result.roc_auc, expected_roc)


def test_aggregate_results():
    preds1, targets1 = np.array([0, 0, 0]), np.array([1, 1, 1])
    preds2, targets2 = np.array([1, 1, 0]), np.array([1, 1, 0])
    result = Result(preds1, targets1) + Result(preds2, targets2)
    assert np.all(result.y_predicted == np.array([0, 0, 0, 1, 1, 0]))
    assert np.all(result.y_labels == np.array([1, 1, 1, 1, 1, 0]))
