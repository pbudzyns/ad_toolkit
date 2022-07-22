import math

import numpy as np
import pytest

from sops_anomaly.evaluation import Result


@pytest.mark.parametrize("predicted,targets,expected_acc", (
    ([0, 0, 0, 0], [1, 1, 1, 1], 0),
    ([0, 0, 0, 1], [0, 0, 1, 1], 0.75),
    ([0, 0, 1, 1], [1, 1, 1, 1], 0.5),
    ([1, 1, 1, 1], [1, 1, 1, 1], 1.0),
))
def test_get_accuracy(predicted, targets, expected_acc):
    result = Result(predicted, targets)
    assert math.isclose(result.accuracy, expected_acc)


@pytest.mark.parametrize("predicted,targets,expected_recall", (
    ([0, 0, 0, 0], [0, 0, 1, 1], 0),
    ([0, 0, 0, 1], [0, 0, 1, 1], 0.5),
    ([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
    ([1, 1, 1, 1], [0, 0, 1, 1], 1.0),
))
def test_get_recall(predicted, targets, expected_recall):
    result = Result(predicted, targets)
    assert math.isclose(result.recall, expected_recall)


@pytest.mark.parametrize("predicted,targets,expected_precision", (
    ([1, 0, 0, 0], [0, 0, 1, 1], 0),
    ([0, 0, 0, 1], [0, 0, 1, 1], 1.0),
    ([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
    ([1, 1, 1, 1], [0, 0, 1, 1], 0.5),
))
def test_get_precision(predicted, targets, expected_precision):
    result = Result(predicted, targets)
    assert math.isclose(result.precision, expected_precision)


@pytest.mark.parametrize("predicted,targets,expected_f1", (
    ([0, 0, 0, 0], [0, 0, 1, 1], 0),
    ([0, 0, 0, 1], [0, 0, 1, 1], 0.67),
    ([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
    ([1, 1, 1, 1], [0, 0, 1, 1], 0.67),
))
def test_get_f1(predicted, targets, expected_f1):
    result = Result(predicted, targets)
    assert math.isclose(result.f1, expected_f1)


@pytest.mark.parametrize("predicted,targets,expected_roc", (
    ([0, 0, 0, 0], [0, 0, 1, 1], 0.5),
    ([0, 0, 0, 1], [0, 0, 1, 1], 0.75),
    ([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
    ([1, 1, 1, 1], [0, 0, 1, 1], 0.5),
))
def test_get_f1(predicted, targets, expected_roc):
    result = Result(predicted, targets)
    assert math.isclose(result.roc_auc, expected_roc)


def test_aggregate_results():
    preds1, targets1 = [0, 0, 0], [1, 1, 1]
    preds2, targets2 = [1, 1, 0], [1, 1, 0]
    result = Result(preds1, targets1) + Result(preds2, targets2)
    assert np.all(result.y_predicted == np.array(preds1 + preds2))
    assert np.all(result.y_labels == np.array(targets1 + targets2))
