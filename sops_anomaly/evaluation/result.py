import dataclasses

import numpy as np
import sklearn.metrics


@dataclasses.dataclass()
class Result:
    """Binary classification results representation. Allows for easy
    representation of basic metrics.
    """
    y_predicted: np.ndarray
    y_labels: np.ndarray

    @classmethod
    def _round(cls, number: float) -> float:
        return round(number, 2)

    @property
    def tp(self) -> int:
        """Computes a number of true positives."""
        return self.y_predicted[self.y_labels == 1].sum()

    @property
    def fp(self) -> int:
        """Computes a number of false positives."""
        return self.y_predicted[self.y_labels == 0].sum()

    @property
    def tn(self) -> int:
        """Computes a number of true negatives."""
        x = self.y_predicted[self.y_labels == 0]
        return len(x) - x.sum()

    @property
    def fn(self) -> int:
        """Computes a number of false negatives."""
        x = self.y_predicted[self.y_labels == 1]
        return len(x) - x.sum()

    @property
    def recall(self) -> float:
        """Compute recall score."""
        return self._round(
            sklearn.metrics.recall_score(
                self.y_labels, self.y_predicted, zero_division=1)
        )

    @property
    def f1(self) -> float:
        """Compute F1 score."""
        return self._round(
            sklearn.metrics.f1_score(
                self.y_labels, self.y_predicted, zero_division=1),
        )

    @property
    def accuracy(self) -> float:
        """Compute accuracy score."""
        return self._round(
            sklearn.metrics.accuracy_score(self.y_labels, self.y_predicted))

    @property
    def precision(self) -> float:
        """Compute precision score."""
        return self._round(
            sklearn.metrics.precision_score(
                self.y_labels, self.y_predicted, zero_division=1),
        )

    @property
    def roc_auc(self) -> float:
        """Roc area under the curve score."""
        if not np.any(self.y_labels):
            return np.nan

        return self._round(
            sklearn.metrics.roc_auc_score(self.y_labels, self.y_predicted)
        )

    def __repr__(self) -> str:
        return (
            f"Result(accuracy={self.accuracy},\n"
            f"\t(tp, fp, tn, fp)={(self.tp,self.fp,self.tn,self.fp)},\n"
            f"\tprecision={self.precision},\n"
            f"\trecall={self.recall},\n"
            f"\tf1={self.f1},\n"
            f"\troc_auc={self.roc_auc},\n"
            f"\ty_pred%={np.sum(self.y_predicted)/len(self.y_predicted)},\n"
            f"\ty_label%={np.sum(self.y_labels)/len(self.y_labels)},\n"
            f")"
        )

    def __add__(self, other: "Result") -> "Result":
        y_predicted = np.concatenate((self.y_predicted, other.y_predicted))
        y_labels = np.concatenate((self.y_labels, other.y_labels))
        return Result(y_predicted, y_labels)
