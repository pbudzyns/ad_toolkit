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

    @property
    def recall(self) -> float:
        """Compute recall score."""
        return sklearn.metrics.recall_score(self.y_labels, self.y_predicted)

    @property
    def f1(self) -> float:
        """Compute F1 score."""
        return sklearn.metrics.f1_score(self.y_labels, self.y_predicted)

    @property
    def accuracy(self) -> float:
        """Compute accuracy score."""
        return sklearn.metrics.accuracy_score(self.y_labels, self.y_predicted)

    @property
    def precision(self) -> float:
        """Compute precision score."""
        return sklearn.metrics.precision_score(self.y_labels, self.y_predicted)

    def __repr__(self) -> str:
        return (
            f"Result(accuracy={round(self.accuracy, 2)},\n"
            f"\tprecision={round(self.precision, 2)},\n"
            f"\trecall={round(self.recall, 2)},\n"
            f"\tf1={round(self.f1, 2)},\n"
            f"\ty_pred={np.sum(self.y_predicted)/len(self.y_predicted)},\n"
            f"\ty_label={np.sum(self.y_labels)/len(self.y_labels)},\n"
            f")"
        )
