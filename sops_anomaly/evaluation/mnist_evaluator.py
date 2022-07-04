from sops_anomaly.datasets import MNIST
from sops_anomaly.evaluation.result import Result
from sops_anomaly.models import BaseDetector


def evaluate_model_on_mnist(
    model: BaseDetector,
    n_train_samples: int,
    n_test_samples: int,
):
    results = {}

    for anomaly_class in range(10):
        mnist = MNIST(anomaly_class=anomaly_class)
        x_train = mnist.get_train_samples(n_train_samples)
        x_test, y_test = mnist.get_test_samples(n_test_samples)

        model.train(x_train, epochs=50)

        y_predicted = model.detect(x_test)

        result = Result(y_predicted=y_predicted, y_labels=y_test)
        print(result)
        results[anomaly_class] = result

    return results


if __name__ == '__main__':
    from sops_anomaly.models import AutoEncoder
    model = AutoEncoder(input_size=MNIST.sample_size(), threshold=0.9)

    results = evaluate_model_on_mnist(
        model,
        n_train_samples=1000,
        n_test_samples=200,
    )
