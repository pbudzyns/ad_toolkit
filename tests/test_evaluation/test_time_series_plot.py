import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ad_toolkit.evaluation import TimeSeriesPlot

DATA_SIZE = 300


def random_anomalies() -> np.ndarray:
    return np.random.randint(2, size=DATA_SIZE)


@pytest.mark.parametrize("labels", (
    None, pd.Series(random_anomalies()),
))
@pytest.mark.parametrize("anomalies", (
    None, {}, {'a': random_anomalies()},
    {'a': random_anomalies(), 'b': random_anomalies()},
))
@pytest.mark.parametrize("vertical_margin", (0, 10))
@pytest.mark.parametrize("show_legend", (True, False))
@pytest.mark.parametrize("data_style_kwargs", (None, {}, {'lw': 1}))
@pytest.mark.parametrize("anomaly_style_kwargs", (None, {}, {'lw': 1}))
@pytest.mark.parametrize("legend_style_kwargs", (None, {}, {'fontsize': 10}))
def test_time_series_plot(
    labels, anomalies, vertical_margin, show_legend, data_style_kwargs,
    anomaly_style_kwargs, legend_style_kwargs,
):
    data = pd.DataFrame(np.random.random((DATA_SIZE, 20)))
    fig, ax = TimeSeriesPlot.plot(
        data, labels=labels, anomalies=anomalies, show_legend=show_legend,
        vertical_margin=vertical_margin, data_style_kwargs=data_style_kwargs,
        anomaly_style_kwargs=anomaly_style_kwargs,
        legend_style_kwargs=legend_style_kwargs,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    plt.close(fig)
