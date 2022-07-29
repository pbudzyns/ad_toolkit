from numbers import Number
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimeSeriesPlot:
    _anomaly_colors = (
        'red', 'magenta', 'green', 'navy', 'dodgerblue', 'orange', 'brown')

    @classmethod
    def plot(
        cls,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        anomalies: Optional[Dict[str, np.ndarray]] = None,
        vertical_margin: int = 10,
        show_legend: bool = False,
        fig_size: Tuple[int, int] = (10, 5),
        data_style_kwargs: Optional[Dict[str, Any]] = None,
        anomaly_style_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ax = plt.gca()
        fig = plt.gcf()

        x_lim, y_lim = cls._get_x_y_lim(vertical_margin, data)
        if labels is not None:
            cls._plot_known_anomalies(labels, y_lim)
        if anomalies is not None:
            cls._plot_predicted_anomalies(
                anomalies, labels, y_lim, anomaly_style_kwargs)

        data_style = {'lw': 1, 'ls': '-'}
        if data_style_kwargs is not None:
            data_style.update(data_style_kwargs)
        plt.plot(data, **data_style)
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        fig.set_size_inches(*fig_size)
        if show_legend:
            plt.legend()

    @classmethod
    def _plot_predicted_anomalies(
            cls, anomalies, labels, y_lim, anomaly_style_kwargs) -> None:
        style = {'ls': '-.', 'lw': 0.5, 'alpha': 0.1}
        if anomaly_style_kwargs is not None:
            style.update(anomaly_style_kwargs)
        for i, (name, points) in enumerate(anomalies.items()):
            anomalies_idx = labels.index[points.astype(bool)]
            color = cls._anomaly_colors[i % len(cls._anomaly_colors)]
            plt.vlines(
                x=anomalies_idx,
                ymin=y_lim[0],
                ymax=y_lim[1],
                colors=color,
                label=name,
                **style,
            )

    @classmethod
    def _get_x_y_lim(
        cls, vertical_margin: int, x: pd.DataFrame,
    ) -> Tuple[Tuple[Number, Number], Tuple[Number, Number]]:
        y_lim = (
            float(x.min().min() - vertical_margin),
            float(x.max().max() + vertical_margin),
        )
        x_lim = (
            np.min(x.index),
            np.max(x.index),
        )
        return x_lim, y_lim

    @classmethod
    def _plot_known_anomalies(cls, labels, y_lim):
        anomalous_labels = labels[labels != 0].index
        plt.vlines(
            x=anomalous_labels,
            ymin=y_lim[0],
            ymax=y_lim[1],
            colors='#eda8a6',
            ls='-',
            lw=2,
        )
