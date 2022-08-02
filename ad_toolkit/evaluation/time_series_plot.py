from numbers import Number
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimeSeriesPlot:
    """Time series plotter. Allows for easy visualization of time series
    together with marked anomalous windows and detected anomalies."""

    # Set of colors to use for anomaly markers.
    _anomaly_colors = (
        'red', 'navy', 'green', 'orange', 'dodgerblue', 'brown', 'magenta')

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
        legend_style_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """The plotting function.

        Parameters
        ----------
        data
            Time series data.
        labels
            Labels marking known anomalies.
        anomalies
            Dictionary with predictions made by detectors.
        vertical_margin
            Vertical margin of the plot.
        show_legend
            Controls legend appearance.
        fig_size
            Size of the final plot.
        data_style_kwargs
            Kwargs for styling time series plot.
        anomaly_style_kwargs
            Kwargs for styling vertical lines that marks detected anomalies.
        legend_style_kwargs
            Kwargs for styling legend.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
        """
        fig, ax = plt.subplots()

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
            legend_style = {}
            if legend_style_kwargs is not None:
                legend_style.update(legend_style_kwargs)
            plt.legend(**legend_style)

        return fig, ax

    @classmethod
    def _plot_predicted_anomalies(
            cls, anomalies, labels, y_lim, anomaly_style_kwargs) -> None:
        style = {
            'ls': '-.', 'lw': 0.5, 'alpha': 0.1,
            'ymin': y_lim[0], 'ymax': y_lim[1],
        }

        if anomaly_style_kwargs is not None:
            # Apply additional styling if provided.
            style.update(anomaly_style_kwargs)

        for i, (name, points) in enumerate(anomalies.items()):
            # Plot vertical lines marking detected anomalies.
            anomalies_idx = labels.index[points.astype(bool)]
            color = cls._anomaly_colors[i % len(cls._anomaly_colors)]
            plt.vlines(
                x=anomalies_idx,
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
        # Makes shaded background areas as known anomaly ranges.
        anomalous_labels = labels[labels != 0].index
        plt.vlines(
            x=anomalous_labels,
            ymin=y_lim[0],
            ymax=y_lim[1],
            colors='#eda8a6',
            ls='-',
            lw=2,
        )
