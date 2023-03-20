import torch
import numpy as np
import itertools
from torch.optim import Adam
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from .eval import evaluate_model
from .model import LTRModel


def create_results(data, net, train_fn, prediction_fn, results_file, *train_params):
    metrics = train_fn(net, *train_params, data)
    net.eval()
    test_metrics, test_qq = evaluate_model(
        data, prediction_fn, "test", print_results=False, q_level=True
    )
    test_q = {}
    for m in train_params[0].metrics:
        test_q[m] = test_qq[m]
        print(f'\t"{m}": {test_metrics[m]}')

    with open(results_file, "w") as writer:
        json.dump(
            {
                "metrics": metrics,
                "test_metrics": test_metrics,
                "test_query_level_metrics": test_q,
            },
            writer,
            indent=2,
        )

    return {
        "metrics": metrics,
        "test_metrics": test_metrics,
        "test_query_level_metrics": test_q,
    }


def seed(random_seed):
    import random

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def autolabel(ax, rects, labels):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for l, rect in zip(labels, rects):
        height = rect.get_height()
        ax.annotate(
            "{}".format(l),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def compare_methods(
    labels, metrics, metrics_to_plot={"ndcg", "precision@05", "recall@05"}
):
    """
    Constructs bar plots to compare different methods. 
    
    labels: list/tuple of length N
    metrics: list/tuple of length N, containing dictionary containing the test set results 
    metrics_to_plot: set of metrics to plot - each metric creates a separate plot 
    """
    assert len(metrics) == len(labels)

    x = np.arange(len(metrics_to_plot))
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot), sharey=False)
    fig.set_figheight(7)
    fig.set_figwidth(15)

    colors = cm.get_cmap("Set1").colors

    for metric, ax, c in zip(metrics_to_plot, axes, colors):
        m = [_[metric][0] for _ in metrics]
        std = [_[metric][1] for _ in metrics]
        x = np.arange(len(labels))
        rects = ax.bar(x, m, label=metric, color=c)

        l = ["{0:.4f}({1:.4f})".format(_[metric][0], _[metric][1]) for _ in metrics]
        autolabel(ax, rects, l)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(np.linspace(0, 1, num=11))
        ax.set_ylim(ymin=min(m) - 0.05, ymax=max(m) + 0.05)
        ax.set_title(metric)


def plot_distribution(labels, q_metrics, metric="ndcg"):
    """
    Plots the distribution of NDCG scores
    
    labels: list/tuple of length N
    q_metrics: list/tuple of dictionaries with length N, containing the query level results
    metric: the metric to plot
    
    """

    n = len(labels)
    # nC2
    n_plots = int((n * (n - 1)) / 2)

    fig, axes = plt.subplots(nrows=n_plots, ncols=1)
    fig.set_figheight(8 * n_plots)
    fig.set_figwidth(10)

    colors = cm.get_cmap("Set1").colors

    for idx, (i, j) in enumerate(itertools.combinations(range(n), 2)):
        ax = axes[idx]

        im = q_metrics[i][metric]
        jm = q_metrics[j][metric]

        ax.hist(im, bins=50, label=labels[i], color=colors[i], alpha=0.55)
        ax.hist(jm, bins=50, label=labels[j], color=colors[j], alpha=0.55)

        ax.set_title(f"{labels[i]} vs {labels[j]}")
        ax.legend()
        ax.set_ylabel("Count")
        ax.set_xlabel("NDCG (binned)")
