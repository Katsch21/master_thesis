import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import bbTT.utils.transformations as fn
from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.plotting import append_text_to_legend

# TODO general transformation passing: ctx does not currently track which
# transformation (if any) a given set of bin edges lives in. The logit transform
# applied to predictions below is hardcoded to match the space `active_edges` is
# known to live in today. Revisit once ctx carries transform metadata.


@register_plot(
    "output_score",
    requires=None,
    optional=False,
)
def plot_network_predictions(
    ctx,
    normalize=True,
    single_legend=False,
    **kwargs,
) -> tuple[Figure, Axes]:
    # create a figure with one subplot per node, showing the score distribution
    # nodes are ordered by target_map
    y_true = ctx.targets
    y_pred = ctx.predictions
    target_map = ctx.target_map

    fig, axes = plt.subplots(1, len(target_map), figsize=(8 * len(target_map), 8))
    title = kwargs.get("title")
    if title is not None:
        fig.suptitle(title)

    hist_cfg = {
        "bins": kwargs.get("bins", 20),
        "histtype": kwargs.get("histtype", "step"),
        "alpha": kwargs.get("alpha", 0.7),
    }

    weight = None
    # for each node, histogram the predictions of every true class
    for node, node_idx in target_map.items():
        for data_cls, data_idx in target_map.items():
            y_label = "frequency"

            if not normalize:
                axes[node_idx].set_yscale("log")
                axes[node_idx].set_ylim(top=len(y_pred))
            else:
                axes[node_idx].set_yscale("linear")
                axes[node_idx].set_ylim(top=1.0)
                y_label += " normalized"

            axes[node_idx].set_xlabel(f"{node} node")
            axes[node_idx].set_ylabel(y_label)
            axes[node_idx].grid()

            # select events whose TRUTH is this class, then take the node score
            correct_cls_mask = y_true[:, data_idx] == 1
            filtered_predictions = y_pred[correct_cls_mask][:, node_idx]

            if normalize:
                weight = np.full(filtered_predictions.shape, 1 / len(filtered_predictions))

            _ = axes[node_idx].hist(
                filtered_predictions,
                label=data_cls,
                weights=weight,
                **hist_cfg,
            )
        if not single_legend:
            axes[node_idx].legend()
    if single_legend:
        lines_labels = [fig.axes[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
    return fig, axes


def _split_signal_background(ctx, signal="hh"):
    """
    Return (y_true, y_pred, target_map, signal_idx, background_names).

    `background_names` is every class in `target_map` other than `signal`, so the
    plot survives a different class set instead of assuming dy/tt.
    """
    target_map = ctx.target_map
    signal_idx = target_map[signal]
    background_names = [name for name in target_map if name != signal]
    return ctx.targets, ctx.predictions, target_map, signal_idx, background_names


def _plot_signal_node_scores(
    ctx,
    edges_key: str,
    transform=None,
    signal: str = "hh",
    normalize: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Histogram the signal-node output score, split by true process.

    Two panels: (0) signal vs. combined background, (1) signal vs. each
    background process separately.

    Args:
        ctx (EvalContext): must expose `targets`, `predictions`, `target_map`
            and provide `edges_key`.
        edges_key (str): context key of the bin edges to histogram into.
        transform (callable | None): applied to predictions before
            histogramming, e.g. `fn.logit.forward`; None leaves raw scores.
        signal (str): name of the signal class in `target_map`. Defaults "hh".
        normalize (bool): weight each process by 1/n_events so shapes are
            comparable rather than raw counts. Defaults True.
        **kwargs: optional "title" (str), "histtype" (str, default "step"),
            "alpha" (float, default 0.7). Other keys are ignored.

    Returns:
        tuple[Figure, numpy.ndarray]: the figure and its array of 2 axes.
    """

    y_true, y_pred, target_map, signal_idx, background_names = _split_signal_background(ctx, signal)

    if transform is not None:
        y_pred = transform(y_pred)

    binning_edges = ctx.get(edges_key).flatten()

    # TRUTH masks -> signal-node score per process
    masks = {name: (y_true[:, idx] == 1) for name, idx in target_map.items()}
    node = {name: y_pred[masks[name]][:, signal_idx] for name in target_map}
    node["background"] = torch.cat([node[name] for name in background_names], dim=0)

    weights = {name: None for name in node}
    if normalize:
        weights = {name: np.full(value.shape, 1 / len(value)) for name, value in node.items()}

    hist_cfg = {
        "histtype": kwargs.get("histtype", "step"),
        "alpha": kwargs.get("alpha", 0.7),
        "bins": binning_edges,  # dynamic variable, comes from the context
    }

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    title = kwargs.get("title")
    if title is not None:
        fig.suptitle(title)

    # panel 0: signal vs combined background
    _ = axes[0].hist(node[signal], label="signal", weights=weights[signal], hatch="/", **hist_cfg)
    _ = axes[0].hist(node["background"], label="background", weights=weights["background"], hatch="\\", **hist_cfg)

    # panel 1: signal vs each background process
    hatches = ["/", "\\", "*", "o", "."]
    _ = axes[1].hist(node[signal], label=signal, weights=weights[signal], hatch=hatches[0], **hist_cfg)
    for i, name in enumerate(background_names, start=1):
        _ = axes[1].hist(
            node[name],
            label=name,
            weights=weights[name],
            hatch=hatches[i % len(hatches)],
            **hist_cfg,
        )

    counts_text = "\n".join(f"{name}: {len(node[name])}" for name in target_map)
    left_bound = float(binning_edges[0]) - 0.1
    right_bound = float(binning_edges[-1]) + 0.1
    for ax in axes:
        append_text_to_legend(ax, counts_text)
        ax.set_xlabel(f"{signal} node", size=20)
        ax.set_ylabel("frequency" + (" (normalized)" if normalize else ""), size=20)
        ax.set_xlim((left_bound, right_bound))
        ax.grid()
    return fig, axes


@register_plot(
    "output_score_hh_node_untransformed",
    requires={"original_edges"},
    optional=True,
)
def plot_signal_node_scores_raw(ctx, **kwargs) -> tuple[Figure, Axes]:
    """
    Signal-node score vs. the original ([0, 1] space) bin edges, no transform.

    `original_edges` is only produced by the `kernel_state` builder, which needs
    a model with a binning layer; marked optional so it is cleanly skipped
    otherwise.
    """
    return _plot_signal_node_scores(ctx, edges_key="original_edges", transform=None, **kwargs)


@register_plot(
    "output_score_hh_node",
    requires={"evaluation_state.binning_edges"},
)
def plot_signal_node_scores_binned(ctx, **kwargs) -> tuple[Figure, Axes]:
    """
    Signal-node score vs. active bin edges, predictions logit-transformed.

    Edges come from the `score_bins` builder, which always resolves (model
    active_edges, else linspace(0, 1)). NOTE: in the linspace fallback (model
    without a binning layer) the edges are in raw-score space while the data is
    logit-transformed - a known space mismatch, acceptable for monitoring only.
    See the module-level TODO on transform tracking.
    """
    return _plot_signal_node_scores(
        ctx,
        edges_key="evaluation_state.binning_edges",
        transform=fn.logit.forward,
        **kwargs,
    )
