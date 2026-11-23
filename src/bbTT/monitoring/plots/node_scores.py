import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import bbTT.utils.transformations as fn
from bbTT.monitoring.register import register_plot
from bbTT.monitoring.utils.plotting import append_text_to_legend


def _histogram_weights(ctx, normalize: bool, axis: str = "composition") -> torch.Tensor:
    """
    Per-event weights of one weighting axis, prepared for histogramming the output scores.

    The node-score plots split events by true class, so the relative weight of the processes
    inside a class (and between classes) depends on the question asked, see EvalContext.weights:
    "composition" shows the classes as the training sees them, "physics" as the analysis and
    the Asimov loss see them, i.e. every process with its expected yield.

    Args:
        ctx (EvalContext): Context holding the weights.
        normalize (bool): True if every histogram is later normalized to unit sum per
            class (weights are then used as they are). False keeps a count-like y-axis:
            on the composition axis the weights are rescaled to sum to the number of events
            ("effective events", the total still matches ``len(predictions)``); on the physics
            axis they are kept, the y-axis then reads expected events.
        axis (str, optional): "composition" or "physics". Defaults to "composition".

    Returns:
        torch.Tensor: Weights of shape (num_events,).
    """
    weights = ctx.weights(axis)
    if normalize or axis == "physics":
        return weights
    return weights * (weights.shape[0] / weights.sum())


def _per_class_weights(weights: torch.Tensor, mask: torch.Tensor, normalize: bool) -> torch.Tensor:
    """
    Weights of one class, optionally normalized to unit sum so shapes are comparable.

    Args:
        weights (torch.Tensor): Weights of all events, shape (num_events,).
        mask (torch.Tensor): Boolean mask selecting the class, shape (num_events,).
        normalize (bool): Divide by the sum over the selected events.

    Returns:
        torch.Tensor: Weights of the selected events.
    """
    class_weights = weights[mask]
    if normalize:
        class_weights = class_weights / class_weights.sum()
    return class_weights


def _y_label(normalize: bool, weight_axis: str) -> str:
    """
    Y-axis label matching the weighting of the histogram.

    Args:
        normalize (bool): Whether every histogram is normalized to unit sum.
        weight_axis (str): "composition" or "physics".

    Returns:
        str: Axis label.
    """
    if normalize:
        return f"frequency (normalized, {weight_axis} weights)"
    if weight_axis == "physics":
        return "expected events"
    return "effective events (composition weights)"




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

    weights = _histogram_weights(ctx, normalize=normalize)
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

            _ = axes[node_idx].hist(
                filtered_predictions,
                label=data_cls,
                weights=_per_class_weights(weights, correct_cls_mask, normalize=normalize),
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
    weight_axis: str = "composition",
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
        normalize (bool): normalize every process (and the combined background)
            to unit sum so shapes are comparable rather than counts. Defaults True.
        weight_axis (str): "composition" (training view) or "physics" (expected
            yields, what the Asimov loss sees), see _histogram_weights. Decides how
            the processes are mixed in the combined background. Defaults "composition".
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

    masks["background"] = torch.stack([masks[name] for name in background_names], dim=0).any(dim=0)
    node = {name: y_pred[mask][:, signal_idx] for name, mask in masks.items()}

    all_weights = _histogram_weights(ctx, normalize=normalize, axis=weight_axis)
    weights = {name: _per_class_weights(all_weights, mask, normalize=normalize) for name, mask in masks.items()}

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
        ax.set_ylabel(_y_label(normalize=normalize, weight_axis=weight_axis), size=20)

        if weight_axis == "physics" and not normalize:
            ax.set_yscale("log")  # signal yield is orders of magnitude below the background

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

@register_plot(
    "output_score_hh_node_physics",
    requires={"evaluation_state.binning_edges"},
)
def plot_signal_node_scores_binned_physics(ctx, **kwargs) -> tuple[Figure, Axes]:
    """
    Same as output_score_hh_node, but in expected events: every process enters with its physics
    weight and nothing is normalized. The combined background panel is b(x), the signal s(x), i.e.
    the histograms the Asimov significance is computed from.
    """
    return _plot_signal_node_scores(
        ctx,
        edges_key="evaluation_state.binning_edges",
        transform=fn.logit.forward,
        normalize=False,
        weight_axis="physics",
        **kwargs,
    )
