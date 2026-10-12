import torch

from bbTT.monitoring.register import register_builder


@register_builder("cross_entropy", provides={"cross_entropy"}, requires=None)
def build_cross_entropy(ctx, **kwargs):
    # TODO multi signal not implemented yet!
    # prediction is used to determine in which bin a truth value would land
    # ex. bins are [0,0.3,1] and prediction said 0, while truth is hh -> first bin signal count goes up by 1.
    # HINT: DOES REQUIRE LOGIT input
    y_true = ctx.targets
    y_pred = ctx.predictions
    weights = ctx.event_weights

    cross_entropy = torch.nn.functional.cross_entropy(y_pred, y_true, weight=None, reduction="none")

    weights = weights.reshape(cross_entropy.shape)
    cross_entropy = torch.mean(cross_entropy * weights)
    return {"cross_entropy": cross_entropy}
