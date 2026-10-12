import torch

from bbTT.monitoring.metrics.transformations.decoding import map_prediction_to_binary

# ------------------
# Binary classification
# ------------------


def binary_true_positive(target, pred, threshold=0.5):
    # target is positive, and identified as positive
    return torch.sum((target == 1) & (pred > threshold))


def binary_true_negative(target, pred, threshold=0.5):
    # target is negative, and identified as positive
    return torch.sum((target == 0) & (pred < threshold))


def binary_false_positive(target, pred, threshold=0.5):
    # target is negative, but identified as positive
    return torch.sum((target == 0) & (pred > threshold))


def binary_false_negative(target, pred, threshold=0.5):
    # target is positive, but identified as negative
    return torch.sum((target == 1) & (pred < threshold))


def binary_rates(target, pred, threshold=0.5):
    # calculate all classification rates for binary classification
    tp = binary_true_positive(target, pred, threshold)
    tn = binary_true_negative(target, pred, threshold)
    fp = binary_false_positive(target, pred, threshold)
    fn = binary_false_negative(target, pred, threshold)
    return tp, tn, fp, fn


### Common Metrics, one needs to calculate rates first!
def accuracy(tp, tn, fp, fn):
    # how many samples were correctly classified (true positive and true negative)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp, fp):
    # how many of the positive predictions were actually correct
    return tp / (tp + fp)


def sensitivity(tp, fn):
    # how many targets of a class were correctly identified
    return tp / (tp + fn)


def recall(tp, fn):
    # same as sensitivity
    return sensitivity(tp, fn)


def matthews_correlation_coefficient_binary(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return torch.tensor(0.0)
    return numerator / denominator


# ------------------
# MultiClass classification
# ------------------


def multiclass_rates(targets, predictions, weights=None, dtype:torch.dtype=torch.float32):
    """
    Calculate tp, tn, fp, fn for a 'Multi Label Classifier' by utilizing the *targets and *predictions* tensors.
    If a *weights* of events is given, this is applied to the rates calculation.
    The result is a tensor of shape (num_classes, 4) where each row corresponds to a class and contains tp, tn, fp, fn.

    To get the right score for the current class slice the returned tensor by class index.
    Example if the current class is in targets on index 1 one gets tp, tn, fp, fn by:
        tp, tn, fp, fn = rates[0]

    Args:
        targets (torch.tensor): truth tensor of shape (num_events, num_classes) in one-hot encoded form
        pred (torch.tensor): prediction tensor of shape (num_events, num_classes) with class scores/probabilities
        weights (torch.tensor, optional): Event based weights. Defaults to None.
        dtype (torch.dtype, optional): Dtype of the returned tensor HINT: For weighted counts int32 will truncate the value. Defaults to torch.float32.

    Returns:
        torch.tensor: tensor of shape (num_classes, 4) with tp, tn, fp, fn per class
    """
    # get predicted class by ceil arg max index and floor everything else
    _prediction = map_prediction_to_binary(prediction=predictions)

    # bring weights in correct form for broadcasting
    if weights is None:
        weights = torch.ones(targets.shape[0])
    weights = weights.reshape(-1, 1)

    # calculate tp, tn, fp, fn for all classes
    tp_diag = torch.sum(torch.logical_and(targets == 1, _prediction == 1) * weights, dim=0)
    tn_rest = torch.sum(torch.logical_and(targets == 0, _prediction == 0) * weights, dim=0)
    fp_per_column = torch.sum(torch.logical_and(targets == 0, _prediction == 1) * weights, dim=0)
    fn_per_line = torch.sum(torch.logical_and(targets == 1, _prediction == 0) * weights, dim=0)

    # stack rates into tensor, slice by truth class index
    # returns tp, tn, fp, fn for each class in order
    rates = torch.stack((tp_diag, tn_rest, fp_per_column, fn_per_line), dim=1).to(dtype)
    return rates


def _F1_macro_score(precision, recall, weights=None):
    f1 = 2 * (precision * recall) / (precision + recall)
    if weights is None:
        return torch.mean(f1, dim=0)
    f1 = weights * f1
    return torch.sum(f1, dim=0)


def F1_macro_score(prediction, target, weights=None):
    # calculate precision and recall for all classes
    rates = multiclass_rates(target, prediction)
    tp, fp, fn = rates[:, 0], rates[:, 2], rates[:, 3]

    _precision = precision(tp=tp, fp=fp)
    _recall = recall(tp=tp, fn=fn)
    return _F1_macro_score(_precision, _recall, weights=weights)


# --------------
# Aggregation
# --------------


def calculate_metrics(target, pred, label=None, weights=None, normalize=False):
    metrics = {}
    rates = multiclass_rates(target, pred, weights=weights)
    num_cls = target.shape[-1]
    for c in range(num_cls):
        tp, tn, fp, fn = rates[c]

        # TODO wrong definition!
        if normalize:
            raise NotImplementedError("Normalization of metrics not implemented yet.")
            num_true = tp + fn
            tp, tn, fp, fn = tp / num_true, tn / num_true, fp / num_true, fn / num_true

        _precision = precision(
            tp,
            fp,
        )
        _sensitivity = sensitivity(tp, fn)
        _f1_score = _F1_macro_score(_precision, _sensitivity)
        if label is not None:
            c = label[c]
        metrics[c] = {
            "tp": tp.item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "precision": _precision.item(),
            "sensitivity": _sensitivity.item(),
            "f1_score": _f1_score.item(),
        }
    return metrics
