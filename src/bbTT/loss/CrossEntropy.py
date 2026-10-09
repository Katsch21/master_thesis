from __future__ import annotations

import torch


class WeightedCrossEntropy(torch.nn.CrossEntropyLoss):
    """
    An extension to normal CrossEntropy enabling to add event weights and not only class specific weights.
    """

    def forward(self, prediction, target, event_weights: torch.Tensor | None = None):
        # save original reduction mode
        reduction = self.reduction
        if event_weights is not None:
            self.reduction = "none"
            # torch.nn.CrossEntropyLoss ALREADY has Softmax built in, so no need to apply it here
            loss = super().forward(prediction, target)
            self.reduction = reduction

            # dot product is only defined for flat tensors, so flatten
            loss = torch.flatten(loss)
            event_weights = torch.flatten(event_weights)
            loss = torch.dot(loss, event_weights)
            if self.reduction == "mean":
                loss = loss / torch.sum(event_weights)
        else:
            loss = super().forward(prediction, target)
        return loss
