from __future__ import annotations

import torch


class EventWeightedLossFunction(torch.nn.Module):
    def __init__(self, loss_cls, *args, **kwargs):
        """
        Extend every loss instance with an event weighted forward pass.

        Args:
            loss_cls (torch.nn.Module): Loss Function Module class
        """
        super().__init__()
        self.loss_inst = loss_cls(*args, **kwargs)

    def forward(self, *args, **kwargs):
        loss = self.loss_inst(*args, **kwargs, reduction="none")
        if torch.isnan(loss):
            from IPython import embed
            embed(header="Loss value IsNan")

        if event_weights := kwargs.get("event_weights") is not None:
            loss = torch.flatten(loss)
            event_weights = torch.flatten(event_weights)
            loss = torch.dot(loss, event_weights)
        return loss.mean()
