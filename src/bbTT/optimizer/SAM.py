from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    """
    Implementation of Sharpness Aware Minimization (SAM) optimizer (https://arxiv.org/abs/2010.01412).
    Got Code from https://github.com/davda54/sam/blob/main/sam.py
    Usage is described in the README of the repository.

    Needs to adjust optimizer init and trainloop.


    STATUS:
    parked, not wired into configs, loops or the optimizer factory.
    When reviving: call it through step(closure) so that it works with every loss, and note that
    sam_training_routine below is outdated (wrong model call signature).
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)

        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # scout from "w" to "w + e(w)"
        # calculate sharpness max(L(w + e(w))), where e(w) is the "adversarial" perturbation that maximizes the loss
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        # from IPython import embed; embed(header="gradnorm - 71 in optimizer.py ")
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def sam_training_routine(self, model, categorical_x, continuous_x, target, loss_fn):
        pred = model((categorical_x, continuous_x))
        loss = loss_fn(pred, target)

        loss.backward()
        self.first_step(zero_grad=True)

        # second forward step with disabled bachnorm running stats in second forward step
        self.disable_running_stats(model)
        pred_2 = model(categorical_x, continuous_x)
        loss_fn(pred_2, target).backward()
        self.second_step(zero_grad=True)

    def disable_running_stats(self, model):
        # set save previous moment as backup_momentum set momentum to 0
        # this is necessary since SAM doing a 2 step forward and batchnorm statistics would be wrong
        def _disable(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(self, model):
        # renable previous moment
        def _enable(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)
