import torch

from bbTT.loss.YieldCalculator import YieldCalculator
from bbTT.monitoring import HookableMixin
from bbTT.statistics.asimov import asimov, asimov_no_background, asimov_small_signal_and_no_background

asimov_functions = {
    "full": asimov,
    "no_unc": asimov_no_background,
    "approximation": asimov_small_signal_and_no_background,
}


class SignalEfficiency(HookableMixin, torch.nn.Module):
    def __init__(
        self,
        sampler_inst,
        device,
        is_training,
        asimov_cfg=None,
        *args,
        **kwargs,
    ) -> torch.tensor:
        """
        This function creates a loss using the Asimov Significance as introduced in https://arxiv.org/abs/1806.00322.

        The loss comes in three *modes*: "full", "no_unc", "approximation".
        The loss is calculated differently in train and validation mode, which is set by *train* flag.
        All tensors are placed on *device*.
        Information about phase space and weights is contained in the *sampler_inst*, which is defined Sampler.py.
        *uncertainty* sets the relative uncertainty of the background yield, everything above 1 is a constant uncertainty.

        Args:
            sampler_inst: Sampler instance that provides necessary information about phase space and weights.
            device: Device on which the tensors are placed.
            is_training: Bool to signal if the loss is used in training or validation mode, which changes the way the loss is calculated.
            mode: String to signal which implementation of the Asimov Significance is used, choose between "full", "no_unc", "approximation".
            uncertainty: Float to signal the level of uncertainty, if between 0 and 1 a relative uncertainty of

        Returns: (torch.tensor): Torch Tensor Scalar defining the loss.

        """
        super().__init__()
        self.yield_calculator = YieldCalculator(
            sampler_inst=sampler_inst,
            training=is_training,
        )

        self.is_training = is_training
        self.target_map = sampler_inst.target_map

        self.asimov_name = asimov_cfg.asimov_mode
        self.asimov_fn = asimov_functions[self.asimov_name]
        self.uncertainty = asimov_cfg.background_uncertainty
        self.asimov_epsilon = {
            "eps_log": asimov_cfg.epsilon_log,
            "eps_sqrt": asimov_cfg.epsilon_sqrt,
            "eps": asimov_cfg.epsilon_small_signal,
        }

        self.s_cls = self.target_map["hh"]  # definition of signal class
        self.device = device

    def _uncertainty(self, b):
        if self.uncertainty >= 1:
            return self.uncertainty
        return self.uncertainty * b + 1

    def reduce_yield(self, event):
        # sum over all predictions in bins
        return torch.sum(event, dim=-1)

    def stabilize(self, x, eps):
        return torch.clamp(x, min=eps)

    def _loss(self, x):
        return 1 / x.sum()

    def loss(self, *args, **kwargs):
        return self._loss(*args, **kwargs)

    def monitored_gradient_names(self):
        return []

    def monitored_tensor_names(self):
        return ["signal_yield", "background_yield", "binned_significance"]

    def forward(self, prediction, truth, product_of_weights, evaluation_mask):
        # prediction can be of shape [bin, event, node] or [event, node]
        # only signal node is necessary
        signal_node_prediction = prediction[..., self.s_cls]  # can be 2D or 1D
        signal_node_truth = truth[..., self.s_cls]  # is 1D

        # filter weights after signal and background only events
        s, b = self.yield_calculator(
            prediction=signal_node_prediction,
            truth=signal_node_truth,
            event_weight=product_of_weights,
            evaluation_mask=evaluation_mask,
        )

        significance = self.asimov_fn(s=s, b=b, unc_b=self._uncertainty(b), **self.asimov_epsilon)
        self.monitor_tensor(name="binned_significance", tensor=significance)
        self.monitor_tensor(name="signal_yield", tensor=s)
        self.monitor_tensor(name="background_yield", tensor=b)
        loss = self.loss(significance)
        return loss
