import torch


def asimov_small_signal_and_no_background(
    s: torch.Tensor, b: torch.Tensor, eps: torch.Tensor = torch.as_tensor(0), *args, **kwargs
) -> torch.Tensor:
    """
    Asimov Significance for the case when no background uncertainty is given AND signal is small relative to background.
    Approximation coming from https://arxiv.org/abs/1806.00322 eq. 3.3.

    This approximation is unstable for s + b = 0. To increase stability *eps* introduces lower bound.

    Args:
        s (torch.Tensor): Tensor representing signal in bin.
        b (torch.Tensor): Tensor representing background in bin.
        eps (torch.Tensor, optional): _description_. Defaults to torch.as_tensor(0).

    Returns:
        torch.Tensor: Asimov Significance for no background uncertainty and low signal
    """
    # approximation for small signal yield see formular 3.3
    return s / torch.sqrt(s + b + eps)


def asimov_no_background(
    s: torch.tensor, b: torch.tensor, eps_log: torch.Tensor = torch.as_tensor(0), *args, **kwargs
) -> torch.Tensor:
    """
    Asimov Significance for the case when background uncertainty is set to 0.
    Approximation coming from asimov for no background uncertainty: https://arxiv.org/abs/1806.00322 eq. 3.2
    This approximation is unstable for two cases, and thus certain epsilons are introduced to stabilize  it.:
    It is unstable for no-background (b=0) regions, where *eps_log* helps.
    But also when s > (s+b)(ln(1+s/b)), for which *eps_sqrt* increase stability and lower bound the significance.

    Args:
        s (torch.Tensor): Tensor representing signal in bin.
        b (torch.Tensor): Tensor representing background in bin.
        log (int, optional): _description_. Defaults to 0.
        eps_sqrt (_type_, optional): _description_. Defaults to 1e-9.

    Returns:
        torch.Tensor: Asimov Significance without background uncertainty
    """
    b_log = b + eps_log
    # b_log = b
    return torch.sqrt(2 * ((s + b) * torch.log(1 + s / b_log) - s))


def asimov(s: torch.Tensor, b: torch.Tensor, unc_b: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Full asimov formula with background uncertainty as described in https://arxiv.org/abs/1806.00322 eq. 3.1.
    This function is unstable for:
    - no background uncertainty regions: unc_b = 0
    - no background regions: b = 0
    - combination of signal and background being small and low uncertainty

    Args:
        s (torch.Tensor): Tensor representing signal in bin.
        b (torch.Tensor): Tensor representing background in bin.
        unc_b (torch.Tensor): Tensor representing the uncertainty of the background as absolute value

    Returns:
        torch.Tensor: Full Asimov Significance
    """

    ln_nominator_1 = (s + b) * (b + unc_b**2)
    ln_denominator_1 = b**2 + (s + b) * unc_b**2
    part_1 = (s + b) * torch.log(ln_nominator_1 / ln_denominator_1)

    ln_nominator_2 = (unc_b**2 * s)
    ln_denominator_2 = b * (b + unc_b**2)
    part_2 = (b**2 / unc_b**2) * torch.log(1 + ln_nominator_2 / ln_denominator_2)

    _asimov = torch.sqrt(2 * (part_1 - part_2))
    return _asimov
