
from .binning import BinningLayer
from .kernels.base_kernel import BaseKernel, OverflowKernel, UnderflowKernel

# backwards compatibility
from .kernels.gaussian_kernel import GaussianKernel, GaussianOverflowKernel, GaussianUnderflowKernel
from .kernels.linear_kernel import LinearKernel, LinearOverflowKernel, LinearUnderflowKernel
from .kernels.tanh_kernel import TanhKernel, TanhOverflowKernel, TanhUnderflowKernel

KERNEL_MAP = {
    "Tanh":
        {
        "underflow" : TanhUnderflowKernel,
        "overflow" : TanhOverflowKernel,
        "normal" : TanhKernel,
    },
    "Gaussian":
        {
        "underflow" : GaussianUnderflowKernel,
        "overflow" : GaussianOverflowKernel,
        "normal" : GaussianKernel,
    },
    "Linear":
        {
        "underflow" : LinearUnderflowKernel,
        "overflow" : LinearOverflowKernel,
        "normal" : LinearKernel,
    },
}

__all__ = [
    "KERNEL_MAP",
    "BinningLayer",

    "BaseKernel",
    "UnderflowKernel",
    "OverflowKernel",

    "TanhKernel",
    "TanhUnderflowKernel",
    "TanhOverflowKernel",

    "GaussianKernel",
    "GaussianUnderflowKernel",
    "GaussianOverflowKernel",

    "LinearKernel",
    "LinearUnderflowKernel",
    "LinearOverflowKernel",

]
