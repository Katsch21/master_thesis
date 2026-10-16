from __future__ import annotations

import os
import pathlib

import torch

from bbTT.evaluation.load_models import rebuild_checkpoint_information
from bbTT.models.utils.append_layers import AddActFnToModel
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.parser import ParserBuilder

logger_inst = get_logger(__name__)


def export_pt2(
    model_inst: torch.nn.Module, name: str, fold: str, base_dir: str | None = None, activation_fn_name: str = None
) -> pathlib.Path:
    """
    Takes *model_inst* and export it to pt2 format, with dynamic batch dimension.
    The exported model is stored under a name defined by *name* and *fold* in the directory defined by *base_dir*.
    If *activation_fn_name* is given, add the respective activation function as last layer to the model before exporting.
    This is helpful, when the model for example is trained with BCEWithLogitsLoss and thus does not have a sigmoid at the end, but for inference one would like to have probabilities as output.

    Args:
        model_inst (torch.nn.Module): PyTorch model instance, no path.
        name (str): Name of the model, used to name the result.
        fold (str): Current fold of the model. Is part of the naming scheme
        base_dir (str | None, optional): Directory where model is stored, if None take from MODELS_DIR environment. Defaults to None.
        activation_fn_name (bool, optional): Adds activation function as last layer. Defaults to None, which means no activation is added. If given, must be one of "sigmoid" or "softmax".

    Returns:
        pathlib.Path: Path to exported pt2 model.
    """
    # by default set CPU device, to enable most compatible export.
    DEVICE = torch.device("cpu")
    if activation_fn_name is not None:
        model_inst = AddActFnToModel(model_inst, activation_fn_name)

    # prepare model_inst
    model_inst = model_inst.to(DEVICE)
    model_inst = model_inst.eval()

    # prepare model input signature, which requires the exact shape - using dummy inputs
    num_cat = len(model_inst.categorical_features)
    num_cont = len(model_inst.continuous_features)

    # HINT: batch size 1 is an edge case and overwrites dynamic batch size
    # batch size is set to 2 instead to prevent this.
    categorical_input = torch.zeros(2, num_cat).to(torch.int32).to(DEVICE)
    continuous_inputs = torch.zeros(2, num_cont).to(torch.float32).to(DEVICE)

    dim = torch.export.Dim("batch", min = 2, max = 1_000_000)
    dynamic_shapes = {
        "categorical_inputs": {0: dim, 1: categorical_input.shape[-1]},
        "continuous_inputs": {0: dim, 1: continuous_inputs.shape[-1]},
    }

    # do actual export and saving
    exp = torch.export.export(
        model_inst,
        args=(categorical_input, continuous_inputs),
        dynamic_shapes=dynamic_shapes,
    )

    if base_dir is None:
        base_dir = os.environ["MODELS_DIR"]

    dst = (pathlib.Path(base_dir) / f"{name}_fold{fold}").with_suffix(".pt2")
    # HINT: pickle protocol 4 ensures that size of network is not a limiting factor.
    # otherwise 4 GBit would be max.
    torch.export.save(exp, dst, pickle_protocol=4)
    return dst


def run_pt2(pt2_path: str, cat: torch.tensor, cont: torch.tensor) -> torch.tensor:
    """
    Run a given model stored at *pt2_path* with *cat* and *cont* tensors.
    This is a method to check if exporting the model resulted in same results.

    Args:
        pt2_path (str): Path to Pt2 file.
        cat (torch.tensor): Tensor with categorical features.
        cont (torch.tensor): Tensor with continuous features.

    Returns:
        torch.tensor: Tensor with model output.
    """
    exp = torch.export.load(pt2_path)
    scores = exp.module()(cat, cont)
    return scores


def compare_pt2_with_original(
    pt2_path: str, original_model: torch.nn.Module, cat: torch.tensor, cont: torch.tensor, atol: float = 1e-6
) -> bool:
    """
    Compare the output of a pt2 model with the output of the original model for given *cat* and *cont* tensors. Returns True if outputs are close within given *atol*, False otherwise.
    """
    pt2_output = run_pt2(pt2_path, cat, cont)
    original_output = original_model(cat, cont)
    return torch.allclose(pt2_output, original_output, atol=atol)


def resolve_models_path(path: pathlib.Path, folds) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Small helper to resolve *path* to the model, if only model name is given, look in MODELS_DIR for it, otherwise take given path
    Returns a tuple of the resolved pt2 path and the original pt path, which is needed to load the original model instance for comparison.
    """
    models_path = pathlib.Path(path)
    is_only_model_name = len(path.parts) == 1
    paths = {}
    for fold in folds:
        if is_only_model_name:
            models_root = pathlib.Path(os.environ["MODELS_DIR"])
            models_path = models_root / f"{models_path.stem}_fold{fold}"
        paths[fold] = models_path
    return paths


if __name__ == "__main__":
    parser = ParserBuilder(
        "load_checkpoint", "activation_fn", description="Export model to torch-export (.pt2) with dynamic batch dim"
    )
    args = parser.args

    paths = resolve_models_path(args.path, args.fold)

    # model is dict with keys
    # epoch ,model_inst, model_state_dict, optimizer
    # we want an model_instance OR if not existing we create new model instance and load the state dict
    # TODO add_activation_FN
    for fold, path in paths.items():
        logger_inst.info(f"Export fold {fold} with model checkpoint from {path}")
        model_inst, full_cfg = rebuild_checkpoint_information(path.with_suffix(".pt"))
        # model_inst = rebuild_model_from_checkpoint()
        export_pt2(model_inst, name=path.with_suffix(".pt2"), fold=fold)
    logger_inst.info(f"Finished exporting models for folds {args.fold} with name {args.path.stem}")
