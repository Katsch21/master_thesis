from __future__ import annotations

# standard imports
from typing import Iterable, Literal

# package imports
import numpy as np
import torch

# personal imports
from data import load_data, preprocessing
from train.train_config import full_config
from utils import logger
from utils.load_models import rebuild_checkpoint_information
from utils.parser import ParserBuilder

logger_inst = logger.get_logger(__name__)

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(full_config.training_config.seed)
np.random.seed(full_config.training_config.seed)

logger_inst = logger.get_logger(__name__)
logger_inst.info(f"DEVICE: {DEVICE}")

def evaluate_model_on_fold(
    model_inst: torch.nn.Module,
    full_config: Literal["DataClass"],
    folds: Iterable[int],
    evaluate_on: Iterable[str],
    add_activation: str | None = None
    ) -> torch.tensor:
    """
    Function to evaluate *folds* with given *model_inst*.
    The configuration if handled by *full_config* and describes the network when it was trained.
    By default only the test set it evaluated, but *evaluate_on* can be extended to also evaluate training and validation set again.

    Args:
        model_inst (torch.nn.Module): Loaded model instance, that is used to evaluate the data.
        full_config (Literal[&quot;DataClass&quot;]): Dataclass containing all configs defined in train_config.py -> is also located in checkpoint of the network.
        folds (Iterable[int]): Iterable of folds to evaluate on. Careful: Network should match the fold it was trained on, no check is performed.
        evaluate_on (Iterable[str]): Iterable containing: "test", "training" or "validation", defines which indices are used of the input data used for training.

    Raises:
        ValueError: When wrong value in evaluate_on is used.

    Returns:
        torch.Tensor: Tensor with output scores of the network.
    """
    with torch.no_grad():
        dnn_scores = {}

        # --- load data and split by indices ---
        logger_inst.info("Loading Data")
        events = load_data.get_data(full_config.dataset_config, ignore_cache=False, _save_cache=False)

        if add_activation.lower() == "softmax":
            last_fn = torch.nn.functional.softmax
        elif add_activation.lower() == "sigmoid":
            last_fn = torch.nn.functional.sigmoid
        else:
            last_fn = torch.nn.Identity()

        for fold in folds:
            dnn_scores[fold] = {}

            fold_split_coordinator = preprocessing.FoldAndSplitCoordinator(
                events=events,
                c_fold=fold,
                k_fold=full_config.training_config.k_fold,
                seed=full_config.training_config.seed,
                training_percentage=full_config.training_config.train_ratio,
                randomize=False,
            )
        # --- run model on specific splits ---
        for _evaluate_on in evaluate_on:
            logger_inst.info(f"Evaluate {_evaluate_on} data of fold {fold}")
            dnn_scores[fold][_evaluate_on] = {}
            columns_to_split = ("continuous", "categorical", "event_id", "normalization_weights", "product_of_weights", "evaluation_mask")
            splitted_events = fold_split_coordinator(events, which=_evaluate_on, columns=columns_to_split)

            for uid, uid_events in splitted_events.items():
                continuous_inputs, categorical_inputs = uid_events["continuous"], uid_events["categorical"]
                scores = model_inst(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)
                scores = last_fn(scores, dim=1)
                data = {
                    "scores": scores,
                    "event_weight": uid_events["product_of_weights"],
                    "event_id": uid_events["event_id"]
                    }
                dnn_scores[fold][_evaluate_on][uid] = data

        return dnn_scores


def evaluate_model_on_arbitrary_data(model_inst, input_data):
    # input_data (dict[torch.Tensor]): Dictionary with different tensors, must match the keywords of the model_instance. This needs to be checked by yourself.
    scores = model_inst(**input_data)
    return scores

if __name__ == "__main__":

    parser = ParserBuilder(
        "load_checkpoint",
        "activation_fn",
        "save_path",
        "evaluate_choices",
        description="Evaluate Model in Checkpoint on test, training or validation set of the given input data"
        )
    args = parser.args
    model_inst, full_config = rebuild_checkpoint_information(args.path)
    evaluated_data = evaluate_model_on_fold(
        model_inst=model_inst,
        full_config=full_config,
        folds=args.fold,
        evaluate_on=args.evaluate_on,
        add_activation=args.add_activation,
        )

    torch.save(evaluated_data, args.file_path)
