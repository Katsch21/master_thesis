from __future__ import annotations

# standard imports
from typing import Iterable, Literal

# package imports
import numpy as np
import torch

# personal imports
from bbTT.data_handling import io
from bbTT.data_handling.preprocessing import k_fold
from bbTT.evaluation.load_models import rebuild_checkpoint_information
from bbTT.monitoring.logger.logger import get_logger
from bbTT.utils.parser import ParserBuilder

logger_inst = get_logger(__name__)


def last_fn_picker(last_fn):
    def sigmoid(x):
        return torch.sigmoid(x)

    def softmax(x):
        return torch.softmax(x, dim=-1)

    def identity(x):
        return x

    choices = {"sigmoid": sigmoid, "softmax": softmax, "none": identity}
    return choices[last_fn.lower()]


def evaluate_model_on_fold(
    events: dict[torch.Tensor],
    model_inst: torch.nn.Module,
    full_config: Literal["DataClass"],
    folds: Iterable[int],
    evaluate_on: str,
    last_activation_fn: Literal["sigmoid", "softmax", "identity"] = "identity",
    batch_size=1,
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
        model_inst.eval()

        # --- load data and split by indices ---
        columns_to_split = (
            "event_id",
            "normalization_weights",
            "product_of_weights",
            "evaluation_mask",
        )

        num_targets = len(full_config.dataset_config.target_map.values())
        last_fn = last_fn_picker(last_activation_fn)

        dnn_scores = {}
        for fold in folds:
            fold_split_coordinator = k_fold.FoldAndSplitCoordinator(
                events=events,
                c_fold=fold,
                k_fold=full_config.training_config.k_fold,
                seed=full_config.training_config.seed,
                training_percentage=full_config.training_config.train_ratio,
                randomize=False,
            )
            # --- run model on specific splits ---
            dnn_scores[fold] = {}
            logger_inst.info(f"Evaluate {evaluate_on} data of fold {fold}")

            for uid in events.keys():
                idx = fold_split_coordinator.indices[uid][evaluate_on]

                # when no events left after split move on
                total_n = len(idx)
                if total_n == 0:
                    continue

                out_shape = (total_n, num_targets)
                uid_scores = torch.empty(out_shape, dtype=torch.float32)

                for start_idx in range(0, total_n, batch_size):
                    end_idx = min(start_idx + batch_size, total_n)
                    batch_idx = idx[start_idx:end_idx]

                    continuous_inputs = events[uid]["continuous"][batch_idx]
                    categorical_inputs = events[uid]["categorical"][batch_idx]

                    scores = model_inst(categorical_inputs=categorical_inputs, continuous_inputs=continuous_inputs)

                    # in multi-head networks the first output is always the normal prediction
                    if model_inst.is_binned:
                        scores = scores[0]

                    scores = last_fn(scores)
                    uid_scores[start_idx:end_idx] = scores.detach().cpu()

                dnn_scores[fold][uid] = {
                    "scores": uid_scores,
                    "fold_index": idx,
                }

                # add splitted content on top
                for column in columns_to_split:
                    dnn_scores[fold][uid][column] = events[uid][column][idx]

            del uid_scores
        return dnn_scores


if __name__ == "__main__":
    parser = ParserBuilder(
        "load_checkpoint",
        "activation_fn",
        "save_path",
        "evaluate_choices",
        "batching",
        "num_threading",
        description="Evaluate Model in Checkpoint on test, training or validation set of the given input data",
    )
    args = parser.args
    model_name = args.path.stem

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger_inst.info(f"DEVICE: {DEVICE}")

    model_inst, full_config = rebuild_checkpoint_information(args.path)
    # is part of the name so everybody nows which data was used to train the network
    used_data_hash = full_config.dataset_config.content_hash()
    torch.manual_seed(full_config.training_config.seed)
    np.random.seed(full_config.training_config.seed)

    events = io.get_data(full_config.dataset_config, ignore_cache=False, save_cache=False)
    # evaluate data on each part separate and then flush scores
    for evaluate_on in args.evaluate_on:
        stem = args.file_path.stem
        suffix = args.file_path.suffix

        path = args.file_path.with_stem(f"{used_data_hash}_{model_name}_{evaluate_on}_{stem}").with_suffix(suffix)

        evaluated_data = evaluate_model_on_fold(
            model_inst=model_inst,
            full_config=full_config,
            folds=args.fold,
            evaluate_on=evaluate_on,
            last_activation_fn=args.add_activation,
            events=events,
            batch_size=args.batch_size,
        )

        torch.save(evaluated_data, path)
