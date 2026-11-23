from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import pathlib
import re
import types
import typing
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from bbTT.monitoring.logger.logger import get_logger

if TYPE_CHECKING:
    from bbTT.configs.full_config import FullConfig
    from bbTT.train.setup import TrainingComponents

logger_inst = get_logger(__name__)

# origins of Optional[X] / Union[X, Y]; "X | None" (types.UnionType) exists from python 3.10 on, the project allows 3.9
UNION_ORIGINS: tuple[Any, ...] = (typing.Union, getattr(types, "UnionType", typing.Union))


# Which config attributes are protected from overrides
# Expected Style: {"WHAT IS FORBIDDEN": "WHY IT IS FORBIDDEN"}
# whole config is possible, e.g. "model_building_config"
# but also arguments in dot notation, e.g. dataset_config.eras
PROTECTED_OVERRIDES: dict[str, str] = {
    "model_building_config": "defines the architecture the checkpoint was written for",

    # TODO train/validation split is position based over the merged eras, so another era set moves events
    # between training and validation; blocked until the cache is stored per (era, fold)
    "dataset_config.eras": "changes the train/validation split of the checkpoint",
}

# Special Case: Incompatibility due to loss changes.
# overrides below this prefix change what the validation loss means, so the best loss of the checkpoint
# is no longer comparable and the checkpoint criteria start fresh
LOSS_OVERRIDE_PREFIX = "loss_config"

# tag that is included when doing a resume
RESUME_TAG_SEPARATOR = "-resume-"


@dataclasses.dataclass
class ResumeState:
    """
    Everything a resumed training needs that a fresh training does not.

    Args:
        checkpoint (dict[str, Any]): Loaded checkpoint, see CheckPoint._snapshot for its keys.
        full_config (FullConfig): Checkpoint config with the requested overrides applied.
        fold (int): Fold the checkpoint was trained on, the only fold the resume trains.
        start_iteration (int): First iteration of the resumed loop.
        tag (str): Short hash of the overrides, suffix for checkpoint and tensorboard names of the resume.
        keep_checkpoint_criteria (bool): Whether best loss and counters of the checkpoint still apply.
        checkpoint_base_name (str): Name the resumed checkpoint was written under, without the "_fold<k>" suffix.
    """

    checkpoint: dict[str, Any]
    full_config: FullConfig
    fold: int
    start_iteration: int
    tag: str
    keep_checkpoint_criteria: bool
    checkpoint_base_name: str

    def run_name(self, base_name: str) -> str:
        """
        Args:
            base_name (str): Name of the original run, e.g. the tensorboard name or the checkpoint name.

        Returns:
            str: Name of the resumed run, derived from the original so both sort next to each other.
        """
        return f"{base_name}{RESUME_TAG_SEPARATOR}{self.tag}"

def _base_name_of(path: pathlib.Path) -> str:
    """
    Args:
        path (pathlib.Path): Checkpoint file "<name>_fold<k>.pt".

    Returns:
        str: The "<name>" part, i.e. what --save_model_name was for that run.
    """
    return re.sub(r"_fold\d+$", "", path.stem)

# --- override specs -----------------------------------------------------------------------------------------
# An override is the last truth: it is applied after everything else (code config, checkpoint config).
# Two forms exist:
#   "PATH"        take the value of the current code config (only meaningful on a resume)
#   "PATH=VALUE"  set VALUE, parsed with the type hint of the field (works for fresh trainings and resumes)


@dataclasses.dataclass(frozen=True)
class Override:
    """
    One parsed --override argument.

    Args:
        path (str): Dotted path into FullConfig, e.g. "optimizer_config.lr".
        value (str | None): Raw value string of "PATH=VALUE", None for the plain "PATH" form.
    """

    path: str
    value: str | None = None


def parse_override(spec: str) -> Override:
    """
    Args:
        spec (str): Raw --override argument, "PATH" or "PATH=VALUE". VALUE may contain further "=".

    Returns:
        Override: Parsed spec, the path is validated.

    Raises:
        ValueError: On an invalid path or an empty value after "=".
    """
    path, separator, value = spec.partition("=")
    path = path.strip()
    _split_path(path)
    if separator and not value.strip():
        raise ValueError(f"override '{spec}': empty value after '='")
    return Override(path=path, value=value.strip() if separator else None)


def _field_type(parent: Any, name: str) -> Any:
    """
    Args:
        parent (Any): Dataclass instance holding the field.
        name (str): Field name.

    Returns:
        Any: Resolved type hint of the field, ``Any`` if the attribute is no annotated field.
    """
    return typing.get_type_hints(type(parent)).get(name, Any)


def _coerce(value: str, hint: Any) -> Any:
    """
    Parse a command line string into the type a config field expects.

    Strings and Literals are taken verbatim, "none" gives None for Optional fields, booleans accept
    true/false/1/0, everything else goes through ``ast.literal_eval`` (so "3e-3", "(0, 1)", "{'a': 1}" work).

    Args:
        value (str): Raw value from "PATH=VALUE".
        hint (Any): Type hint of the target field.

    Returns:
        Any: The parsed value.

    Raises:
        ValueError: If the value does not fit the hint or cannot be parsed.
    """
    origin, args = typing.get_origin(hint), typing.get_args(hint)

    # Optional[X] / X | None: strip the None and recurse on the remaining type
    if origin in UNION_ORIGINS:
        if value.lower() in ("none", "null"):
            return None
        remaining = [a for a in args if a is not type(None)]
        return _coerce(value, remaining[0] if len(remaining) == 1 else Any)

    if origin is typing.Literal:
        for choice in args:
            if value == str(choice):
                return choice
        raise ValueError(f"'{value}' is not one of {args}")

    if dataclasses.is_dataclass(hint):
        raise ValueError(f"a whole sub config ({hint.__name__}) cannot be set from the command line, set its fields")

    if hint is bool:
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"'{value}' is not a boolean")

    if hint is str:
        return value

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"cannot parse '{value}' for type {hint}: {error}") from error

    # literal_eval gives int for "1", the field wants a float
    if hint is float and isinstance(parsed, int) and not isinstance(parsed, bool):
        return float(parsed)
    if origin is tuple and isinstance(parsed, list):
        return tuple(parsed)
    return parsed


def apply_overrides(config: FullConfig, overrides: Iterable[Override], current: FullConfig | None = None) -> None:
    """
    Write the overrides into *config* in place. This is the last step of building a config, nothing after it
    changes a value again except ``__post_init__`` derived fields (see _rerun_post_init).

    Args:
        config (FullConfig): Config to change.
        overrides (Iterable[Override]): Parsed --override arguments.
        current (FullConfig | None): Source of the plain "PATH" form, the code config on a resume.
            None for a fresh training, where "PATH" without a value is a no-op and therefore an error.

    Raises:
        ValueError: For "PATH" without *current*, or a value that does not fit the field.
        AttributeError: If a path does not exist.
    """
    for override in overrides:
        segments = _split_path(override.path)
        target_parent = _resolve_parent(config, segments)

        if override.value is not None:
            hint = _field_type(target_parent, segments[-1])
            try:
                value = _coerce(override.value, hint)
            except ValueError as error:
                raise ValueError(f"override '{override.path}': {error}") from error
        elif current is None:
            raise ValueError(
                f"override '{override.path}' names no value: without --resume the code config is already in use, "
                "use 'PATH=VALUE'"
            )
        else:
            value = copy.deepcopy(getattr(_resolve_parent(current, segments), segments[-1]))

        setattr(target_parent, segments[-1], value)
        logger_inst.i_info(f"Override: {override.path} = {value!r}")

    # new values must pass the same checks as code defaults (choice_check etc.) and refresh derived fields
    _rerun_post_init(config)


# --- config merge -------------------------------------------------------------------------------------------
# first the config from checkpoint is loaded, then the overrides are applied on top.

def _split_path(path: str) -> list[str]:
    """
    Args:
        path (str): Dotted path into FullConfig, e.g. "record_config.validation_interval".

    Returns:
        list[str]: Path segments.

    Raises:
        ValueError: On an empty path or empty segment.
    """
    segments = path.split(".")
    if not path or any(not s for s in segments):
        raise ValueError(f"invalid override path '{path}'")
    return segments


def _resolve_parent(root: Any, segments: Sequence[str]) -> Any:
    """
    Walk all but the last segment, so the caller can get or set the last attribute on the returned object.

    Args:
        root (Any): Object to start from.
        segments (Sequence[str]): Path segments, at least one.

    Returns:
        Any: Object holding the attribute named by the last segment.

    Raises:
        AttributeError: If a segment does not exist, the message names the full path.
    """
    obj = root
    for segment in segments[:-1]:
        if not hasattr(obj, segment):
            raise AttributeError(f"override path '{'.'.join(segments)}': '{segment}' not found")
        obj = getattr(obj, segment)
    if not hasattr(obj, segments[-1]):
        raise AttributeError(f"override path '{'.'.join(segments)}': '{segments[-1]}' not found")
    return obj


def _rerun_post_init(config: Any) -> None:
    """
    Run ``__post_init__`` again on *config* and every dataclass below it.

    A config rebuilt from a checkpoint dict skipped ``__post_init__``, so fields derived there (e.g.
    DataConfig.datasets from dataset_pattern and eras) still hold the checkpoint's values after an override.
    Re-running makes the merged config equal to a freshly constructed one with the same fields.
    This should only matter, when an overwrite contains 'wrong' values.

    Args:
        config (Any): Dataclass instance.
    """
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            _rerun_post_init(value)
    post_init = getattr(config, "__post_init__", None)
    if post_init is not None:
        post_init()


def _protected_by(path: str) -> str | None:
    """
    Handling PROTECTED_OVERRIDES modules.

    Args:
        path (str): Dotted override path.

    Returns:
        str | None: The protected path that covers *path* (itself or a parent of it), None if unprotected.
    """
    segments = _split_path(path)
    for protected in PROTECTED_OVERRIDES:
        protected_segments = _split_path(protected)
        if segments[: len(protected_segments)] == protected_segments:
            return protected
    return None


def merge_config(base: FullConfig, current: FullConfig, overrides: Sequence[Override]) -> FullConfig:
    """
    Take *base* and apply the *overrides*: the plain "PATH" form copies the value from *current*, "PATH=VALUE"
    sets the parsed value (see apply_overrides).

    Both a whole sub config ("record_config") and a single field ("record_config.validation_interval")
    are valid paths. Derived config values are properties, so they follow the overrides by themselves; only the
    compatibility rules of FullConfig are checked again.

    Args:
        base (FullConfig): Config stored in the checkpoint, the source of truth for the model.
        current (FullConfig): Config of the running code, the source of the plain "PATH" overrides.
        overrides (Sequence[Override]): Parsed --override arguments.

    Returns:
        FullConfig: Deep copy of *base* with the overrides applied.

    Raises:
        ValueError: If a path is protected or an override would change a protected value.
    """
    for override in overrides:
        protected = _protected_by(override.path)
        if protected is not None:
            raise ValueError(f"override '{override.path}' is not allowed: {protected} {PROTECTED_OVERRIDES[protected]}")

    merged = copy.deepcopy(base)
    apply_overrides(merged, overrides, current=current)

    # an override of a parent can still change a protected value underneath it
    for protected, reason in PROTECTED_OVERRIDES.items():
        segments = _split_path(protected)
        if getattr(_resolve_parent(merged, segments), segments[-1]) != getattr(_resolve_parent(base, segments), segments[-1]):
            raise ValueError(f"overrides change '{protected}', which is not allowed: {reason}")
    return merged


def override_tag(current: FullConfig, overrides: Iterable[Override]) -> str:
    """
    Short hash over the override paths and their values, so different continue settings get different names.

    Args:
        current (FullConfig): Config the plain "PATH" overrides come from.
        overrides (Iterable[Override]): Parsed --override arguments.

    Returns:
        str: 8 hex characters, "base" when there are no overrides.
    """
    items = []
    for override in sorted(overrides, key=lambda o: o.path):
        if override.value is not None:
            items.append((override.path, override.value))
            continue
        segments = _split_path(override.path)
        value = getattr(_resolve_parent(current, segments), segments[-1])
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        items.append((override.path, repr(value)))
    if not items:
        return "base"
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()[:8]


def loss_changed(overrides: Iterable[Override]) -> bool:
    """
    Args:
        overrides (Iterable[Override]): Parsed --override arguments.

    Returns:
        bool: True if any override touches the loss config.
    """
    return any(_split_path(o.path)[0] == LOSS_OVERRIDE_PREFIX for o in overrides)

def override_paths(config: Any, prefix: str = "") -> list[str]:
    """
    All dotted paths that merge_config accepts for *config*: every dataclass field, and every sub config as a whole.

    Args:
        config (Any): Dataclass instance to walk, normally a FullConfig.
        prefix (str): Path prefix for recursion, empty at the top level.

    Returns:
        list[str]: Paths in field order; a sub config is listed before its fields. Protected sub configs are
            listed too, so the caller can mark them.
    """
    paths: list[str] = []
    for field in dataclasses.fields(config):
        path = f"{prefix}{field.name}"
        paths.append(path)
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            paths.extend(override_paths(value, prefix=f"{path}."))
    return paths

def format_overrides(config: Any) -> str:
    """
    Human readable listing of all override paths with their value in *config*, for --list-overrides.

    Args:
        config (Any): The current code config, a FullConfig.

    Returns:
        str: One line per path, sub configs as headers, protected ones marked.
    """
    lines = ["Override paths (values from the current code config):"]
    for path in override_paths(config):
        segments = path.split(".")
        value = getattr(_resolve_parent(config, segments), segments[-1])
        note = "  [protected, not overridable]" if _protected_by(path) is not None else ""

        if dataclasses.is_dataclass(value):
            lines.append(f"\n{path}{note}")
        else:
            lines.append(f"  {path} = {value!r}{note}")
    return "\n".join(lines)



# --- loading ------------------------------------------------------------------------------------------------


def _fold_of(checkpoint: dict[str, Any], path: pathlib.Path) -> int:
    """
    Args:
        checkpoint (dict[str, Any]): Loaded checkpoint.
        path (pathlib.Path): File it was loaded from.

    Returns:
        int: Fold from the checkpoint, or parsed from the file name "<name>_fold<k>.pt" for older checkpoints.

    Raises:
        ValueError: If neither source yields a fold.
    """
    if "fold" in checkpoint:
        return int(checkpoint["fold"])
    match = re.search(r"_fold(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot determine fold of checkpoint {path}: no 'fold' key and no '_fold<k>' in the name")
    return int(match.group(1))


def load_resume_state(
    checkpoint_name: str | pathlib.Path,
    current_config: FullConfig,
    overrides: Sequence[Override] = (),
) -> ResumeState:
    """
    Load a checkpoint and prepare the config of the resumed training.

    The checkpoint config is the base; the overrides are applied on top, see merge_config.
    The name the resume writes its checkpoints under is not part of the config: train.main derives "<name>-resume-<tag>" from
    `checkpoint_base_name` unless --save_model_name is given, so the loaded checkpoint is never overwritten.

    Args:
        checkpoint_name (str | pathlib.Path): Checkpoint name or path, see loading.resolve_checkpoint_path.
        current_config (FullConfig): Config of the running code.
        overrides (Sequence[Override]): Parsed --override arguments, see merge_config.

    Returns:
        ResumeState: Checkpoint, merged config, fold, start iteration and naming tag.
    """
    from bbTT.checkpoint.loading import load_checkpoint, resolve_checkpoint_path
    from bbTT.configs.full_config import FullConfig
    from bbTT.evaluation.rebuild_dataclasses import from_dict

    path = resolve_checkpoint_path(checkpoint_name, suffix=".pt")
    checkpoint = load_checkpoint(path)
    base_config = from_dict(FullConfig, checkpoint["full_config"])

    merged = merge_config(base=base_config, current=current_config, overrides=overrides)
    tag = override_tag(current_config, overrides)

    keep_criteria = not loss_changed(overrides)
    if not keep_criteria:
        logger_inst.i_info("Resume with changed loss: checkpoint criteria (best loss, patience) start fresh")

    fold = _fold_of(checkpoint, path)
    logger_inst.i_info(f"Resume from {path.name}: fold {fold}, iteration {checkpoint['iteration']}")

    return ResumeState(
        checkpoint=checkpoint,
        full_config=merged,
        fold=fold,
        start_iteration=int(checkpoint["iteration"]) + 1,
        tag=tag,
        keep_checkpoint_criteria=keep_criteria,
        checkpoint_base_name=_base_name_of(path)
    )


def restore_training_state(components: TrainingComponents, resume_state: ResumeState) -> int:
    """
    Load the checkpoint into freshly built components.

    Model, optimizer and scheduler get their state dicts. The CheckPoint gets the loaded checkpoint as
    `last_checkpoint`, so the scheduler handler can roll back to it after a learning rate drop, and its
    criteria (best loss, patience counter) when they still apply.

    Args:
        components (TrainingComponents): Components built from the merged config, see setup.build_training_components.
        resume_state (ResumeState): What load_resume_state returned.

    Returns:
        int: Iteration the loop starts with.
    """
    checkpoint = resume_state.checkpoint
    components.model_inst.load_state_dict(checkpoint["model_state_dict"])
    components.optimizer_inst.load_state_dict(checkpoint["optimizer_state_dict"])
    if components.scheduler_inst is not None and checkpoint.get("scheduler_state_dict") is not None:
        components.scheduler_inst.load_state_dict(checkpoint["scheduler_state_dict"])

    checkpoint_inst = components.checkpoint_inst
    checkpoint_inst.last_checkpoint = checkpoint
    if resume_state.keep_checkpoint_criteria and "checkpoint_state_dict" in checkpoint:
        checkpoint_inst.load_state_dict(checkpoint["checkpoint_state_dict"])

    return resume_state.start_iteration
