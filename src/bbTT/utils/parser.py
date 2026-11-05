import argparse
import os
import pathlib
from typing import Callable


class ParserBuilder:
    """Composes an argparse parser from named argument groups.

    Each group is a method ``add_<name>``; pass the names to the constructor to include them.
    Parsing happens immediately, the result is available as ``self.args``.

    Example:
        parser = ParserBuilder("tensorboard", "cache", description="...", epilog="example: ...")
        parser.args.tensorboard_name
    """


    def __init__(self, *groups: str, description: str = "", epilog: str = "") -> None:
        """
        Args:
            *groups (str): names of argument groups to add, e.g. "tensorboard", "cache".
            description (str, optional): shown at the top of --help. Defaults to "".
            epilog (str, optional): shown at the bottom of --help, line breaks are kept; use it
                for a minimal usage example. Defaults to "".
        """
        self.parser = argparse.ArgumentParser(
            description=description,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.build(groups)
        self.args = self.parser.parse_args()

    def build(self, groups: tuple[str, ...]) -> None:
        """Adds all requested argument groups to the parser.

        Args:
            groups (tuple[str, ...]): group names without the "add_" prefix.

        Raises:
            ValueError: if a group name has no matching add_ method.
        """
        for group in groups:
            method = getattr(self, f"add_{group}", None)
            if method is None:
                raise ValueError(f"unknown argument group '{group}', expected one of {self.available_groups()}")
            method()

    @classmethod
    def available_groups(cls) -> list[str]:
        """Lists all argument group names this builder knows.

        Returns:
            list[str]: group names without the "add_" prefix.
        """
        return [name[len("add_") :] for name in dir(cls) if name.startswith("add_")]

    def add_tensorboard(self) -> None:
        self.parser.add_argument(
            "--tensorboard_name",
            "-tn",
            default=None,
            help="Name of the tensorboard run. If not given a name is generated. (default: generated)",
        )

    def add_cache(self):
        self.parser.add_argument(
            "--rebuild_cache",
            "-rc",
            action="store_true",
            default=False,
            help="Rebuild the preprocessed data instead of reading it from the cache. (default: use cache)",

        )

        self.parser.add_argument(
            "--no_save_cache",
            "-ns",
            dest="save_cache",
            action="store_false",
            help="Do not write the preprocessed data to the cache. (default: write cache)",
        )

    def add_load_checkpoint(self):

        self.parser.add_argument(
            "--checkpoint_src",
            "-c_src",
            dest="path",
            type=pathlib.Path,
            required=True,
            action="store",
            help="Path to model checkpoint (.pt) to load",
        )

        self.parser.add_argument(
            "--fold",
            "-f",
            dest="fold",
            default=0,
            type=self.comma_separated(int),
            help="Comma separated list of folds, e.g. '0' or '0,1,2'. Each fold on its respective test data.",
        )

    def add_activation_fn(self):
        self.parser.add_argument(
            "--add_activation",
            required=False,
            default="identity",
            choices=["sigmoid", "softmax", "identity"],
            help=(
                "Activation function appended on the network output. Must be 'identity' if "
                "model already use their last activation (use_last_activation=True). (default: identity)"
                ),
            )

    def add_model_activation(self):
        self.parser.add_argument(
            "--model_activation",
            required=False,
            default="from_checkpoint",
            choices=["on", "off", "from_checkpoint"],
            help=(
                "Models can define their own activation. This choice is also saved in their saved configuration. "
                "This function determines whether the activation defined inside the model (use_last_activation) is applied. "
                "'from_checkpoint' keeps the value stored in the checkpoint config. (default: from_checkpoint)"
            ),
        )


    def add_save_path(self):

        self.parser.add_argument(
            "--save_dst",
            "-s_dst",
            type=self.output_path_in_evaluation_dir,
            required=True,
            help=(
            """
            Path to destination of saved scores.
            A relative path is placed in EVALUATION_DIR. An absolute path is used as is.
            The suffix is replace by .pt.
            """
            ),
        )

    def add_evaluate_choices(self):
        self.parser.add_argument(
            "--evaluate_on",
            "-e",
            dest="evaluate_on",
            default=["test"],
            type=self.comma_separated(str, allowed=("test", "training", "validation")),
            help="Comma separated list of datasets to evaluate on: test, training, validation. (default: test)",
        )

    def add_batching(self):

        self.parser.add_argument(
            "--batch_size",
            "-bs",
            dest="batch_size",
            type=self.positive_int,
            required=True,
            help="Batch Size used for event loop. When memory is a problem reduce batch size first.",
        )

    def add_num_threading(self):
        self.parser.add_argument(
            "--num_threads",
            "-nt",
            dest="num_threads",
            default=0,
            type=self.non_negative_int,
            required=False,
            help="Number of torch threads. 0 keeps the torch default (all available). (default: 0)",
        )

    # validators ---------------------------------------------------------------
    # type= callables for argparse: convert the raw string, raise ArgumentTypeError on bad input
    # (argparse turns that into a clean "usage: ... error: argument X: <message>" and exits with code 2)

    @staticmethod
    def positive_int(value: str) -> int:
        """Parses an integer >= 1.

        Args:
            value (str): raw command line value.

        Returns:
            int: parsed value.

        Raises:
            argparse.ArgumentTypeError: if value is not an integer or smaller than 1.
        """
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got '{value}'")
        if number < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
        return number

    @staticmethod
    def non_negative_int(value: str) -> int:
        """Parses an integer >= 0.

        Args:
            value (str): raw command line value.

        Returns:
            int: parsed value.

        Raises:
            argparse.ArgumentTypeError: if value is not an integer or negative.
        """
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected an integer, got '{value}'")
        if number < 0:
            raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
        return number

    @staticmethod
    def comma_separated(item_type: Callable[[str], object], allowed: tuple = ()) -> Callable[[str], list]:
        """Builds a parser for comma separated lists, e.g. "0,1,2" or "test,validation".

        Args:
            item_type (Callable[[str], object]): converter applied to each item, e.g. int or str.
            allowed (tuple, optional): if given, every converted item must be contained in it. Defaults to ().

        Returns:
            Callable[[str], list]: function usable as argparse type=.
        """

        def parse(value: str) -> list:
            items = []
            for raw in value.split(","):
                raw = raw.strip()
                try:
                    item = item_type(raw)
                except ValueError:
                    raise argparse.ArgumentTypeError(f"invalid item '{raw}' in '{value}'")
                if allowed and item not in allowed:
                    raise argparse.ArgumentTypeError(f"'{item}' is not one of {allowed}")
                items.append(item)
            return items
        return parse

    @staticmethod
    def output_path_in_evaluation_dir(value: str) -> pathlib.Path:
        """Resolves the destination of saved scores.

        A relative path is placed inside EVALUATION_DIR, an absolute path is taken as is
        (its parent directory must exist). The suffix is always replaced by ".pt".

        Args:
            value (str): raw command line value.

        Returns:
            pathlib.Path: resolved path with ".pt" suffix.

        Raises:
            argparse.ArgumentTypeError: if the parent directory of an absolute path does not exist.
        """
        path = pathlib.Path(value)
        if path.is_absolute():
            if not path.parent.exists():
                raise argparse.ArgumentTypeError(f"parent directory does not exist: {path.parent}")
        else:
            path = pathlib.Path(os.environ["EVALUATION_DIR"]) / path
        return path.with_suffix(".pt")
