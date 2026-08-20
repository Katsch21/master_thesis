import argparse
import os
import pathlib


class ParserBuilder():
    def __init__(self, *args, description=""):
        self.parser = argparse.ArgumentParser(description=description)
        self.build(args)
        self.args = self.parser.parse_args()

    def add_tensorboard(self):
        self.parser.add_argument(
            "--tensorboard_name",
            "-tn",
            dest="tensorboard_name",
            action="store",
            default=None,
            help="Name of the tensorboard, if given, turns off generated name (default: None)"
            )

    def add_cache(self):
        self.parser.add_argument(
        "--ignore_cache",
        "-ic",
        action="store_true",
        default=False,
        help="Ignore cache when running the program (default: False)"
        )

        self.parser.add_argument(
        "--save-cache",
        "-s",
        dest="save_cache",
        action="store_true",
        default=True,
        help="Save cache (default: False)"
        )

    def add_load_checkpoint(self):

        self.parser.add_argument(
            "--src",
            "-s",
            dest="path",
            type=pathlib.Path,
            required=True,
            action="store",
            help="Path to model checkpoint to load, typically with .pt suffix"
        )

        def parse_folds(value):
            return [int(x) for x in value.split(",")]

        self.parser.add_argument(
            "--fold",
            "-f",
            dest="fold",
            required=True,
            default="0",
            type=parse_folds,
            help="Comma separated list of folds respective test data."
        )

    def add_activation_fn(self):
        self.parser.add_argument(
            "--add_activation",
            required=False,
            help="If value is given, get activation function and add at end of network",
            default="identity",
            choices=["sigmoid", "softmax", "identity"],
        )

    def add_save_path(self):

        def valid_output_path(value: str) -> pathlib.Path:
            path = pathlib.Path(value)

            if path.is_absolute():
                if not path.parent.exists():
                    raise argparse.ArgumentTypeError(f"Parent directory does not exist: {path.parent}")
            else:
                eval_dir = pathlib.Path(os.environ["EVALUATION_DIR"])
                path = eval_dir / path
            return path.with_suffix(".pt")

        self.parser.add_argument(
            "--file_path",
            "-fp",
            type=valid_output_path,
            required=True,
            help=(
            """
            Path to destination of saved output scores.
            If absolute path is given a parental check is performed, else data is saved in EVALUATION_DIR.
            """
            )
        )

    def add_evaluate_choices(self):
        def choices(value):
            value = value.split(",")
            value = [v.strip() for v in value]
            value_possible_choices = ("test", "training", "validation")

            _check = [v in value_possible_choices for v in value]

            if not all(_check):
                raise ValueError(f"Evaluate is {value}, but can only be one of these: {value_possible_choices}")
            return value

        self.parser.add_argument(
            "--evaluate_on",
            "-e",
            dest="evaluate_on",
            default="test",
            type=choices,
            required=True,
            help=(
            """
            Comma separated list of choices one can have to evaluate the data.
            """
            )
        )


    def add_batching(self):
        def valid_batch_size(value):
            value = int(value)
            assert value >= 1, "batch size can't be negative or zero"
            return value

        self.parser.add_argument(
            "--batch_size",
            "-bs",
            dest="batch_size",
            type=valid_batch_size,
            required=True,
            help=(
            """
            Batch Size used for event loop.
            """
            )
        )

    def add_num_threading(self):
        def valid_threads(value):
            import torch
            value = int(value)
            assert value >= 0, "num of threads can't be negative"

            if value > 0:
                torch.set_num_threads(num_threads)
            return None

        self.parser.add_argument(
            "--num_threads",
            "-nt",
            dest="num_threads",
            default="0",
            type=valid_threads,
            required=False,
            help=(
            """
            Number of threads for interops calculations. If threads is set 0 all existing threads are used.
            """
            )
        )

    def build(self, args):
        commands = [f"add_{arg}" for arg in args]
        for command in commands:
            getattr(self, command)()
