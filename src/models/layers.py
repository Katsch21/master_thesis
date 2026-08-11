from __future__ import annotations

import copy
from typing import Sequence

import torch

from utils import logger, utils
from torch.nn.utils import parametrize

logger_inst = logger.get_logger(__name__)

def dummy_identity(layer: torch.nn.Module | None) -> torch.nn.Module:
    if layer is None:
        return torch.nn.Identity()
    return layer

class EmptyLayer(torch.nn.Module):
    """
    Empty Layer that return always an empty tensor.
    This kind of layer is useful, when one wants an optional tensor in concatenation operations.
    """
    def __init__(self):
        super().__init__()
        self.ndim = 0

    def forward(self, *args, **kwargs):
        return torch.tensor([])

def dummy_empty(condition, layer: torch.nn.Module | None) -> torch.nn.Module:
    if condition:
        return torch.nn.EmptyLayer()
    return layer

class WeightNormalizedLinear(torch.nn.Linear):  # noqa: F811
    def __init__(self ,*args, normalize=False, **kwargs):
        """
        If normalize is set to True, Linear layer is replaced by weight normalized layer as described in https://arxiv.org/abs/1602.07868.
        If false, the layer is a normal linear layer.

        WeightNormalizedLayer decouple the length of the weight vector from its direction.
        This is done by convert weights from Lienar Layer to weight_original0 and 1.
        0 stands for the magnitude parameter g, while 1 is for the direction v.

        Args:
            normalize (bool, optional): True to replace Linear Layer with WeightNormalizedLayer. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        if normalize:
            self = torch.nn.utils.parametrizations.weight_norm(self, name='weight', dim=0)

class PaddingLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        target_value: float | int = 0,
        padding_value: float | int = utils.EMPTY_FLOAT,
        target_dtype: torch.dtype = torch.float32,
    ):
        """
        Pads input tensor on indices which equals *mask_value* with given *padding_value*.

        Args:
            target_value (int, optional): Target value to mask. Defaults to 0.
            padding_value (float, optional): Value which is used to pad the target. Defaults to utils.EMPTY_FLOAT.
        """
        super().__init__()

        self.target_value = torch.nn.Buffer(
            torch.tensor(target_value).to(torch.float32),
            persistent=True
            )
        self.padding_value = torch.nn.Buffer(
            torch.tensor(padding_value).to(torch.float32),
            persistent=True
            )
        self.target_dtype = target_dtype

    def forward(self, x):
        x = x.to(self.target_dtype)
        mask = x == self.target_value
        x[mask] = self.padding_value
        return x

class CategoricalTokenizer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        categories: tuple[str ],
        expected_categorical_inputs: dict[str, list[int]],
        empty: int = None,
    ):
        """
        Initializes tokenizer for given *expected_categorical_inputs*.
        The tokenizer creates a mapping array in the order of given columns defined in *categories*.
        Empty values are represented as *empty*.
        All categories will be mapped into a common categorical space and ready to be used by a embedding layer.

        Args:
            categories (tuple[str]): Names of the categories as strings.
            Sorting of the entries must correspond to the order of columns in input tensor!
            expected_categorical_inputs (dict[list[int]], optional): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
            empty (int, optional): Value used to represent missing values in the input tensor.
                The empty value must be positive and not already used in the categories.
                If not given, no handling of missing values will be done.
                Defaults to None.
        """
        super().__init__()
        self._expected_inputs, self._empty = self.setup(categories, expected_categorical_inputs, empty)

        # setup lookuptable, returns dummy if None
        _map, _min = self.LookUpTable(self.pad_to_longest())
        _indices = None if _min is None else torch.arange(len(_min))

        # register buffer
        self.map = torch.nn.Buffer(_map, persistent=True)
        self.min = torch.nn.Buffer(_min, persistent=True)
        self.indices = torch.nn.Buffer(_indices, persistent=True)

    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
        # overload load_state_dict to set buffer sizes to same of state dict
        for name in self.state_dict().keys():
            self.__setattr__(name, torch.zeros_like(state_dict[name]))
        super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def setup(
        self,
        categories: list[str],
        expected_inputs: list[str],
        empty: int,
    ) -> tuple[dict[str, list[int]], int | None]:
        # do all the preparation steps like value checking and adding of empty categories
        # also remove double categories and only take the categories used by the network
        def _empty(expected_inputs, empty):
            if empty is None:
                return None
            if empty < 0:
                raise ValueError("Empty value must be positive")
            if empty in set([item for sublist in expected_inputs.values() for item in sublist]):
                raise ValueError(f"Empty value {empty} is already used in on the categories")
            return empty

        # check if cateogries are part of expected_inputs at least one
        if not set(categories) & set(expected_inputs.keys()):
            sep = "\n"
            raise ValueError(
                f"Categories must not be part of Expected categories:\n"
                f"categories:\n{sep.join(categories)}\nexpected categories:\n{sep.join(expected_inputs.keys())}"
            )

        if expected_inputs is None:
            return {}, None

        # check empty for faulty values
        # add empty category with value of empty to each value
        expected_inputs = copy.deepcopy(expected_inputs)
        empty = _empty(expected_inputs, empty)
        _expected_inputs = {}
        for categorie in map(str, categories):
            data = expected_inputs[categorie]
            if empty is not None:
                # when empty value is given append it to category
                data.append(empty)
            _expected_inputs[categorie] = data
        return _expected_inputs, empty

    @property
    def num_dim(self) -> torch.IntTensor:
        return torch.max(self.map) + 1

    def __repr__(self):
        # create dummy input from expected_categorical_inputs
        padded_array = self.pad_to_longest()
        if padded_array is None:
            return "Not initialized Tokenizer"

        expected_pad = padded_array.transpose(0, 1).to(device=self.map.device)
        shifted = expected_pad - self.min
        output_per_feature = self.map[self.indices, shifted].transpose(0, 1)
        _str = []
        _str.append("Translation (input : output):")
        for ind, (categorie, expected_value) in enumerate(self._expected_inputs.items()):
            _str.append(f"{categorie}: {expected_value} -> {output_per_feature[ind][:len(expected_value)].tolist()}")
        return "\n".join(_str)

    def check_for_values_outside_range(self, input_tensor: torch.FloatTensor):
        """
        Helper function checks *input_tensor* for values the tokenizer does not expect but found.

        Args:
            input_tensor (torch.tensor): Input tensor of categorical features.
        """
        # reshape to have features in the first dimension
        input_tensor = input_tensor.transpose(0, 1)
        for i, (category, expected_value) in enumerate(self._expected_inputs.items()):
            uniques = set(torch.unique(input_tensor[i]).to(torch.int32).tolist())
            expected = set(expected_value)
            if uniques != expected:
                difference = uniques - expected
                logger_inst.critical(
                    f"{category} has values outside the expected range: {difference}.\n"
                    "The tokenizer will return wrong values for these inputs."
                    )

    def pad_to_longest(self) -> torch.FloatTensor:
        if not self._expected_inputs:
            return None
        # helper function to pad the input tensor to the longest category
        # first value of the category is used as padding value
        local_max = max([
            len(input_for_category)
            for input_for_category in self._expected_inputs.values()
        ])
        # pad with first value of the category, so we guarantee to not introduce new values
        array = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(input_for_category),
                    (0, local_max - len(input_for_category)),
                    mode="constant",
                    value=input_for_category[0],
                )
                for input_for_category in self._expected_inputs.values()
            ],
        )
        return array

    def LookUpTable(
        self,
        array: torch.FloatTensor,
        padding_value: int = utils.EMPTY_INT,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor] | None:
        """
        Maps multiple categories given in *array* into a sparse vectoriced lookuptable.
        The given *padding_value* represents no matching categories.

        Args:
            array (torch.tensor): 2D array of categories.

        Returns:
            tuple([torch.tensor]), None: Returns minimum and LookUpTable
        """
        if array is None:
            return None, None
        # append empty to the array representing the empty category if empty is set
        # if self._empty is not None:
        #     array = torch.cat([array, torch.ones(array.shape[0], dtype=torch.int32).reshape(-1, 1) * self.empty], axis=-1)

        # shift input by minimum, pushing the categories to the valid indice space
        minimum = array.min(axis=-1).values
        # shift the input array by their respective minimum
        indice_array = array - minimum.reshape(-1, 1)
        # biggest shifted value + 1
        upper_bound = torch.max(indice_array) + 1

        # warn for big categories
        if upper_bound > 100:
            logger_inst.warning("Be aware that a large number of categories will result in a large sparse lookup array")

        # create mapping empty
        mapping_array = torch.full(
            size=(len(minimum), upper_bound),
            fill_value=padding_value,
            dtype=torch.int32,
        )

        # fill empty with vocabulary
        stride = 0
        # transpose from event to feature loop
        for feature_idx, feature in enumerate(indice_array):
            unique = torch.unique(feature, dim=None)
            mapping_array[feature_idx, unique] = torch.arange(
                stride, stride + len(unique),
                dtype=torch.int32,
            )
            stride += len(unique)
        return mapping_array, minimum

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # shift input array by their respective minimum and slice translation accordingly
        # map to int to be used as indices
        from IPython import embed; embed(header="MESSAGE Line 499 | File: layers.py")

        shifted = (x - self.min).to(torch.int32)
        output = self.map[self.indices, shifted]
        return output

class CatEmbeddingLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        embedding_dim: int,
        categories: tuple[str],
        expected_categorical_inputs: dict[str, list[int]] | None = None,
        category_dims: int | None = None,
        empty: int = 15,
    ):
        """
        Initializes the categorical feature interface with a tokenizer and an embedding layer with
        given *embedding_dim*.

        The tokenizer maps given *categories* to values defined in *expected_categorical_inputs*.
        Missing values are given a *empty* value, which
        The mapping is defined in .
        The embedding layer then maps this combined feature space into a dense representation.

            embedding_dim (int): Number of dimensions for the embedding layer.
            categories (tuple[str]): Names of the categories as strings.
            expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
                names and values are lists of integers representing the expected values for
                each category.
            empty (int, optional): Value used to represent missing values in the input tensor.
        """
        super().__init__()
        self.tokenizer = None
        self.category_dims = category_dims
        if not self.category_dims and all(x is not None for x in (categories, expected_categorical_inputs)):
            self.tokenizer = CategoricalTokenizer(
                categories=categories,
                expected_categorical_inputs=expected_categorical_inputs,
                empty=empty)
            self.category_dims = self.tokenizer.num_dim
        self.embeddings = torch.nn.Embedding(
            self.category_dims,
            embedding_dim,
        )

        self.ndim = embedding_dim * len(categories)

    @property
    def look_up_table(self) -> torch.FloatTensor | None:
        return self.tokenizer.map if self.tokenizer else None

    def normalize_embeddings(self):
        # normalize the embedding layer to have unit length
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.embeddings.weight**2, dim=-1)).reshape(-1, 1)
            self.embeddings.weight = torch.nn.Parameter(self.embeddings.weight / norm)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.tokenizer:
            x = self.tokenizer(x)

        x = self.embeddings(x)
        return x.flatten(start_dim=1)


class CategoricalInputLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        padding_categorical_layer: torch.nn.Module | None = None,

        *args,
        **kwargs,
    ):
        """
        Input Layer for categorical features.
        Convert the categorical feature into tokens, via Tokenizer and result are Embedding vectors

        embedding_dim (int): Number of dimensions for the embedding layer.
        categories (tuple[str]): Names of the categories as strings.
        expected_categorical_inputs (dict[list[int]]): Dictionary where keys are category
            names and values are lists of integers representing the expected values for
            each category.
        empty (int, optional): Value used to represent missing values in the input tensor.
        """

        super().__init__()
        self.embedding_layer = embedding_layer
        self.ndim = self.embedding_layer.ndim
        self.padding_categorical_layer = dummy_identity(padding_categorical_layer)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        x = self.padding_categorical_layer(x)
        x = self.embedding_layer(x)
        return x

class ContinuousInputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_inputs: tuple[str],
        std_layer: torch.nn.Module | None = None,
        rotation_layer: torch.nn.Module | None = None,
        padding_continuous_layer: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        """
        Enables the use of categorical and continuous features in a single model.
        A tokenizer and embedding layer are created  is created using and an embedding layer.
        The continuous features are passed through a linear layer and then concatenated with the
        categorical features.
        """
        super().__init__()
        self.ndim = len(continuous_inputs)

        self.rotation_layer = dummy_identity(rotation_layer)
        self.std_layer = dummy_identity(std_layer)
        self.padding_continuous_layer = dummy_identity(padding_continuous_layer)

    def forward(self, x):
        x = self.padding_continuous_layer(x)
        x = self.rotation_layer(x)
        x = self.std_layer(x)
        return x


class OptionalInputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_layer_inst: torch.nn.Module | None = None,
        categorical_layer_inst: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.continuous_layer = continuous_layer_inst
        self.categorical_layer = categorical_layer_inst
        self.ndim = self.continuous_layer.ndim + self.categorical_layer.ndim

    def forward(self, *, categorical_inputs, continuous_inputs):

        x = torch.cat(
            [
                self.continuous_layer(continuous_inputs),
                self.categorical_layer(categorical_inputs),
            ],
            dim=1,
        )
        return x

class InputLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        continuous_inputs: tuple[str],
        embedding_dim: int,
        categorical_inputs: tuple[str] | None = None,
        category_dims: int | None = None,
        expected_categorical_inputs: dict[str, list[int]] | None = None,
        empty: int = 15,
        std_layer: torch.nn.Module | None = None,
        rotation_layer: torch.nn.Module | None = None,
        padding_continuous_layer: torch.nn.Module | None = None,
        padding_categorical_layer: torch.nn.Module | None = None,
        *args,
        **kwargs,
    ):
        """
        Enables the use of categorical and continuous features in a single model.
        A tokenizer and embedding layer are created  is created using and an embedding layer.
        The continuous features are passed through a linear layer and then concatenated with the
        categorical features.
        """
        super().__init__()
        self.empty = empty
        self.ndim = len(continuous_inputs)
        self.embedding_layer = None
        # when categories exist
        if categorical_inputs is not None:
            # and categories has clear ranges
            if expected_categorical_inputs is not None:
                self.embedding_layer = CatEmbeddingLayer(
                    embedding_dim=embedding_dim,
                    categories=categorical_inputs,
                    expected_categorical_inputs=expected_categorical_inputs,
                    empty=empty)
            # otherwise ?
            elif category_dims:
                self.embedding_layer = CatEmbeddingLayer(
                    embedding_dim=embedding_dim,
                    category_dims=category_dims,
                    categories=categorical_inputs,
                    empty=empty,
                )

        if self.embedding_layer:
            self.ndim += self.embedding_layer.ndim

        self.rotation_layer = dummy_identity(rotation_layer)
        self.std_layer = dummy_identity(std_layer)
        self.padding_continuous_layer = dummy_identity(padding_continuous_layer)
        self.padding_categorical_layer = dummy_identity(padding_categorical_layer)


    def dummy_identity(self, layer: torch.nn.Module | None) -> torch.nn.Module:
        if layer is None:
            return torch.nn.Identity()
        return layer

    def categorical_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.padding_categorical_layer(x)
        return self.embedding_layer(x)

    def continuous_preprocessing_pipeline(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # preprocessing
        x = self.padding_continuous_layer(x)
        x = self.rotation_layer(x)
        x = self.std_layer(x)
        return x

    def forward(self, categorical_inputs, continuous_inputs):
        # HINT: When comparing this layer with other compare order of inputs
        x = torch.cat(
            [
                self.continuous_preprocessing_pipeline(continuous_inputs),
                self.categorical_preprocessing_pipeline(categorical_inputs),
            ],
            dim=1,
        )
        return x

class DenseNetBlock(torch.nn.Module):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        skip_connection_init=1.0,
        freeze_skip_connection=False,
        activation_functions = "PReLu",
        eps=1e-5,
        normalize=False,
        *args,
        **kwargs,
        ):
        # TODO Docstring
        super().__init__(*args, **kwargs)
        self.input_dim = input_nodes
        self.output_dim = output_nodes + input_nodes
        self.dense_block = DenseBlock(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            activation_functions=activation_functions,
            normalize=normalize,
            eps=eps
        )
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

    def forward(self, x):
        _input = x * self.skip_connection_amplifier
        x = self.dense_block(x)
        x = torch.concatenate((x, _input), dim = 1)
        return x

class ResNetBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        nodes: int,
        activation_functions: str = "LeakyReLu",
        skip_connection_init: float = 1,
        freeze_skip_connection: float = False,
        eps: float = 1e-5,
        normalize = True,
        *args,
        **kwargs,
    ):
        """
        ResNetBlock consisting of a linear layer, batch normalization, and an activation function.
        A adjustable skip connection connects input and output of the block.
        The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
        The dimension of the input and output of the block are defined by *nodes*.
        If skip_connection_init is set to 0, the skip connection is disabled.
        This also make if possible to use different in_nodes and out_nodes must can be different.
        To freeze the skip connection parameter, set *freeze_skip_connection* to True.

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
            skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
            freeze_skip_connection (bool, optional): Freeze leanable skipconnection parameter. Defaults to False.
        """
        super().__init__(*args, **kwargs)

        self.nodes = nodes
        self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

        self.linear = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn = self.act_func


    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        skip_connection = self.skip_connection_amplifier * x
        x = self.linear(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = x + skip_connection
        return x

class DenseBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        input_nodes: float,
        output_nodes: float,
        activation_functions: str = "LeakyReLu",
        eps: float = 1e-5,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        DenseBlock is a dense block that consists of a linear layer, batch normalization, and an activation function.

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
        """
        super().__init__()
        self.input_dim = input_nodes
        self.output_dim = output_nodes

        self.linear = WeightNormalizedLinear(self.input_dim, self.output_dim, bias=True, normalize=normalize)
        self.bn = torch.nn.BatchNorm1d(self.output_dim, eps=eps)
        self.act_fn = self._get_attr(torch.nn.modules.activation, activation_functions)()

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x

class ResNetPreactivationBlock(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        nodes: int,
        activation_functions: str = "PReLu",
        skip_connection_init: float = 1,
        freeze_skip_connection: bool = False,
        eps: float = 1e-5,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        """
        Residual block that consists of a linear layer, batch normalization, and an activation function.
        A adjustable skip connection connects input and output of the block.
        The adjustable skip connection has a learnable parameter, *skip_connection_amplifier*.
        The dimension of the input and output of the block are defined by *nodes*.
        If skip_connection_init is set to 0, the skip connection is disabled.
        This also make if possible to use different in_nodes and out_nodes must can be different.
        To freeze the skip connection parameter, set *freeze_skip_connection* to True.

        More information can be found in the original paper: https://arxiv.org/abs/1603.05027

        Args:
            nodes (int): Number of nodes in the block.
            activation_functions (str, optional): Name of the pytorch activation function, case insenstive.
                Defaults to "LeakyReLu".
            skip_connection_init (int, optional): Start value of the skipconnection. Defaults to 1.
            freeze_skip_connection (bool, optional): Freeze skipconnection parameter. Defaults to False.
        """
        super().__init__()
        self.nodes = nodes
        self.act_func = self._get_attr(torch.nn.modules.activation, activation_functions)()
        self.skip_connection_amplifier = torch.nn.Parameter(torch.ones(1) * skip_connection_init)
        if freeze_skip_connection:
            self.skip_connection_amplifier.requires_grad = False

        self.linear_1 = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn_1 = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn_1 = self.act_func
        self.linear_2 = WeightNormalizedLinear(self.nodes, self.nodes, bias=True, normalize=normalize)
        self.bn_2 = torch.nn.BatchNorm1d(self.nodes, eps=eps)
        self.act_fn2 = self.act_func

    def _get_attr(self, obj, attr):
        for o in dir(obj):
            if o.lower() == attr.lower():
                return getattr(obj, o)
        else:
            raise AttributeError(f"Object has no attribute '{attr}'")

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        skip_connection = self.skip_connection_amplifier * x
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.act_fn_1(x)
        x = self.linear_2(x)
        x = self.bn_2(x)
        x = x + skip_connection
        return self.act_fn2(x)


class StandardizeLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        mean: float = 0.,
        std: float = 1.,
    ):
        """
        Standardizes the input tensor with given *mean* and *std* tensor.
        If no value is provided, mean and std are set to 0 and 1, resulting in no scaling.

        Args:
            mean (torch.tensor, optional): Mean tensor. Defaults to torch.tensor(0.).
            std (torch.tensor, optional): Standard tensor. Defaults to torch.tensor(1.).
        """
        super().__init__()
        self.mean = torch.nn.Buffer(mean.detach().clone(), persistent=True)
        self.std = torch.nn.Buffer(std.detach().clone(), persistent=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = (x - self.mean) / self.std
        return x.to(torch.float32)

    def _type_check(self, mean: torch.FloatTensor, std: torch.FloatTensor):
        if not all([isinstance(value, torch.Tensor) for value in [mean, std]]):
            raise TypeError(f"given mean or std needs to be tensor, but is {type(mean)}{type(std)}")

    def update_buffer(self, mean: torch.
                        tensor, std: torch.tensor):
        """
        Update the mean and std parameter.

        Args:
            mean (torch.tensor): Mean value.
            std (torch.tensor): Standard deviation value.
        """
        self._type_check(mean=mean, std=std)
        self.mean = mean.type_as(self.mean)
        self.std = std.type_as(self.std)

class RotatePhiLayer(torch.nn.Module):  # noqa: F811
    def __init__(
        self,
        columns: list[str] | None,
        ref_phi_columns: list[str] | None = ("lepton1", "lepton2"),
        rotate_columns: list[str] | None = ("bjet1", "bjet2", "fatjet", "lepton1", "lepton2"),
        separator="_",
    ):
        """
        Rotate specific *columns* given in *rotate_columns* relative to reference in *ref_phi_columns*.
        """
        super().__init__()
        self.separator=separator
        self.ref_indices = torch.nn.Buffer(self.find_indices_of(columns, ref_phi_columns, True), persistent=True)
        self.rotate_indices = torch.nn.Buffer(self.find_indices_of(columns, rotate_columns, True), persistent=True)

    def load_state_dict(self, state_dict: dict, strict: bool, assign: bool):
        # overload load_state_dict to set buffer sizes to same of state dict
        for name in self.state_dict().keys():
            self.__setattr__(name, torch.zeros_like(state_dict[name]))
        super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)

    def find_indices_of(
        self,
        search_in: list[str],
        search_for: list[str],
        _expand: bool = False,

    ) -> torch.FloatTensor | None:
        if search_in is None or search_for is None:
            return None

        if _expand:
            search_for = self._expand(search_for, separator=self.separator)
        return torch.tensor([tuple(map(search_in.index, particle)) for particle in search_for])

    def _expand(self, columns: list[str] | str, separator="_") -> list[tuple[str, str]]:
        # adds px, py to columns and return them as tuple
        columns = [columns] if isinstance(columns, str) else columns
        return [tuple(f"{col}{separator}{suffix}" for suffix in ("px", "py")) for col in columns]

    def calc_phi(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
    ) -> torch.FloatTensor:
        def arctan2(
            x: torch.FloatTensor,
            y: torch.FloatTensor,
        ) -> torch.FloatTensor:
            # calculate arctan2 using arctan, since onnx does not have support for arctan2
            # torch constants are not tensors for some reason
            pi = torch.tensor(torch.pi)

            # handle special cases
            # quadrant handling: (x < 0,y >= 0) -> arctan + pi, (x < 0,y < 0) -> arctan - pi
            phi_shift = torch.zeros_like(x)
            torch.where(torch.logical_and(x < 0, y < 0), -pi, phi_shift, out=phi_shift)
            torch.where(torch.logical_and(x < 0, y >= 0), pi, phi_shift, out=phi_shift)

            # edges  with x == 0
            phis = torch.arctan(y / x)
            # right edge -> pi/2, left edge -> -pi/2, both 0 -> 0
            torch.where(torch.logical_and(x == 0, y > 0), (pi / 2), phis, out=phis)
            torch.where(torch.logical_and(x == 0, y < 0), -(pi / 2), phis, out=phis)
            torch.where(torch.logical_and(x == 0, y == 0), torch.tensor(0), phis, out=phis)
            return phis + phi_shift
        return arctan2(x=x, y=y)

    def rotate_pt_to_phi(
        self,
        px: torch.FloatTensor,
        py: torch.FloatTensor,
        ref_phi: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # rotate px, py relative to ref_phi
        # returns px, py rotated
        pt = torch.sqrt(torch.square(px) + torch.square(py))
        old_phi = self.calc_phi(x=px, y=py)
        new_phi = old_phi - ref_phi
        return pt * torch.cos(new_phi), pt * torch.sin(new_phi)

    def calc_ref_phi(
        self,
        array,
    ) -> torch.FloatTensor:
        px1, py1 = self.get_kinematics(array, self.ref_indices[0])
        px2, py2 = self.get_kinematics(array, self.ref_indices[1])

        ref_phi = self.calc_phi(
            x=px1 + px2,
            y=py1 + py2,
        )
        return ref_phi

    def get_kinematics(
        self,
        array: torch.FloatTensor,
        ref_indices: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        x, y = ref_indices
        return array[:, x], array[:, y]

    def rotate_columns(self, array: torch.FloatTensor, ref_phi: torch.FloatTensor) -> torch.FloatTensor:
        # px, py pairs in fixed order
        for rotate_indice in self.rotate_indices:
            px, py = self.get_kinematics(array, rotate_indice)

            new_px, new_py = self.rotate_pt_to_phi(
                px=px,
                py=py,
                ref_phi=ref_phi,
            )
            array[:, rotate_indice[0]] = new_px
            array[:, rotate_indice[1]] = new_py
        return array

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        ref_phi = self.calc_ref_phi(x)
        return self.rotate_columns(x, ref_phi)


class LBN(torch.nn.Module):
    """
    Torch implementation of the LBN (Lorentz Boosted Network) feature extractor.
    For details and nomenclature see https://arxiv.org/pdf/1812.09722.
    """

    KNOWN_FEATURES = [
        "e", "px", "py", "pz",
        "pt", "eta", "phi", "m",
        "pair_cos", "pair_dr",
    ]

    DEFAULT_FEATURES = ["e", "pt", "eta", "phi", "m", "pair_cos"]

    def __init__(
        self,
        N: int,
        M: int,
        *,
        features: Sequence[str] | None = None,
        weight_init_scale: float | int = 1.0,
        clip_weights: bool = False,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()


        # validate features
        if features is None:
            features = self.DEFAULT_FEATURES
        for f in features:
            if f not in self.KNOWN_FEATURES:
                raise ValueError(f"unknown feature '{f}', known features are: {self.KNOWN_FEATURES}")

        # store settings
        self.N = N
        self.M = M
        self.features = list(features)
        self.weight_init_scale = weight_init_scale
        self.clip_weights = clip_weights
        self.eps = eps

        # constants
        self.register_buffer("I4", torch.eye(4, dtype=torch.float32))  # (4, 4)
        self.register_buffer("U", torch.tensor([[-1, 0, 0, 0], *(3 * [[0, -1, -1, -1]])], dtype=torch.float32))
        self.register_buffer("U1", self.U + 1)
        self.register_buffer(
            "lower_tril_indices",
            torch.arange(M**2).reshape((M, M))[torch.tril(torch.ones(M, M, dtype=torch.bool), -1)],
        )

        # randomly initialized weights for projections
        self.particle_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)
        self.restframe_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)

    @property
    def out_features(self) -> int:
        # determine number of pair-wise feature projections
        n_pair = sum(1 for f in self.features if f.startswith("pair_"))

        # compute output dimension
        n = (
            (
            len(self.features) - n_pair) * self.M +
            n_pair * (self.M**2 - self.M) // 2
        )
        return n

    def ndim(self):
        # dim normal features: m * 4
        non_pair_features_dim = len([f for f in self.features if "pair" not in f]) * self.M

        # dim pair-wise features is MxM Matrix, which diagonal = 0, and symmetric: (M^2 - M)/2
        pair_features_dim = len([f for f in self.features if "pair" in f]) * ((self.M**2 - self.M) / 2)
        num = int(non_pair_features_dim + pair_features_dim)
        return num

    def __repr__(self) -> str:
        params = {
            "N": self.N,
            "M": self.M,
            "features": ",".join(self.features),
            "clip": self.clip_weights,
        }
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str}, {hex(id(self))})"

    def update_boosted_vectors(self, boosted_vecs: torch.Tensor) -> torch.Tensor:
        return boosted_vecs

    def forward(self, input_vecs) -> torch.Tensor:
        # e, px, py, pz: (B, N)
        E, PX, PY, PZ = range(4)

        # stack 4-vectors
        # input_vecs = torch.stack((e, px, py, pz), dim=1)  # (B, 4, N)

        # optionally clip weights to prevent them going negative
        particle_w = self.particle_w
        restframe_w = self.restframe_w
        if self.clip_weights:
            particle_w = torch.clamp(particle_w, min=0.0)
            restframe_w = torch.clamp(restframe_w, min=0.0)

        # create combinations
        particle_vecs = torch.matmul(input_vecs, particle_w)  # (B, 4, M)
        restframe_vecs = torch.matmul(input_vecs, restframe_w)  # (B, 4, M)
        # transpose to (B, M, 4)
        particle_vecs = particle_vecs.permute(0, 2, 1)
        restframe_vecs = restframe_vecs.permute(0, 2, 1)

        # regularize vectors such that e > p
        particle_p = torch.sum(particle_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
        # avoid in-place modifications which break autograd when tensors are used in multiple
        # places; construct a new tensor with the adjusted energy component
        new_particle_E = torch.maximum(particle_vecs[..., E], particle_p + self.eps)
        particle_vecs = torch.stack(
            [new_particle_E, particle_vecs[..., PX], particle_vecs[..., PY], particle_vecs[..., PZ]],
            dim=-1,
        )

        restframe_p = torch.sum(restframe_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
        new_restframe_E = torch.maximum(restframe_vecs[..., E], restframe_p + self.eps)
        restframe_vecs = torch.stack(
            [new_restframe_E, restframe_vecs[..., PX], restframe_vecs[..., PY], restframe_vecs[..., PZ]],
            dim=-1,
        )

        # create boost objects
        restframe_m = (restframe_vecs[..., E]**2 - restframe_p**2)**0.5  # (B, M)
        gamma = restframe_vecs[..., E] / (restframe_m + self.eps) # (B, M)
        beta = restframe_p / restframe_vecs[..., E]  # (B, M)
        beta_vecs = restframe_vecs[..., PX:] / restframe_vecs[..., E, None]  # (B, M, 3)
        n_vecs = beta_vecs / beta[..., None]  # (B, M, 3)
        e_vecs = torch.cat([torch.ones_like(n_vecs[..., :1]), -n_vecs], dim=-1)  # (B, M, 4)

        # build Lambda
        Lambda = self.I4 + (
            (self.U + gamma[..., None, None]) *
            (self.U1 * beta[..., None, None] - self.U) *
            (e_vecs[..., None] * e_vecs[..., None, :])
        )  # (B, M, 4, 4)

        # apply boosting
        boosted_vecs = (Lambda @ particle_vecs[..., None])[..., 0]

        # hook to update boosted vectors if desired
        boosted_vecs = self.update_boosted_vectors(boosted_vecs)

        # cached feature provision
        cache = {}
        def get(feature: str) -> torch.Tensor:
            # check cache first
            if feature in cache:
                return cache[feature]
            # live feature access
            if feature == "e":
                return boosted_vecs[..., E]
            if feature == "px":
                return boosted_vecs[..., PX]
            if feature == "py":
                return boosted_vecs[..., PY]
            if feature == "pz":
                return boosted_vecs[..., PZ]
            # cached  access
            if feature == "pt2":
                f = get("px")**2 + get("py")**2
            elif feature == "pt":
                f = get("pt2")**0.5
            elif feature == "p2":
                f = get("pt2") + get("pz")**2
            elif feature == "p":
                f = get("p2")**0.5
            elif feature == "eta":
                # clamp when near -1 or 1
                ratio = torch.clip(get("pz") / get("p"), min = -1 + self.eps, max = 1 - self.eps)
                f = torch.atanh(ratio)
            elif feature == "phi":
                f = torch.atan2(get("py"), get("px"))
            elif feature == "m":
                f = (torch.maximum(get("e")**2, get("p2")) - get("p"))**0.5
            elif feature == "pair_cos":
                boosted_pvecs = boosted_vecs[..., PX:]  # (B, M, 3)
                boosted_p = get("p")
                f = (
                    (boosted_pvecs @ boosted_pvecs.transpose(1, 2)) /
                    (boosted_p[..., None] @ boosted_p[:, None, :])
                ).flatten(start_dim=1)[..., self.lower_tril_indices]  # (B, (M**2-M)/2)
            elif feature == "pair_dr":
                boosted_phi = get("phi")
                boosted_eta = get("eta")
                boosted_dphi = abs(boosted_phi[..., None] - boosted_phi[:, None, :])  # (B, M, M)
                boosted_dphi = boosted_dphi.flatten(start_dim=1)[..., self.lower_tril_indices]  # (B, (M**2-M)/2)
                boosted_dphi = torch.where(boosted_dphi > torch.pi, 2 * torch.pi - boosted_dphi, boosted_dphi)
                boosted_deta = boosted_eta[..., None] - boosted_eta[:, None, :]  # (B, M, M)
                boosted_deta = boosted_deta.flatten(start_dim=1)[..., self.lower_tril_indices]  # (B, (M**2-M)/2)
                f = (boosted_dphi**2 + boosted_deta**2)**0.5
            else:
                raise RuntimeError(f"unknown feature '{feature}'")
            # cache and return
            cache[feature] = f
            return f

        # when not clipping weights, boosted vectors can have e < p
        if not self.clip_weights:
            new_boosted_E = torch.maximum(boosted_vecs[..., E], get("p") + self.eps)
            boosted_vecs = torch.stack(
                [new_boosted_E, boosted_vecs[..., PX], boosted_vecs[..., PY], boosted_vecs[..., PZ]],
                dim=-1,
            )

        # collect and combine features
        features = torch.cat([get(feature) for feature in self.features], dim=1)  # (B, F)
        return features


class LBNFeaturerExtractor(torch.nn.Module):
    """
    Helper Layer to filter out correct features necessary for the LBN to run. The filtering is done by naming!
    Thus changing names of features result automatically into  a broken functionality.

    Args:
        continuous_features (list[str]): List of all continuous features that should be extracted for the LBN.
    """

    def __init__(
        self,
        continuous_features: list[str],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.continuous_features = continuous_features
        self.particles = self.find_particles_components_in_features()

    @property
    def num_particles(self):
        return len(self._particles)

    @property
    def _particles(self):
        return ("vis_tau1", "vis_tau2", "bjet1", "bjet2", "met", "nu1", "nu2")

    def find_particles_components_in_features(self):
        # returns dictionary with all indicies of the feature components
        def find_components(particle):
            particle_components = {}
            for idx, s in enumerate(self.continuous_features):
                if particle in s:
                    component = s.split("_")[-1]
                    particle_components[component] = idx
            return particle_components

        particles_components = {particle: find_components(particle) for particle in self._particles}

        # filter the indices
        indicies = lambda particle, features : [particles_components[particle][f] for f in features]
        particles = {}

        for f in self._particles[:-3]:
            particles[f] = indicies(f, ("e", "px", "py", "pz"))
        particles["met"] = indicies("met", ("px", "py"))

        particles["nu1"] = indicies("nu1", ("px", "py", "pz"))
        particles["nu2"] = indicies("nu2", ("px", "py", "pz"))
        return particles

    def slice_particles_from_tensor(self, tensor):
        t = []
        for f in self._particles[:-3]:
            t.append(tensor[:, self.particles[f]])

        # met is special, since we need to reconstruct it: (pt, px, py ,0)
        met_kinematics = tensor[:, self.particles["met"]]

        met_pt = torch.sqrt(torch.sum(met_kinematics**2, axis=1)) # TODO float64?
        met_pz = torch.zeros_like(met_pt)
        met = torch.stack((met_pt, met_kinematics[:, 0], met_kinematics[:, 1], met_pz), axis=1)
        t.append(met)
        # add nu, separetley since pt = e: (pt, px, py, pz)
        for num in (1, 2):
            nu_kinematics = tensor[:, self.particles[f"nu{num}"]]
            nu_e = torch.sqrt(torch.sum(nu_kinematics**2, axis=1))
            nu = torch.stack((nu_e, nu_kinematics[:, 0], nu_kinematics[:, 1], nu_kinematics[:, 2]), axis=1)
            t.append(nu)
        # combine everything
        t = torch.stack(t, axis=-1) # B, FEATURES (4), particles (7)
        return t

    def forward(self, x):
        return self.slice_particles_from_tensor(x)
        # return the indices of all particle features so the layer can slice them from tensors

class LBN_DNN(torch.nn.Module):
    def __init__(
        self,
        continuous_features: list[str],
        M: int = 10,
        # no N necessary for LBN, getting information from Feature extractor
        weight_init_scale: float | int = 1.0,
        clip_weights: bool = False,
        eps: float = 1.0e-5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lbn_feature_extractor = LBNFeaturerExtractor(continuous_features=continuous_features)

        self.lbn = LBN(M=M, N=self.lbn_feature_extractor.num_particles, clip_weights=clip_weights, eps=eps, weight_init_scale=weight_init_scale)
        self.lbn_batch_norm = torch.nn.BatchNorm1d(self.lbn.ndim())


    @property
    def ndim(self):
        return self.lbn.ndim()

    def forward(self, x):
        x = self.lbn_feature_extractor(x)
        x = self.lbn(x)
        x = self.lbn_batch_norm(x)
        return x


class BinningLayerV1(torch.nn.Module):
    def __init__(
        self,
        init_edges: list[float],
        kernel_cls,
        kernel_cfg,
        *args,
        **kwargs
        ):
        """
        Args:
            init_edges (list[float]): _description_
        """
        super().__init__(*args, **kwargs)
        # TODO currently no fusion allowed
        # TODO maybe good idea, when going below certain threshold
        self.init_edges = init_edges
        self.edges = torch.nn.Parameter(init_edges)
        self.kernel_cls = kernel_cls

        # config allows bin-wise configuration
        # if bin number (as int) present overwrite general value
        self.original_kernel_cfg = kernel_cfg
        # self.kernel_configs = self.build_cfg(self.original_kernel_cfg)
        self.kernels = self.build_kernels(kernel_cfg=kernel_cfg)

    def get_edges(self):
        e = self.edges.detach()
        low, up = e[:-1], e[1:]
        return [(l,u) for l,u in zip(low,up)]

    def build_cfg(self, base_cfg):
        config = {}
        for _bin in range(0, self.num_bins):
            if _bin in base_cfg.keys():
                fill_cfg = base_cfg[_bin]
            else:
                fill_cfg = base_cfg["general"].copy()
            fill_cfg["bin_type"] = "normal"
            config[_bin] = fill_cfg

        config[0]["bin_type"] = "underflow"
        config[self.num_bins - 1]["bin_type"] = "overflow"
        return config

    def build_kernels(self, kernel_cfg):
        edges = self.get_edges()
        kernels = []

        for bin_index in range(0, self.num_bins):
            bin_config = kernel_cfg.copy()
            edge = edges[bin_index]

            if bin_index == 0:
                bin_config["bin_type"] = "underflow"
            elif bin_index == (self.num_bins - 1):
                bin_config["bin_type"] = "overflow"
            else:
                bin_config["bin_type"] = "normal"

            kernels.append(self.kernel_cls(edge, **bin_config))
        return kernels

    # def build_kernels(self,):
    #     edges = self.get_edges()
    #     kernels = []
    #     range(0, self.num_bins):
    #     for bin_index, cfg in self.kernel_configs.items():
    #         edge = edges[bin_index]
    #         kernels.append(self.kernel_cls(edge=edge, **cfg))
    #     return kernels
    def __repr__(self):
        edges = self.get_edges()
        edges = [(float(l), float(u)) for l,u in edges]

        msg = "Bin Index | Edge Value\n"
        msg+="\n".join([f"{idx}:{edge}" for idx, edge in enumerate(edges)])
        return msg

    @property
    def num_bins(self):
        return len(self.edges) - 1

    def check(self):
        # config has to have "general" key
        assert "general" in self.kernel_cfg.keys()

    def forward(self, x):
        scaled_x = []
        for kernel in self.kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        binned_x = torch.sum(torch.stack(scaled_x, axis=0), axis=0)
        return binned_x


class BinningLayerV2(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        lower_bound: float,
        upper_bound: float,
        space_fn: callable, # like linspace or logspace to create initial edges
        *args,
        **kwargs
        ):
        """
        Args:
            init_edges (list[float]): _description_
        """
        super().__init__(*args, **kwargs)
        # TODO currently no fusion allowed
        # TODO maybe good idea, when going below certain threshold
        self.num_bins = num_bins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.bin_lengths = torch.nn.Parameter(self.init_learnable_edges(space_fn=space_fn), requires_grad=True)


    def init_learnable_edges(self, space_fn):
        # create initial edges with space fn, then create learnable edges by taking the difference of the initial edges
        space_intervalls = space_fn(self.lower_bound, self.upper_bound, self.num_bins + 1)
        differences = space_intervalls[1:] - space_intervalls[:-1]
        return differences

    def reconstruct_bins(self):
        # reconstruct bins from bin_length
        edges = torch.cumsum(self.bin_length, dim=0)
        bin_center = edges - edges / 2
        return edges

    def kernel(self, x: torch.tensor, bin_type="normal") -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """

        # prepare edges and function
        start, end = 0, 0 # TODO
        gaussian_fn = lambda x, shift, std: torch.exp(-((x - shift) / (2* std))**2)
        # set values depending on the bin type
        if bin_type == "overflow":
            f = torch.where(x < start, gaussian_fn(x, shift=start), 1)
        elif bin_type == "underflow":
            f = torch.where(x > end, gaussian_fn(x, shift=end), 1)
        else:
            f = torch.where(x < start, gaussian_fn(x, shift=start), 1)
            f = torch.where(x > end, gaussian_fn(x, shift=end), f)
        return f



    def forward(self, x):
        scaled_x = []
        for kernel in self.kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        binned_x = torch.sum(torch.stack(scaled_x, axis=0), axis=0)
        return binned_x


class BinningLayerRight(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        bounds: tuple[float],

        binning_fn: callable, # like linspace or logspace to create initial edges
        kernel_cls,
        kernel_cfg,
        *args,
        **kwargs
        ):
        """
        Creates *num_bins* kernel instances of *kernel_cls* with configuration defined in *kernel_cfg*.
        The initial edge are defined by a given binning function *binning_fn*.
        The lower and upper bounds are given as tuple *bounds*.

        For every prediction, add another axis, with num_bins entries.

        Example we have an prediction vector of shape [100, 3] and 20 kernels.
        The resulting Tensors would be [20, 100, 3]

        Args:
            init_edges (list[float]): _description_
        """
        super().__init__(*args, **kwargs)
        # TODO currently no fusion allowed

        self.num_bins = num_bins
        self.bounds = bounds
        self.init_learnable_edges(space_fn=binning_fn)

        self.kernel_cls = kernel_cls
        self.kernel_cfg = kernel_cfg
        self.kernel_cache = None



    @property
    def lower_edge(self):
        return self.bounds[0]

    @property
    def upper_edge(self):
        return self.bounds[1]

    @property
    def interval(self):
        return self.upper_edge - self.lower_edge

    @property
    def bin_edges(self):
        # calculate absolute widht of bins
        relative_part = self.relative_bin_width.detach()
        interval = self.upper_edge - self.lower_edge
        abs_width = interval * relative_part

        # right edge is sum of abs_bins + lowest edge
        # left edge is
        shift = self.lower_edge
        right_edge = shift + torch.cumsum(abs_width, dim=0)
        left_edge = right_edge - abs_width
        return torch.stack((left_edge, right_edge), dim = 1)

    def init_learnable_edges(self, space_fn):
        """
        Takes *space_fn* that defines an interval space, like linspace and register this interval.
        Afterwards parametrize the interval to their relative contribution.

        Args:
            space_fn (_type_): _description_

        Returns:
            _type_: _description_
        """
        intervalls = space_fn(self.lower_edge, self.upper_edge, self.num_bins + 1)
        relative_width = (intervalls[1:] - intervalls[:-1]) / self.interval

        self.relative_bin_width = torch.nn.Buffer(relative_width)
        parametrize.register_parametrization(self, "relative_bin_width", torch.nn.Softmax(dim=0))

    def kernels(self ,load_cache=False) -> torch.tensor:
        """
        Actual kernel implementation, containing 3 parts: left gaussian, horizontal 1 and right gaussian.
        A value *x* is mapped to an y value using these function.

        Args:
            x (torch.tensor): Input tensor of x values that are mapped to y.

        Returns:
            torch.tensor: y value tensor resulting form the function. Is by default between 0 and 1
        """
        # kernels should only be rebuild, when there is no cache or load_cache is false
        if load_cache and self.kernel_cache:
            return self.kernel_cache

        kernels = []
        for bin_num, edge in enumerate(self.bin_edges):
            if bin_num == 0:
                bin_type = "underflow"
            elif bin_num == (self.num_bins - 1):
                bin_type = "overflow"
            else:
                bin_type = "normal"
            kernel = self.kernel_cls(
                edges=edge,
                bin_type=bin_type,
                **self.kernel_cfg,
                )
            kernels.append(kernel)
        # save in cache
        self.kernel_cache = kernels
        return kernels

    def forward(self, x):
        scaled_x = []
        kernels = self.kernels(load_cache=False)
        for kernel in kernels:
            scale = kernel(x)
            scaled_x.append(scale * x)
        return torch.stack(scaled_x, dim=0)
