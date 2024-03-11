from math import prod
from numbers import Number
from typing import Dict, List

from torch import Tensor, cat, nn, randn


class VisionTransformer(nn.Module):
    """Configurable vision transformer (ViT).

    Implements the vision transformer as proposed by Dosovitskiy et al., ICLR 2021.

    Args:
        num_channels (int): Number of channels of the input.
        input_sizes (list of int): Spatial sizes of the input.
        output_size (int): Size of the output layer.
        patch_size (int): Size of a patch in one dimension.
        latent_size (int): Size of the embedding.
        num_heads (int): Number of attention heads in each encoder block.
        num_layers (int): Number of encoder blocks.
        encoder_size (int): Number of hidden units in each encoder MLP.
        head_size (int or list of int): Sizes of hidden layers in MLP head.
        dropout (float): Dropout probabilities of embedding, encoder and MLP head.
            Default: 0.
        activation (str): Activation function string, either `'relu'` or `'gelu'`.
            Default: `'gelu'`.
    """

    def __init__(
        self,
        num_channels: int,
        input_sizes: List[int],
        output_size: int,
        patch_size: int,
        latent_size: int,
        num_heads: int,
        num_layers: int,
        encoder_size: int,
        head_size: int | List[int],
        dropout: float = 0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.input_sizes = input_sizes
        self.output_size = output_size
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.encoder_size = encoder_size
        self.head_size = head_size
        self.dropout = dropout
        self.activation = activation
        self.embedding = Embedding(
            num_channels, input_sizes, patch_size, latent_size, dropout
        )
        layer_norm = nn.LayerNorm(latent_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=num_heads,
            dim_feedforward=encoder_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, layer_norm, enable_nested_tensor=False
        )
        if isinstance(head_size, Number):
            head_size = [head_size]
        self.mlp_head = MLP(
            input_size=latent_size,
            output_size=output_size,
            hidden_sizes=head_size,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, data: Tensor) -> Tensor:
        data = self.embedding(data)
        data = self.encoder(data)
        # Take encoder output corresponding to class_token
        output = self.mlp_head(data[:, 0, :])
        return output

    def get_init_kwargs(self):
        """Collects all `__init__` keyword arguments of the object.

        The return dictionary can be used as keyword arguments to initialize a new
        object with the same properties.

        Returns:
            dict: Keyword arguments including `'num_channels'`, `'input_size'`,
            `'output_size'`, `'patch_size'`, `'latent_size'`, `'num_heads'`,
            `'num_layers'`, `'encoder_size'`, `'head_size'`, `'dropout'` and
            `'activation'`.
        """
        kwargs = {
            "num_channels": self.num_channels,
            "input_sizes": self.input_sizes,
            "output_size": self.output_size,
            "patch_size": self.patch_size,
            "latent_size": self.latent_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "encoder_size": self.encoder_size,
            "head_size": self.head_size,
            "dropout": self.dropout,
            "activation": self.activation,
        }
        return kwargs


class Embedding(nn.Module):
    """An embedding for a vision transformer.

    Splits the input into patches, and projects the patches.  Also adds a class token
    and a position embedding.

    Args:
        num_channels (int): Number of channels of the input.
        input_sizes (list of int): Spatial sizes of the input.
        patch_size (int): Size of a patch in one dimension.
        latent_size (int): Size of the embedding.
        dropout (float): Dropout probability.  Default: 0.
    """

    def __init__(
        self,
        num_channels: int,
        input_sizes: List[int],
        patch_size: int,
        latent_size: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        # Use Unfold to split the input image into patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        num_patches = prod(
            [(input_size - patch_size) // patch_size + 1 for input_size in input_sizes]
        )
        flattened_size = num_channels * patch_size ** len(input_sizes)
        self.linear = nn.Linear(flattened_size, latent_size)
        self.class_token = nn.Parameter(randn(1, 1, latent_size))
        self.position_embeddings = nn.Parameter(randn(1, num_patches + 1, latent_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Tensor) -> Tensor:
        # Split images into patches and flatten into
        # batch_size x num_patches x (num_channels * patch_size ** len(input_sizes))
        batch_size = data.shape[0]
        data = self.unfold(data).permute(0, 2, 1).contiguous()
        data = self.linear(data)
        output = self.dropout(
            cat([self.class_token.expand(batch_size, -1, -1), data], dim=1)
            + self.position_embeddings
        )
        return output


class MLP(nn.Module):
    """Configurable multilayer perceptron.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_sizes (list of int, optional): Sizes of hidden layers.  Default: `None`.
        dropout (float or list of float, optional): Dropout probabilities of each
            hidden layer.  If `None`, no dropout will be used.  If single float, the
            same dropout probability will be used for all hidden layers.
            Default: `None`.
        activation (str): Activation function string, either `'relu'` or `'gelu'`.
            Default: `'relu'`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        dropout: float | List[float] = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        if hidden_sizes is None:
            hidden_sizes = []
        layers = [
            nn.Linear(i, j)
            for i, j in zip([input_size] + hidden_sizes, hidden_sizes + [output_size])
        ]
        if dropout is None:
            dropout_modules = []
        elif isinstance(dropout, Number):
            dropout_modules = [nn.Dropout(dropout) for _ in range(len(hidden_sizes))]
        else:
            dropout_modules = [nn.Dropout(rate) for rate in dropout]
        modules = []
        for i, layer in enumerate(layers):
            modules.append(layer)
            if i < len(layers) - 1:
                if activation == "relu":
                    modules.append(nn.ReLU())
                elif activation == "gelu":
                    modules.append(nn.GELU())
                else:
                    raise ValueError("unknown activation")
                if len(dropout_modules) > i:
                    modules.append(dropout_modules[i])
        self.linear_stack = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> Tensor:
        data = self.flatten(data)
        output = self.linear_stack(data)
        return output

    def get_init_kwargs(self) -> Dict:
        """Collects all `__init__` keyword arguments of the object.

        The return dictionary can be used as keyword arguments to initialize a new
        object with the same properties.

        Returns:
            dict: Keyword arguments including `'input_size'`, `'output_size'`,
                `'hidden_sizes'`, `'dropout'` and `'activation'`.
        """
        input_size = self.linear_stack[0].in_features
        output_size = self.linear_stack[-1].out_features
        hidden_sizes = []
        dropout = []
        for module in self.linear_stack[:-1]:
            if isinstance(module, nn.Linear):
                hidden_sizes.append(module.out_features)
            if isinstance(module, nn.Dropout):
                dropout.append(module.p)
        if hidden_sizes and isinstance(self.linear_stack[1], nn.GELU):
            activation = "gelu"
        else:
            activation = "relu"
        kwargs = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "activation": activation,
        }
        return kwargs
