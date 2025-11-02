from math import prod
from numbers import Number
from typing import Literal

from torch import Tensor, cat, nn, randn

type EncoderActivation = Literal["relu", "gelu"]
type MLPActivation = Literal["relu", "gelu", "tanh"]


class VisionTransformer(nn.Module):
    """Configurable vision transformer (ViT).

    Implements the vision transformer as proposed by Dosovitskiy et al., ICLR 2021.

    Args:
        num_channels (int): Number of channels of the input.
        input_sizes (list of int): Spatial sizes of the input.
        output_size (int): Size of the output layer.
        patch_size (int): Size of a patch in one dimension.
        num_heads (int): Number of attention heads in each encoder block.
        latent_size_multiplier (int): Yields the size of the embedding when multiplied
            with `num_heads`.
        num_layers (int): Number of encoder blocks.
        encoder_size (int): Number of hidden units in each encoder MLP.
        head_size (int or list of int): Sizes of hidden layers in MLP head.
        dropout (float, optional): Dropout probabilities of embedding, encoder and MLP
            head.  Default: 0.
        encoder_activation (EncoderActivation, optional): Encoder activation function
            string, either `'relu'` or `'gelu'`.  Default: `'gelu'`.
        head_activation (MLPActivation, optional): MLP head activation function string,
            `'relu'`, `'gelu'` or `'tanh'`.  Default: `'gelu'`.
    """

    def __init__(
        self,
        num_channels: int,
        input_sizes: list[int],
        output_size: int,
        patch_size: int,
        num_heads: int,
        latent_size_multiplier: int,
        num_layers: int,
        encoder_size: int,
        head_size: int | list[int],
        dropout: float = 0,
        encoder_activation: EncoderActivation = "gelu",
        head_activation: MLPActivation = "gelu",
    ) -> None:
        super().__init__()
        latent_size = latent_size_multiplier * num_heads
        self.embedding = Embedding(
            num_channels, input_sizes, patch_size, latent_size, dropout
        )
        layer_norm = nn.LayerNorm(latent_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_size,
            nhead=num_heads,
            dim_feedforward=encoder_size,
            dropout=dropout,
            activation=encoder_activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, layer_norm, enable_nested_tensor=False
        )
        if isinstance(head_size, int):
            hidden_sizes = [head_size]
        else:
            hidden_sizes = head_size
        self.mlp_head = MLP(
            input_size=latent_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            activation=head_activation,
        )

    def forward(self, data: Tensor) -> Tensor:
        data = self.embedding(data)
        data = self.encoder(data)
        # Take encoder output corresponding to class_token
        output = self.mlp_head(data[:, 0, :])
        return output


class Embedding(nn.Module):
    """An embedding for a vision transformer.

    Splits the input into patches, and projects the patches.  Also adds a class token
    and a position embedding.

    Args:
        num_channels (int): Number of channels of the input.
        input_sizes (list of int): Spatial sizes of the input.
        patch_size (int): Size of a patch in one dimension.
        latent_size (int): Size of the embedding.
        dropout (float, optional): Dropout probability.  Default: 0.
    """

    def __init__(
        self,
        num_channels: int,
        input_sizes: list[int],
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
        hidden_sizes (list of int or None, optional): Sizes of hidden layers.
            Default: `None`.
        dropout (float or list of float or None, optional): Dropout probabilities of
            each hidden layer.  If `None`, no dropout will be used.  If single float,
            the same dropout probability will be used for all hidden layers.
            Default: `None`.
        activation (MLPActivation, optional): Activation function string, `'relu'`,
            `'gelu'` or `'tanh'`.  Default: `'relu'`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float | list[float] | None = None,
        activation: MLPActivation = "relu",
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
            assert isinstance(dropout, list)
            dropout_modules = [nn.Dropout(rate) for rate in dropout]
        modules: list[nn.Linear | nn.ReLU | nn.GELU | nn.Tanh | nn.Dropout] = []
        for i, layer in enumerate(layers):
            modules.append(layer)
            if i < len(layers) - 1:
                if activation == "relu":
                    modules.append(nn.ReLU())
                elif activation == "gelu":
                    modules.append(nn.GELU())
                elif activation == "tanh":
                    modules.append(nn.Tanh())
                else:
                    raise ValueError(f"unknown activation '{activation}'")
                if len(dropout_modules) > i:
                    modules.append(dropout_modules[i])
        self.linear_stack = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> Tensor:
        data = self.flatten(data)
        output = self.linear_stack(data)
        return output
