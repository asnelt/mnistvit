from typing import List, Union, Dict
from numbers import Number
from torch import nn, Tensor, randn, cat


class VisionTransformer(nn.Module):

    def __init__(self, num_channels, height, width, patch_size, latent_size, num_layers,
                 output_size, mlp_head_sizes):
        super().__init__()
        self.embedding = Embedding(num_channels, height, width, patch_size, latent_size)
        encoders = [TransformerEncoder(latent_size) for _ in range(num_layers)]
        self.encoder_stack = nn.Sequential(encoders)
        self.mlp_head = MLP(input_size=latent_size, output_size=output_size,
                            hidden_sizes=mlp_head_sizes)

    def forward(self, data):
        data = self.embedding(data)
        data = self.encoder_stack(data)
        output = self.mlp_head(data)
        return output

    def get_hyperparameters(self):
        pass


class Embedding(nn.Module):

    def __init__(self, num_channels, height, width, patch_size, latent_size):
        super().__init__()
        # Use Unfold to split the input image into patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        num_patches = (height // patch_size) * (width // patch_size)
        flattened_size = num_channels * patch_size ** 2
        self.linear = nn.Linear(flattened_size, latent_size)
        self.class_token = nn.Parameter(randn(1, 1, latent_size))
        self.position_embeddings = nn.Parameter(randn(1, num_patches+1, latent_size))

    def forward(self, data):
        # Split images into patches and flatten into
        # batch_size x num_patches x (num_channels * patch_size * patch_size)
        data = self.unfold(data).permute(0, 2, 1).contiguous()
        data = self.linear(data)
        output = cat(self.class_token, data, dim=1) + self.position_embeddings
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

    def forward(self, data):
        pass


class MLP(nn.Module):
    """Configurable multilayer perceptron.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_sizes (list of int, optional): Sizes of hidden layers. Default: `None`.
        dropout (float or list of float, optional): Dropout probabilities of each
            hidden layer. If `None`, no dropout will be used. If single float, the
            same dropout probability will be used for all hidden layers.
            Default: `None`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        dropout: Union[float, List[float]] = None,
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
                modules.append(nn.ReLU())
                if len(dropout_modules) > i:
                    modules.append(dropout_modules[i])
        self.linear_relu_stack = nn.Sequential(*modules)

    def forward(self, data: Tensor) -> Tensor:
        data = self.flatten(data)
        output = self.linear_relu_stack(data)
        return output

    def get_hyperparameters(self) -> Dict:
        """Collects all hyperparameters of the object.

        The return dictionary can be used as keyword arguments to initialize a new
        object with the same hyperparameters.

        Returns:
            dict: Hyperparameters including `'input_size'`, `'output_size'`,
                `'hidden_sizes'` and `'dropout'`.
        """
        input_size = self.linear_relu_stack[0].in_features
        output_size = self.linear_relu_stack[-1].out_features
        hidden_sizes = []
        dropout = []
        for module in self.linear_relu_stack[:-1]:
            if isinstance(module, nn.Linear):
                hidden_sizes.append(module.out_features)
            if isinstance(module, nn.Dropout):
                dropout.append(module.p)
        hyperparameters = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
        }
        return hyperparameters
