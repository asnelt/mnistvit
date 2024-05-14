from os import PathLike
from typing import IO, BinaryIO, Union

import torch
from typing_extensions import TypeAlias

from .model import VisionTransformer

FILE_LIKE: TypeAlias = Union[str, PathLike, BinaryIO, IO[bytes]]


def save_model(model: VisionTransformer, model_file: FILE_LIKE) -> None:
    """Saves the vision transformer model, including model keyword arguments, to a file.

    Args:
        model (mnistvit.model.VisionTransformer): The model to save.
        model_file (FILE_LIKE): File name to save the model to.
    """
    model_dict = {
        "kwargs": model.kwargs,
        "state_dict": model.state_dict(),
    }
    torch.save(model_dict, model_file)


def load_model(
    model_file: FILE_LIKE, device: torch.device = "cpu"
) -> VisionTransformer:
    """Loads the model from a file.

    Args:
        model_file (FILE_LIKE): File name to load the model from.
        device (torch.device, optional): Device to load the model to.  Default: `'cpu'`.

    Returns:
        mnistvit.model.VisionTransformer: The loaded model.
    """
    model_dict = torch.load(model_file, map_location=torch.device("cpu"))
    model = VisionTransformer(**model_dict["kwargs"])
    model.load_state_dict(model_dict["state_dict"])
    model = model.to(device)
    return model
