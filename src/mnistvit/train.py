import argparse
import os
from typing import Any, Callable, Mapping, TypedDict

import torch

from .model import EncoderActivation, MLPActivation, VisionTransformer
from .predict import prediction_loss
from .preprocess import get_train_loaders_mnist
from .utils import get_device, save_model


class TrainConfigDict(TypedDict):
    """Represents a configuration with all training parameters."""

    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    epoch_lr_restart: int
    patch_size: int
    num_heads: int
    latent_size_multiplier: int
    num_layers: int
    encoder_size: int
    head_size: int
    dropout: float
    encoder_activation: EncoderActivation
    head_activation: MLPActivation


class ResumeStatesDict(TypedDict):
    """Represents a complete state to resume training."""

    epoch: int
    model: Mapping[str, Any]
    optimizer: dict[str, Any]
    lr_scheduler: dict[str, Any]


def train_mnist(
    config: TrainConfigDict,
    data_dir: str | os.PathLike,
    use_validation: bool = True,
    use_augmentation: bool = True,
    model_dir: str | os.PathLike | None = None,
    report_fn: (
        Callable[
            [
                int,
                float,
                float | None,
                torch.nn.Module,
                torch.optim.Optimizer,
                torch.optim.lr_scheduler.LRScheduler,
            ],
            None,
        ]
        | None
    ) = None,
    resume_states: ResumeStatesDict | None = None,
    device: str | torch.device = "cpu",
) -> None:
    """Train a single model on MNIST and eventually save the model.

    Args:
        config (TrainConfigDict): Training configuration.
        data_dir (str or os.PathLike): Directory of the MNIST dataset.
        use_validation (bool, optional): If true, sets aside a validation set from the
            training set, else uses all training samples for training.  Default: `True`.
        use_augmentation (bool, optional): If true, augments the training dataset with
            random affine transformations.  Default: `True`.
        model_dir (str or os.PathLike or None, optional): Directory to save the model
            to.  If `None` then the model is not saved.  Default: `None`.
        report_fn (Callable or None, optional): A function for reporting the training
            state.  The function must accept arguments for epoch number (`int`),
            training loss (`float`), validation loss (`float` or `None`),
            model (`torch.nn.Module`), optimizer (`torch.optim.Optimizer`) and
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`).  Default: `None`.
        resume_states (ResumeStatesDict or None, optional): Dictionary with states to
            resume.  Default: `None`.
        device (str or torch.device, optional): Device to train the model on.
            Default: `'cpu'`.
    """
    train_fraction = 0.8 if use_validation else 1.0
    train_loader, val_loader = get_train_loaders_mnist(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        train_fraction=train_fraction,
        use_augmentation=use_augmentation,
    )
    model_config = make_mnist_model_config(config)
    model = VisionTransformer(**model_config)
    model = model.to(device, dtype=torch.bfloat16)
    if resume_states is not None:
        model.load_state_dict(resume_states["model"])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        epoch_lr_restart=config["epoch_lr_restart"],
        val_loader=val_loader,
        report_fn=report_fn,
        resume_states=resume_states,
        device=device,
    )
    if model_dir is not None:
        save_model(model_config, model.state_dict(), model_dir)


def make_mnist_model_config(train_config: TrainConfigDict) -> dict[str, Any]:
    """Make configuration for initializing a vision transformer for training on MNIST.

    Takes a training configuration and makes a configuration that can be used to
    initialize a vision transformer for training on MNIST.

    Args:
        config (TrainConfigDict): Training configuration with model hyperparameters.

    Returns:
        model_config (dict): The vision transformer initialization configuration.
    """
    model_config = {
        "num_channels": 1,
        "input_sizes": [28, 28],
        "output_size": 10,
        "patch_size": train_config["patch_size"],
        "num_heads": train_config["num_heads"],
        "latent_size_multiplier": train_config["latent_size_multiplier"],
        "num_layers": train_config["num_layers"],
        "encoder_size": train_config["encoder_size"],
        "head_size": train_config["head_size"],
        "dropout": train_config["dropout"],
        "encoder_activation": train_config["encoder_activation"],
        "head_activation": train_config["head_activation"],
    }
    return model_config


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    epoch_lr_restart: int,
    val_loader: torch.utils.data.DataLoader | None = None,
    report_fn: (
        Callable[
            [
                int,
                float,
                float | None,
                torch.nn.Module,
                torch.optim.Optimizer,
                torch.optim.lr_scheduler.LRScheduler,
            ],
            None,
        ]
        | None
    ) = None,
    resume_states: ResumeStatesDict | None = None,
    device: str | torch.device = "cpu",
) -> None:
    """Train model.

    Initializes an optimizer and learning rate scheduler, contains the loop over epochs
    and eventually evaluates validation performance.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        loss_fn (torch.nn.Module): Loss function for model training.
        num_epochs (int): The number of epochs to use.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient.
        epoch_lr_restart (int): Epoch for first learning rate scheduler restart.
        val_loader (torch.utils.data.DataLoader or None, optional): Validation data
            loader.  Default: `None`.
        report_fn (Callable or None, optional): A function for reporting the training
            state.  The function must accept arguments for epoch number (`int`),
            training loss (`float`), validation loss (`float` or `None`),
            model (`torch.nn.Module`), optimizer (`torch.optim.Optimizer`) and
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`).  Default: `None`.
        resume_states (dict or None, optional): Dictionary with states to resume.
            Default: `None`.
        device (str or torch.device, optional): Device to train the model on.
            Default: `'cpu'`.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=epoch_lr_restart,
    )
    if resume_states is None:
        start_epoch = 0
    else:
        start_epoch = resume_states["epoch"] + 1
        optimizer.load_state_dict(resume_states["optimizer"])
        lr_scheduler.load_state_dict(resume_states["lr_scheduler"])
    iters = len(train_loader)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + step / iters)
        if val_loader is None:
            val_loss = None
        else:
            val_loss = prediction_loss(model, val_loader, loss_fn, device)
        if report_fn is not None:
            report_fn(epoch, loss, val_loss, model, optimizer, lr_scheduler)


def main() -> None:
    """Process command line arguments with training."""
    parser = argparse.ArgumentParser(description="MNIST Vision Transformer Training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=60,
        metavar="N",
        help="number of epochs to train (default: 60)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="R",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-2,
        metavar="R",
        help="weight decay coefficient (default: 2e-2)",
    )
    parser.add_argument(
        "--epoch-lr-restart",
        type=int,
        default=22,
        metavar="N",
        help="epoch for first scheduler restart (default: 22)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        metavar="P",
        help="single dimension size of an image patch (default: 4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=28,
        metavar="N",
        help="number of attention heads (default: 28)",
    )
    parser.add_argument(
        "--latent-size-multiplier",
        type=int,
        default=15,
        metavar="M",
        help="yields latent size when multiplied with num_heads (default: 15)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=9,
        metavar="L",
        help="number of encoder blocks (default: 9)",
    )
    parser.add_argument(
        "--encoder-size",
        type=int,
        default=615,
        metavar="H",
        help="number of hidden units of transformer encoder MLPs (default: 615)",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=88,
        metavar="H",
        help="number of hidden units of MLP head (default: 88)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=6e-2,
        metavar="R",
        help="dropout rate (default: 6e-2)",
    )
    parser.add_argument(
        "--encoder-activation",
        type=str,
        choices=["relu", "gelu"],
        default="gelu",
        help="encoder activation function (default: 'gelu')",
    )
    parser.add_argument(
        "--head-activation",
        type=str,
        choices=["relu", "gelu", "tanh"],
        default="gelu",
        help="MLP head activation function (default: 'gelu')",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        metavar="PATH",
        help="directory to save the model to (default: '.')",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        default=False,
        help="enables validation set",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        default=False,
        help="disables data augmentation",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="device for training \
              (default: None, meaning 'cuda' if available, else 'cpu')",
    )
    args = parser.parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    config: TrainConfigDict = {
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epoch_lr_restart": args.epoch_lr_restart,
        "patch_size": args.patch_size,
        "num_heads": args.num_heads,
        "latent_size_multiplier": args.latent_size_multiplier,
        "num_layers": args.num_layers,
        "encoder_size": args.encoder_size,
        "head_size": args.head_size,
        "dropout": args.dropout,
        "encoder_activation": args.encoder_activation,
        "head_activation": args.head_activation,
    }

    # Define function for reporting training progress
    def report_fn(
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        print(
            f"epoch: {epoch + 1}\t"
            f"training loss: {train_loss}\t"
            f"validation loss: {val_loss}"
        )

    use_augmentation = not args.no_augmentation
    train_mnist(
        config,
        data_dir=os.path.abspath("data"),
        use_validation=args.use_validation,
        use_augmentation=use_augmentation,
        model_dir=os.path.abspath(args.model_dir),
        report_fn=report_fn,
        device=device,
    )


if __name__ == "__main__":
    main()
