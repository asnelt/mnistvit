import argparse
from typing import Callable, Dict

import torch

from .model import VisionTransformer
from .predict import prediction_loss
from .preprocess import get_train_loaders_mnist
from .utils import FILE_LIKE, save_model


def train_mnist(
    config: Dict,
    data_dir: str,
    use_validation: bool = True,
    use_augmentation: bool = True,
    model_file: FILE_LIKE = None,
    report_fn: Callable = None,
    resume_states: Dict = None,
    device: torch.device = "cpu",
) -> None:
    """Trains a single model on MNIST and eventually saves the model.

    Args:
        config (dict): Training configuration including `'batch_size'`, `'num_epochs'`,
            `'lr'`, `'weight_decay'`, `'epoch_lr_restart'`, `'patch_size'`,
            `'num_heads'`, `'latent_size_multiplier'`, `'num_layers'`, `'encoder_size'`,
            `'head_size'` and `'dropout'`.
        data_dir (str): Directory of the MNIST dataset.
        use_validation (bool, optional): If true, sets aside a validation set from the
            training set, else uses all training samples for training.  Default: `True`.
        use_augmentation (bool, optional): If true, augments the training dataset with
            random affine transformations.  Default: `True`.
        model_file (FILE_LIKE, optional): File name to save the model to.  If `None`
            then the model is not saved.  Default: `None`.
        report_fn (callable, optional): A function for reporting the training state.
            The function must accept arguments for epoch number (`int`),
            training loss (`float`), validation loss (`float`),
            model (`torch.nn.Module`),
            optimizer (`torch.optim.Optimizer`) and
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`).  Default: `None`.
        resume_states (dict, optional): Dictionary with states for `'epoch'`, `'model'`,
            `'optimizer'` and `'lr_scheduler'`.  Default: `None`.
        device (torch.device, optional): Device to train the model on.
            Default: `'cpu'`.
    """
    train_fraction = 0.8 if use_validation else 1.0
    train_loader, val_loader = get_train_loaders_mnist(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        train_fraction=train_fraction,
        use_augmentation=use_augmentation,
    )
    model = init_mnist_model(config, device)
    if resume_states is not None:
        model.load_state_dict(resume_states["model"])
    loss_fn = torch.nn.CrossEntropyLoss()
    train(
        model,
        train_loader,
        loss_fn,
        config["num_epochs"],
        config["lr"],
        config["weight_decay"],
        config["epoch_lr_restart"],
        val_loader,
        report_fn,
        resume_states,
        device,
    )
    if model_file is not None:
        save_model(model, model_file)


def init_mnist_model(config: Dict, device: torch.device = "cpu") -> torch.nn.Module:
    """Initializes a vision transformer for training on MNIST.

    Args:
        config (dict): Training configuration including `'patch_size'`, `'num_heads'`,
            `'latent_size_multiplier'`, `'num_layers'`, `'encoder_size'`, `'head_size'`
            and `'dropout'`.
        device (torch.device, optional): Device to load the model on.
            Default: `'cpu'`.
    Returns:
        torch.nn.Module: The initialized model.
    """
    model = VisionTransformer(
        num_channels=1,
        input_sizes=[28, 28],
        output_size=10,
        patch_size=config["patch_size"],
        num_heads=config["num_heads"],
        latent_size_multiplier=config["latent_size_multiplier"],
        num_layers=config["num_layers"],
        encoder_size=config["encoder_size"],
        head_size=config["head_size"],
        dropout=config["dropout"],
    )
    model = model.to(device)
    return model


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    epoch_lr_restart: int,
    val_loader: torch.utils.data.DataLoader = None,
    report_fn: Callable = None,
    resume_states: Dict = None,
    device: torch.device = "cpu",
) -> None:
    """Main training function for model training.

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
        val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
            Default: `None`.
        report_fn (callable, optional): A function for reporting the training state.
            The function must accept arguments for epoch number (`int`),
            training loss (`float`), validation loss (`float`),
            model (`torch.nn.Module`),
            optimizer (`torch.optim.Optimizer`) and
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`).  Default: `None`.
        resume_states (dict, optional): Dictionary with states for `'epoch'`,
            `'optimizer'` and `'lr_scheduler'`.  Default: `None`.
        device (torch.device, optional): Device to train the model on.
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
    """Processes command line arguments with training."""
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
        "--model-file",
        type=str,
        default="model.pt",
        metavar="FILE",
        help="file to save the model to (default: 'model.pt')",
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
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args()
    no_cuda = args.no_cuda or not torch.cuda.is_available()
    device = torch.device("cpu" if no_cuda else "cuda")
    torch.manual_seed(args.seed)
    config = {
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
    }

    # Define function for reporting training progress
    def report_fn(
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        print(
            f"epoch: {epoch+1}\t"
            f"training loss: {train_loss}\t"
            f"validation loss: {val_loss}"
        )

    use_augmentation = not args.no_augmentation
    train_mnist(
        config,
        data_dir="data",
        use_validation=args.use_validation,
        use_augmentation=use_augmentation,
        model_file=args.model_file,
        report_fn=report_fn,
        device=device,
    )


if __name__ == "__main__":
    main()
