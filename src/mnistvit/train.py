import argparse
import torch
from typing import Dict, Callable
from .preprocess import train_loaders_mnist
from .model import VisionTransformer
from .utils import save_model
from .predict import prediction_loss


def train_mnist(
    config: Dict,
    data_dir: str,
    use_validation: bool = True,
    report_fn: Callable = None,
    device: torch.device = "cpu",
) -> None:
    """Trains a single vision transformer on MNIST and saves the model.

    Args:
        config (dict): Training configuration including `'batch_size'`, `'num_epochs'`,
            `'lr'`, `'weight_decay'`, `'patch_size'`, `'latent_size'`, `'num_heads'`,
            `'num_layers'`, `'encoder_size'`, `'head_size'` and `'dropout'`.
        data_dir (str): Directory of the MNIST dataset.
        use_validation (bool, optional): If true, sets aside a validation set from the
            training set, else uses all training samples for training. Default: `True`.
        report_fn (callable, optional): A function for reporting the training state.
            The function must accept arguments for epoch number (`int`),
            validation loss (`float`) and model (`mnistvit.model.VisionTransformer`).
            Default: `None`.
        device (torch.device, optional): Device to train the model on. Default: `'cpu'`.
    """
    train_fraction = 0.8 if use_validation else 1.0
    train_loader, val_loader = train_loaders_mnist(
        data_dir,
        config["batch_size"],
        train_fraction,
    )
    model = VisionTransformer(
        num_channels=1,
        input_sizes=[28, 28],
        output_size=10,
        patch_size=config["patch_size"],
        latent_size=config["latent_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        encoder_size=config["encoder_size"],
        head_size=config["head_size"],
        dropout=config["dropout"],
    )
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(
        model,
        train_loader,
        loss_fn,
        config["num_epochs"],
        config["lr"],
        config["weight_decay"],
        val_loader,
        report_fn,
        device,
    )
    save_model(model)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    val_loader: torch.utils.data.DataLoader = None,
    report_fn: Callable = None,
    device: torch.device = "cpu",
) -> None:
    """Main training function for model training.

    Initializes an optimizer, contains the loop over epochs and eventually evaluates
    validation performance.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        loss_fn (torch.nn.Module): Loss function for model training.
        num_epochs (int): The number of epochs to use.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient.
        val_loader (torch.utils.data.DataLoader, optional): Validation data loader.
            Default: `None`.
        report_fn (callable, optional): A function for reporting the training state.
            The function must accept arguments for epoch number, validation loss and
            model. Default: `None`.
        device (torch.device, optional): Device to train the model on. Default: `'cpu'`.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, loss_fn, optimizer, device)
        if val_loader is not None:
            val_loss = prediction_loss(model, val_loader, loss_fn, device)
            if report_fn is not None:
                report_fn(epoch, val_loss, model)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = "cpu",
) -> None:
    """Training function for one training epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        loss_fn (torch.nn.Module): Loss function for model training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device, optional): Device to train the model on. Default: `'cpu'`.
    """
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
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
        default=8,
        metavar="N",
        help="number of epochs to train (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="R",
        help="learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        metavar="R",
        help="weight decay coefficient (default: 0)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        metavar="P",
        help="single dimension size of an image patch (default: 4)",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=64,
        metavar="D",
        help="latent size of the vision transformer (default: 64)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        metavar="N",
        help="number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        metavar="L",
        help="number of encoder blocks (default: 4)",
    )
    parser.add_argument(
        "--encoder-size",
        type=int,
        default=128,
        metavar="H",
        help="number of hidden units of transformer encoder MLPs (default: 128)",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=256,
        metavar="H",
        help="number of hidden units of MLP head (default: 256)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        metavar="R",
        help="dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        default=False,
        help="enables validation set",
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
        "patch_size": args.patch_size,
        "latent_size": args.latent_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "encoder_size": args.encoder_size,
        "head_size": args.head_size,
        "dropout": args.dropout,
    }
    train_mnist(
        config, data_dir="data", use_validation=args.use_validation, device=device
    )
