import argparse
import os
from tempfile import TemporaryDirectory

import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from .train import make_mnist_model_config, train_mnist
from .utils import save_model


def objective(
    config: dict[str, str | int | float], data_dir: str | os.PathLike
) -> None:
    """Objective function for hyperparameter tuning.

    Trains a model on MNIST according to the configuration and reports the mean loss.
    Also saves checkpoints to `'checkpoint.pt'` files.  Checkpoints contain model,
    optimizer and learning rate scheduler states as well as `'epoch'` metadata.

    Args:
        config (dict): Training configuration including `'batch_size'`, `'num_epochs'`,
            `'lr'`, `'weight_decay'`, `'epoch_lr_restart'`, `'patch_size'`,
            `'num_heads'`, `'latent_size_multiplier'`, `'num_layers'`, `'encoder_size'`,
            `'head_size'`, `'dropout'`, `'encoder_activation'` and `'head_activation'`.
        data_dir (str or os.PathLike): Directory of the MNIST training data.
    """

    # Define callback function for checkpoint saving
    def report_fn(
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        metrics = {"mean_loss": val_loss}
        with TemporaryDirectory() as temp_dir:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                os.path.join(temp_dir, "checkpoint.pt"),
            )
            metadata = {"epoch": epoch}
            checkpoint = train.Checkpoint.from_directory(temp_dir)
            checkpoint.set_metadata(metadata)
            train.report(metrics=metrics, checkpoint=checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resume checkpoint if available
    if train.get_checkpoint():
        checkpoint = train.get_checkpoint()
        with checkpoint.as_directory() as checkpoint_dir:
            resume_states = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            resume_states["epoch"] = checkpoint.get_metadata()["epoch"]
    else:
        resume_states = None

    train_mnist(
        config=config,
        data_dir=data_dir,
        report_fn=report_fn,
        resume_states=resume_states,
        device=device,
    )


def fit(
    exp_name: str,
    storage_path: str | os.PathLike,
    num_samples: int,
    num_epochs: int,
    model_dir: str | os.PathLike,
    resources: dict[str, float] = None,
) -> None:
    """Tunes hyperparameters of a model to MNIST.

    Selects the checkpoint with the best validation performance and prints the best
    result and the best checkpoint metadata.  The best model is then saved to the
    provided model directory.

    Args:
        exp_name (str): Name of the experiment.
        storage_path (str or os.PathLike): Path of the experiment directory.
        num_samples (int): The number of hyperparameter configurations to try.
        num_epochs (int): The number of epochs per optimization.
        model_dir (str or os.PathLike): Directory to save the best model to.
        resources (dict, optional): Resource configuration per trial.  Default: `None`.
    """
    search_space = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "num_epochs": num_epochs,
        "lr": tune.loguniform(1e-5, 0.01),
        "weight_decay": tune.loguniform(1e-4, 0.1),
        "epoch_lr_restart": tune.choice([4, 8, 16, 32, 64]),
        "patch_size": tune.choice([2, 4, 7, 14]),
        "num_heads": tune.choice([2, 4, 8, 16]),
        "latent_size_multiplier": tune.choice([4, 8, 16, 32]),
        "num_layers": tune.choice([1, 2, 4, 8]),
        "encoder_size": tune.choice([2**i for i in range(4, 10)]),
        "head_size": tune.choice([2**i for i in range(4, 10)]),
        "dropout": tune.uniform(0, 0.5),
        "encoder_activation": "gelu",
        "head_activation": "gelu",
    }
    data_dir = os.path.abspath("data")
    trainable = tune.with_parameters(objective, data_dir=data_dir)
    metric, mode = "mean_loss", "min"
    if resources is not None:
        trainable = tune.with_resources(trainable, resources=resources)
    storage_path = os.path.abspath(storage_path)
    exp_path = os.path.join(storage_path, exp_name)
    if tune.Tuner.can_restore(exp_path):
        tuner = tune.Tuner.restore(
            exp_path,
            trainable=trainable,
            resume_errored=True,
            restart_errored=False,
            resume_unfinished=True,
            param_space=search_space,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            run_config=train.RunConfig(
                name=exp_name,
                storage_path=storage_path,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_score_attribute=metric,
                    checkpoint_score_order=mode,
                    num_to_keep=5,
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                search_alg=OptunaSearch(),
                scheduler=ASHAScheduler(),
                metric=metric,
                mode=mode,
            ),
            param_space=search_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result(scope="all")
    best_checkpoint = best_result.get_best_checkpoint(
        metric=metric,
        mode=mode,
    )
    print("Best result config: ", best_result.config)
    print("Best checkpoint: ", best_checkpoint.get_metadata())
    with best_checkpoint.as_directory() as checkpoint_dir:
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "checkpoint.pt"),
            map_location=torch.device("cpu"),
        )["model"]
    config = make_mnist_model_config(best_result.config)
    save_model(config, state_dict, model_dir)


def main() -> None:
    """Processes command line arguments with tuning."""
    parser = argparse.ArgumentParser(description="MNIST Vision Transformer Tuning")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="mnistvit",
        metavar="NAME",
        help="name of the experiment to run (default: 'mnistvit')",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="ray_results",
        metavar="PATH",
        help="path of the experiment directory (default: 'ray_results')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        metavar="N",
        help="number of configurations to test (default: 1024)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=64,
        metavar="N",
        help="number of epochs to train (default: 64)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        metavar="PATH",
        help="directory to save the best model to (default: '.')",
    )
    parser.add_argument(
        "--cpu-resource",
        type=float,
        default=None,
        metavar="R",
        help="CPU resource per trial (default: None)",
    )
    parser.add_argument(
        "--gpu-resource",
        type=float,
        default=None,
        metavar="R",
        help="GPU resource per trial (default: None)",
    )
    args = parser.parse_args()
    if args.cpu_resource is None and args.gpu_resource is None:
        resources = None
    else:
        resources = {}
        if args.cpu_resource is not None:
            resources["cpu"] = args.cpu_resource
        if args.gpu_resource is not None:
            resources["gpu"] = args.gpu_resource
    fit(
        exp_name=args.exp_name,
        storage_path=os.path.abspath(args.storage_path),
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        model_dir=os.path.abspath(args.model_dir),
        resources=resources,
    )


if __name__ == "__main__":
    main()
