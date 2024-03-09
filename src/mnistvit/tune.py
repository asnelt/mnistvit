import os
import argparse
import torch
from typing import Dict
from tempfile import TemporaryDirectory
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from .model import VisionTransformer
from .train import train_mnist
from .utils import save_model, load_model


def objective(config: Dict, data_dir: str) -> None:
    """Objective function of the hyperparameter tuning.

    Trains a vision transformer on MNIST according to the configuration and reports the
    mean loss. Also saves checkpoints to `'checkpoint.pt'` files. Checkpoints contain
    saved models as well as `'epoch'` metadata.

    Args:
        config (dict): Training configuration including `'batch_size'`, `'num_epochs'`,
            `'lr'`, `'weight_decay'`, `'patch_size'`, `'latent_size'`, `'num_heads'`,
            `'num_layers'`, `'encoder_size'`, `'head_size'` and `'dropout'`.
        data_dir (str): Directory of the MNIST training data.
    """

    def report_fn(epoch: int, val_loss: float, model: VisionTransformer):
        metrics = {"mean_loss": val_loss}
        with TemporaryDirectory() as temp_dir:
            save_model(model, os.path.join(temp_dir, "checkpoint.pt"))
            metadata = {"epoch": epoch}
            checkpoint = train.Checkpoint.from_directory(temp_dir)
            checkpoint.set_metadata(metadata)
            train.report(metrics=metrics, checkpoint=checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mnist(config, data_dir=data_dir, device=device, report_fn=report_fn)


def fit(num_samples: int, num_epochs: int, resources: Dict = None) -> None:
    """Tunes hyperparameters of a vision transformer to MNIST.

    Selects the checkpoint with the best validation performance and prints the best
    result and the best checkpoint metadata. The best model is then saved to the
    default model file name. Keeps one checkpoint per trial.

    Args:
        num_samples (int): The number of hyperparameter configurations to try.
        num_epochs (int): The number of epochs per optimization.
        resources (dict, optional): Resource configuration per trial.
    """
    search_space = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "num_epochs": num_epochs,
        "lr": tune.loguniform(1e-5, 0.01),
        "weight_decay": tune.loguniform(1e-4, 0.1),
        "patch_size": tune.choice([2, 4, 7, 14]),
        "latent_size": tune.choice([2**i for i in range(4, 10)]),
        "num_heads": tune.choice([2, 4, 8, 16]),
        "num_layers": tune.choice([1, 2, 4, 8]),
        "encoder_size": tune.choice([2**i for i in range(4, 10)]),
        "head_size": tune.choice([2**i for i in range(4, 10)]),
        "dropout": tune.uniform(0, 0.5),
    }
    data_dir = os.path.abspath("data")
    trainable = tune.with_parameters(objective, data_dir=data_dir)
    metric, mode = "mean_loss", "min"
    if resources is not None:
        trainable = tune.with_resources(trainable, resources=resources)
    tuner = tune.Tuner(
        trainable,
        run_config=train.RunConfig(
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
                num_to_keep=1,
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
    with best_checkpoint.as_directory() as checkpoint_dir:
        model = load_model(os.path.join(checkpoint_dir, "checkpoint.pt"))
    print("Best result config: ", best_result.config)
    print("Best checkpoint: ", best_checkpoint.get_metadata())
    save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Vision Transformer Tuning")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        metavar="N",
        help="number of configs to test (default: 1024)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=64,
        metavar="N",
        help="number of epochs to train (default: 64)",
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
    fit(args.num_samples, args.num_epochs, resources)
