import argparse
from typing import Dict

import torch

from .preprocess import read_digit_image, test_loader_mnist
from .utils import FILE_LIKE, load_model


def test_mnist(
    config: Dict,
    data_dir: str,
    use_loss: bool = True,
    use_accuracy: bool = True,
    device: torch.device = "cpu",
) -> None:
    """Loads a model, tests it on MNIST and prints the results.

    Args:
        config (dict): Test configuration with `'batch_size'`.
        data_dir (str): Directory of the MNIST dataset.
        use_loss (bool, optional): If true, evaluates the loss on the test set.
            Default: `True`.
        use_accuracy (bool, optional): If true, evaluates the accuracy on the test set.
            Default: `True`.
        device (torch.device, optional): Device to evaluate the model on.
            Default: `'cpu'`.
    """
    model = load_model(device=device)
    test_loader = test_loader_mnist(data_dir, config["batch_size"])
    if use_loss:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = prediction_loss(model, test_loader, loss_fn, device)
        print("Test loss: ", loss)
    if use_accuracy:
        acc = prediction_accuracy(model, test_loader, device)
        print("Test accuracy: ", acc)


def predict_file(file: FILE_LIKE, device: torch.device = "cpu") -> int:
    """Loads a model and classifies a digit from an image file.

    Args:
        file (FILE_LIKE): The image file.
        device (torch.device, optional): Device to evaluate the model on.
            Default: `'cpu'`.

    Returns:
        int: Predicted class label.
    """
    model = load_model(device=device)
    image = read_digit_image(file)
    predicted = predict_single_image(model, image, device)
    return predicted


def prediction_loss(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device = "cpu",
) -> float:
    """Evaluates the model loss on the data loader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation.
        loss_fn (torch.nn.Module): Loss function for evaluation.
        device (torch.device, optional): Device to evaluate the model on.
            Default: `'cpu'`.

    Returns:
        float: Calculated loss.
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_fn(output, target).cpu().numpy()
    loss /= len(data_loader)
    return loss


def prediction_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = "cpu",
) -> float:
    """Evaluates the model accuracy on the data loader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation.
        device (torch.device, optional): Device to evaluate the model on.
            Default: `'cpu'`.

    Returns:
        float: Calculated accuracy.
    """
    correct = 0
    total = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        predicted = classify(model, data)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return correct / total


def predict_single_image(
    model: torch.nn.Module, image: torch.FloatTensor, device: torch.device = "cpu"
) -> int:
    """Uses model to classify a digit image.

    Args:
        model (torch.nn.Module): Model to use for classification.
        image (torch.FloatTensor): Preprocessed digit image.
        device (torch.device, optional): Device to evaluate the model on.
            Default: `'cpu'`.

    Returns:
        int: Predicted class label.
    """
    data = image.unsqueeze(0)  # Add batch dimension
    data = data.to(device)
    predicted = classify(model, data).cpu().item()
    return predicted


def classify(model: torch.nn.Module, data: torch.utils.data.Sampler) -> torch.Tensor:
    """Uses model to classify given data.

    Args:
        model (torch.nn.Module): Model to use for classification.
        data (torch.utils.data.Sampler): Data sampler.

    Returns:
        torch.Tensor: Predicted class labels.
    """
    output = eval_output(model, data)
    predicted = torch.max(output.data, dim=1)[1]
    return predicted


def class_log_probs(
    model: torch.nn.Module, data: torch.utils.data.Sampler
) -> torch.Tensor:
    """Evaluates model log probabilities of all classes on given data.

    Args:
        model (torch.nn.Module): Model to use for evaluation.
        data (torch.utils.data.Sampler): Data sampler.

    Returns:
        torch.Tensor: Log probabilities of classes.
    """
    output = eval_output(model, data)
    log_probs = torch.nn.functional.log_softmax(output.data, dim=1)
    return log_probs


def eval_output(model: torch.nn.Module, data: torch.utils.data.Sampler) -> torch.Tensor:
    """Evaluates the output of a model on given data.

    Args:
        model (torch.nn.Module): Model to use for evaluation.
        data (torch.utils.data.Sampler): Data sampler.

    Returns:
        torch.Tensor: Model output.
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Prediction")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="FILE",
        help="image file to predict (default: None)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 32)",
    )
    parser.add_argument(
        "--use-loss",
        action="store_true",
        default=False,
        help="enables test loss calculation",
    )
    parser.add_argument(
        "--use-accuracy",
        action="store_true",
        default=False,
        help="enables test accuracy calculation",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=True, help="disables CUDA testing"
    )
    args = parser.parse_args()
    no_cuda = args.no_cuda or not torch.cuda.is_available()
    device = torch.device("cpu" if no_cuda else "cuda")
    config = {
        "batch_size": args.batch_size,
    }
    if args.file is not None:
        predicted = predict_file(args.file, device)
        print(predicted)
    test_mnist(
        config,
        data_dir="data",
        use_loss=args.use_loss,
        use_accuracy=args.use_accuracy,
        device=device,
    )
