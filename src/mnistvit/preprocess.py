from os import PathLike

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.io import ImageReadMode, read_image


def preprocess_mnist(
    data_dir: str | PathLike, train: bool, use_augmentation: bool = False
) -> torch.utils.data.dataset.Dataset:
    """Preprocesses the MNIST dataset.

    Normalizes the MNIST dataset with training mean and variance and eventually
    augments the dataset with random affine transformations.

    Args:
        data_dir (str or os.PathLike): Directory of the MNIST dataset.
        train (bool): If true, loads the training set, else the test set.
        use_augmentation (bool, optional): If true, augments the dataset with random
            affine transformations.  Default: `False`.

    Returns:
        torch.utils.data.dataset.Dataset: Image, target pairs.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if use_augmentation:
        # Data augmentation
        transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=16, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transform,
            ]
        )
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    return dataset


def get_train_loaders_mnist(
    data_dir: str | PathLike,
    batch_size: int,
    train_fraction: float = 1.0,
    use_augmentation: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Training loaders of the MNIST dataset.

    Args:
        data_dir (str or os.PathLike): Directory of the MNIST dataset.
        batch_size (int): Size of the batches of the training loaders.
        train_fraction (float, optional): Fraction of the set used for the training
            loader.  The remainder is used for the validation loader.  Default: 1.0.
        use_augmentation (bool, optional): If true, augments the dataset with random
            affine transformations.  Default: `True`.

    Returns:
        tuple: Training loader and validation loader, where the latter is `None` if
            `train_fraction` is 1.
    """
    assert 0 <= train_fraction <= 1
    loader_kwargs = {"shuffle": True, "batch_size": batch_size}
    dataset = preprocess_mnist(data_dir, train=True, use_augmentation=use_augmentation)
    if train_fraction == 1.0:
        train_loader = DataLoader(dataset, **loader_kwargs)
        val_loader = None
    else:
        num_train = int(len(dataset) * train_fraction)
        train_set, val_set = random_split(
            dataset, [num_train, len(dataset) - num_train]
        )
        train_loader = DataLoader(train_set, **loader_kwargs)
        val_loader = DataLoader(val_set, **loader_kwargs)
    return train_loader, val_loader


def get_test_loader_mnist(data_dir: str | PathLike, batch_size: int) -> DataLoader:
    """Test loader of the MNIST dataset.

    Args:
        data_dir (str or os.PathLike): Directory of the MNIST dataset.
        batch_size (int): Size of the batches of the test loader.

    Returns:
        torch.utils.data.DataLoader: Test loader.
    """
    dataset = preprocess_mnist(data_dir, train=False)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return loader


def read_digit_image(image_file: str | PathLike) -> torch.FloatTensor:
    """Loads a single digit image from a file.

    Center crops and resizes the image to 28 by 28 pixels.  Also inverts the image if
    there are more bright than dark pixels.

    Args:
        image_file (str or os.PathLike): The image file.

    Returns:
        torch.FloatTensor: The preprocessed digit image.
    """
    image = read_image(image_file, mode=ImageReadMode.GRAY)
    transform = transforms.Compose(
        [
            transforms.CenterCrop(min(image.size()[1:])),
            transforms.Resize([28, 28]),
        ]
    )
    # Standardize image
    image = transform(image).type(torch.FloatTensor)
    image = transforms.Normalize(image.mean(), image.std())(image)
    # Check if we need to invert the image
    if (image > 0).count_nonzero() > image.numel() / 2:
        # More bright than dark pixels
        image = transforms.functional.invert(image)
    return image
