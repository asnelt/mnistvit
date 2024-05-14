# Python package mnistvit

A PyTorch-only implementation of a vision transformer (ViT) for training on MNIST,
achieving greater than 99.1% test accuracy with default parameters and without any data
augmentation or pre-training, and greater than 99.6% accuracy with data augmentation.
The ViT architecture and learning parameters can be configured easily.  Code for
hyperparameter optimization is provided as well.


## Requirements

The package requires Python 3.10 or greater and additionally requires the `torch` and
`torchvision` packages.  The ViT itself requires `torch` only.  For hyperparameter
optimization, additionally `ray[tune]` and `optuna` are required.


## Installation

To install the mnistvit package, run the following command in the parent directory of
the repository:

```
pip install mnistvit
```


## Usage

To train a model with default parameters:
```
python -m mnistvit.train
```

The script will produce a file `model.pt` containing the trained model.  Use the `-h`
argument for a list of options.

To evaluate the test set accuracy of the model stored in`model.pt`:
```
python -m mnistvit.predict --use-accuracy
```

To predict the class of the digit stored in the file `sample.jpg`:
```
python -m mnistvit.predict --image-file sample.jpg
```

For hyperparameter optimization with default search parameters:
```
python -m mnistvit.tune
```


## License

mnistvit is released under the GPLv3 license, as found in the [LICENSE](LICENSE) file.
