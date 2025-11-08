# Python package mnistvit

A PyTorch-only implementation of a vision transformer (ViT) for training on MNIST,
achieving 99.65% test accuracy with default parameters and without pre-training.  The
ViT architecture and learning parameters can be configured easily.  Code for
hyperparameter optimization is provided as well.

The code is intended to be used for learning about vision transformers.  With MNIST as a
simple and well understood dataset, the importance of various hyperparameters can be
explored.


## Requirements

The package requires Python 3.10 or greater and additionally requires the `torch` and
`torchvision` packages.  For hyperparameter optimization, additionally `ray[tune]` and
`optuna` are required.  The ViT itself requires `torch` only.


## Installation

To install the mnistvit package, run the following command in the parent directory of
the repository:
```
pip install mnistvit
```

To install the package with hyperparameter optimization support:
```
pip install mnistvit[tune]
```


## Usage

To train a model with default parameters:
```
python -m mnistvit.train
```

The script will produce a file `config.json` with the model configuration and file
`model.pt` containing the trained model.  Use the `-h` argument for a list of options.

To evaluate the test set accuracy of the model stored in `model.pt` with the
configuration in `config.json`:
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

A trained model is available on [Hugging Face](https://huggingface.co/asnelt/mnistvit/).


## License

mnistvit is released under the GPLv3 license, as found in the [LICENSE](LICENSE) file.
