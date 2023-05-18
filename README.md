# Pytorch Inspector
PyTorch library for inspecting the dynamics of a deep neural network model.

## Usage

### Install

torch, torchvision, torch audio should be installed separately.

```console
pip3 install .
```

### Example tasks

```console
python test/test_fork.py
python test/test_spawn.py
```

### Uninstall 

```console
pip3 uninstall pytorch-inspector
```

## Documentation

The documentation can be created offline with:

```
pip3 install pdoc
pdoc -t doc/custom_module pytorch_inspector --math -o doc # to output doc in doc/
open doc/pytorch_inspector.html
```