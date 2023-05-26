# Example

The folder contains some basic examples about how to use the library.

### 01_model_alexnet

The example trains an AlexNet model with CIFAR10 data. The ParallelHandler create the result in the folder output.

All the possible layers that allow hook backpropagation are automatically detected and tracked.

```
python example/01_model_alexnet.py
```

### 02_lightning_mnist

The example trains a simple classifier for MNIST dataset. The minimal ParallelHandler create the result in the folder output.

A deadlock in the child process occurres if the process are created with fork method (CUDA issues with GPU).

```
python example/02_lightning_mnist.py
```
