# Example

The folder contains some basic examples about how to use the library.

### 01_model_alexnet

The example trains an AlexNet model with CIFAR10 data. The ParallelHandler create the result in the folder output.

All the possible layers that allow hook backpropagation are automatically detected and tracked.

```
python example/01_model_alexnet.py
```
