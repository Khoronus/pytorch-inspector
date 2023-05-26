# Test

The folder contains test code to check the correct behavior of the library.  
Some basic examples about PyTorch hook (hook folder) and layers (layers folder).

## Communication

Examples of data passed from the main process to children process.
No backpropagation tested.
 
```
python test/communication/test_communication_fork.py
python test/communication/test_communication_spawn.py
```

## Hook / Layers

Basic examples to test how to use the hook for the forward and backward propagation.  
The example test_lightning_mnist.py contains the code to get the fork or spawn context. In the example lightning module causes a deadlock if child process are created with fork.  

```
python test/hook/test_lightning_mnist.py
```


### test_automatic_hook

The example get all the forward and backward propagation hooks that do not cause exception.

```
python test/hook/test_automatic_hook.py
```
