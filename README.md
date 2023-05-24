# PyTorch Inspector
PyTorch library for inspecting the dynamics of a deep neural network model.
The library handles tensors by hooking forward/backward propagation and passes the tensors to separate processes for further analysis.  
The main goal of this library is to collect discrete information about the tensors of a model.

---

## Install Pytorch-Inspector

torch, torchvision, torch audio should be installed separately.

```console
pip3 install .
```

### Uninstall 

```console
pip3 uninstall pytorch-inspector
```

### Documentation

The documentation can be created offline with:

```
pip3 install pdoc
pdoc -t doc/custom_module pytorch_inspector --math -o doc # to output doc in doc/
open doc/pytorch_inspector.html
```

---

## How to use

The class is instantiated once and any invocation in the code refers to the first instance.

```python
# main.py
# ! pip install torchvision
import torch
from torchvision import models
from pytorch_inspector import ParrallelHandler, DataRecorder

def main():

    model = models.alexnet()

    #----------------------------------
    # Step 1: Define how to process the passed data
    #----------------------------------
    # Callback functions are used to process data passed from the main process to child processes.
    # DataRecorder is an example
    dr = DataRecorder((640,480), 20., 100, 'output')

    #----------------------------------
    # Step 2: Define a ParrallelHandler
    #----------------------------------
    # A ParallelHandler manages all child processes.
    ph = ParrallelHandler(callback_onrun=dr.tensor_plot2D, callback_onclosing=dr.flush, frequency=20.0, timeout=30.0)

    #----------------------------------
    # Step 3: Attach layer/layers or full model
    #----------------------------------
    ph.track_model(-1, {'model': model})
    ph.track_layer(-1, {'classifier0_': model.classifier[0], 'classifier3_': model.classifier[3], 'classifier6_': model.classifier[6]})
```

Furter calls in any part of the program can be done as follow:
```console
    ph = ParrallelHandler()
```

**Note**: If the multiprocess start mode is fork, please create the processes before initializing CUDA or an exception is raised.

**Additional Note**: A method to get all the valid handles is shown in **test/hook/test_automatic_hook.py** or in the examples.

---
## Example 
Folder with examples about how to use the library.

---

## Test 
Folder with basic examples to test the correct behavior of the library.

---

## Output

The default data recorder will create a video of the tensors tracked. The current code creates plot for 2D tensors and 3D tensors in the form of [1xWxH]. Other shapes are converted in histograms.  
<div style="display:flex">
  <div style="flex: 1; padding-left: 10px;">
    <img src="images/Plot.gif" alt="Plot Example" witdh="100%" height="100%"/>
  </div>
  <div style="flex: 1; padding-right: 10px;">
    <img src="images/Histogram.gif" alt="Histogram Example" witdh="100%" height="100%"/>
  </div>
</div>
 
---

## Design Notes

Tensors are passed to child process as CPU, clone, and detached to reduce the use of GPU memory.
It introduces some time delay, but it should be compensated by decreasing the frequency in which the data is pushed to queue.  

### Hint
To run multiple processes on part of the same model: split the list of the layers.

---
## Program Interruption

Interrupting the main process may create corrupted videos (please check cv2.WriteVideo for more information).  
It may also keep child process alive if created as spawn. Please set timeout greater than 0.