from typing import Optional
import torch

class MemoryOp:
    """
    Operation on memory. 
    """
    @staticmethod
    def assignTo(tensor : torch.Tensor, same_device_only : bool) -> Optional[torch.Tensor]:
        """
        Args:
        - **tensor**: Tensor object
        - **same_device_only**: If true it checks if return a tensor only if 
        it is on the same device.
        Check if the tensor can be assigned with the current device
        or if is necessary to pass as CPU. 
        Return
        - Return the tensor to assign with the original device or CPU. 
        If same_device_only is True, it returns None if the device is 
        different.
        """
        if torch.cuda.is_available():
            device = tensor.device
            if tensor.element_size() * tensor.nelement() < torch.cuda.memory_allocated(device):
                return tensor
            else:
                if same_device_only: return None
                return tensor.cpu()
        else:
            if same_device_only: return None
            return tensor.cpu()
    
    @staticmethod
    def enoughMemory_CUDA(tensor : torch.Tensor) -> bool:
        """
        Args:
        - **tensor**: Tensor object
        Check if there is enough device memory to copy the tensor.
        Return
        - Return true if there is enough memory. False otherwise. Return true if cuda is not available.
        """
        if torch.cuda.is_available():
            device = tensor.device
            if tensor.element_size() * tensor.nelement() < torch.cuda.memory_allocated(device):
                return True
            else:
                return False
        else:
            return True
