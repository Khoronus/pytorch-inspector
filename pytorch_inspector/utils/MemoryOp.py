import torch

class MemoryOp:
    """
    Operation on memory. 
    """
    @staticmethod
    def assignTo(tensor : torch.Tensor) -> torch.Tensor:
        """
        Args:
        - **tensor**: Tensor object
        Check if the tensor can be assigned with the current device
        or if is necessary to pass as CPU. 
        Return
        - Return the tensor to assign with the original device or CPU.
        """
        if torch.cuda.is_available():
            device = tensor.device
            if tensor.element_size() * tensor.nelement() < torch.cuda.memory_allocated(device):
                return tensor
            else:
                return tensor.cpu()
        else:
            return tensor.cpu()
    
