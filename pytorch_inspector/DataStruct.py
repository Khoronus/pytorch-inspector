import torch 

__all__ = ["ProcessInfoData"]

class ProcessInfoData:
    """
    Main data structure used to pass data between main process to child process.
    """
    def __init__(self, name : str, internal_message : str, 
                 message : str, shared_data : torch.Tensor):
        """
        Args:
        - **name**: Name of the data.
        - **internal_message**: message passed by the main process.
        - **message**: message to write on the image.
        - **shared_data**: shared data object (expected a tensor).
        """
        self.name = name
        self.internal_message = internal_message
        self.message = message
        self.shared_data = shared_data