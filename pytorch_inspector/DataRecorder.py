from multiprocessing import Value

import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

from pytorch_inspector.utils.DataPlot import DataPlot
from pytorch_inspector.utils.Decorators import *


__all__ = ["DataRecorder"]

class DataRecorder():
    """
    Record passed data.
    """    
    @exception_decorator
    def __init__(self, 
                 shape_expected : tuple, fps : float, maxframes : int, 
                 path_root : str, colorBGR : tuple, displayND_mode : str):
        """
        Args:
        - **shape_expected**: Expected video output size.
        - **fps**: Expected video frame rate.
        - **maxframes**: Maximum number of frames for a video.
        - **path_root**: Root to the output destination.
        - **colorBGR**: Color used for the text inside the video frames.
        - **colorBGR**: Color used for the text inside the video frames.
        - **displayND_mode**: How to display high dimensional data (> 2D) ['default','pca'].
        """
        super().__init__() # Call the parent class constructor

        self.shape_expected = shape_expected
        self.fps = fps
        self.maxframes = maxframes
        self.path_root = path_root
        self.colorBGR = colorBGR
        self.displayND_mode = displayND_mode

        # Internal counter for the number of frames added 
        self.internal_frames = {}
        # Unique internal counter with the number of videos created
        self.internal_counter = {}
        # Container with the video writer associated
        self.internal_out = {}

        from pathlib import Path
        Path(path_root).mkdir(parents=True, exist_ok=True)

    def flush(self):
        """
        Releases existing running videos.
        Note: The content may be empty if data was created in another process.
        """
        # release the video writer associated to the dictionary
        for key_dict in self.internal_out:
            if self.internal_out[key_dict] is not None:
                self.internal_out[key_dict].release()

    @exception_decorator
    def tensor_plot2D(self, 
                      unique_id : int,
                      key : str, internal_message : str, message : str, 
                      input_data : torch.Tensor) -> None:
        """
        Function called when the process starts.
        The process automatically creates a video of the format:
        key + unique_id + '_' + internal_counter + '_video.mp4'
        where the internal_counter is a progressive number starting from 0.

        The data is passed from the calling process to this process via queue.
        The tensor shape is expected to be [B x W x H] where B (batch) is equal to 1.

        Plot a tensor image and write a message on the top. The image is then added to a video.

        Note: 1D Tensors are currently not supported.

        Args:
        - **unique_id**: Unique identifier associated to this process
        - **key**: input unique identifier.
        - **internal_message**: message passed by the main process.
        - **message**: message to write on the image.
        - **input_data**: passed input to visualize.
        """
        # Check if the key was seen before, otherwise add it.
        if key not in self.internal_counter:
            self.internal_frames[key] = 0
            self.internal_counter[key] = 0
            self.internal_out[key] = None

        # Check the data type
        if isinstance(input_data, np.ndarray):
            #tensor_np = input_data
            print('EEE Wrong data type')
        #if isinstance(input_data, torch.Tensor):
        #    if input_data == 'cpu':
        #        tensor_data = input_data.detach().squeeze(0)
        #    else:
        #        tensor_data = input_data.cpu().detach().squeeze(0)

        # Try to remove the case [1xWxH]
        tensor_data = input_data.squeeze(0)

        # Plot to figure
        if tensor_data.dim() == 1:
            minval=torch.min(tensor_data)
            maxval=torch.max(tensor_data)
            hist = torch.histc(tensor_data, bins=10, min=minval, max=maxval)
            fig = DataPlot.plot_1D(hist, minval.item(), maxval.item())
        elif tensor_data.dim() == 2:
            fig = DataPlot.tensor_plot2D(tensor_data)
        elif self.displayND_mode == 'pca':
            #fig = DataPlot.plot_pca_lowrank(tensor_data)
            fig = DataPlot.plot_pca(tensor_data.clone())
        else:
            minval=torch.min(tensor_data)
            maxval=torch.max(tensor_data)
            hist = torch.histc(tensor_data, bins=10, min=minval, max=maxval)
            fig = DataPlot.plot_1D(hist, minval.item(), maxval.item())

        # plot the image to a numpy array
        image = DataPlot.plt2arr(fig)
        # Fix color and shape
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image = cv2.resize(image, self.shape_expected)
        # write a message
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50,50)
        fontScale = 0.5
        fontColor = self.colorBGR
        thickness = 1
        lineType = 2
        cv2.putText(image, internal_message, position, font, fontScale, fontColor, thickness, lineType) 
        position = (50,75)
        cv2.putText(image, message, position, font, fontScale, fontColor, thickness, lineType) 

        # create a new video writer?
        if self.internal_out[key] is None or self.internal_frames[key] == self.maxframes:
            #print(f'key out:{key}')
            self.internal_counter[key] = self.internal_counter[key] + 1
            self.internal_frames[key] = 0
            if self.internal_out[key] is not None:
                self.internal_out[key].release()
            # mp4 | 0x7634706d for mp4 (it works in some machines)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            # Get the current device index that runs the code
            device_index = -1
            device_index = torch.cuda.current_device()
            #fourcc = cv2.VideoWriter_fourcc(*'DIVX') # avi 
            fname_out = self.path_root + '/' + key + '_' + str(device_index) + '_' + str(unique_id) + '_' + str(self.internal_counter[key]) + '_video.mp4'
            print(f'fname_out:{fname_out}')
            self.internal_out[key] = cv2.VideoWriter(fname_out,
                                                fourcc, self.fps, self.shape_expected)

        # place the image in the video
        self.internal_out[key].write(image)
        plt.close()

        self.internal_frames[key] = self.internal_frames[key] + 1