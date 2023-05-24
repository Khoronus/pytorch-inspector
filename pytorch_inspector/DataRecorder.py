import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

from pytorch_inspector.utils.DataPlot import DataPlot

__all__ = ["DataRecorder"]

class DataRecorder():
    """
    Record passed data.
    """    
    def __init__(self, 
                 shape_expected : tuple, fps : float, maxframes : int, 
                 path_root : str):
        """
        Args:
        - **shape_expected**: Expected video output size.
        - **fps**: Expected video frame rate.
        - **maxframes**: Maximum number of frames for a video.
        - **path_root**: Root to the output destination.
        """
        super().__init__() # Call the parent class constructor

        self.shape_expected = shape_expected
        self.fps = fps
        self.maxframes = maxframes
        self.path_root = path_root

        # Internal counter for the number of frames added 
        self.internal_frames = {}
        # Unique internal counter with the number of videos created
        self.internal_counter = {}
        # Container with the video writer associated
        self.internal_out = {}

        try:
            from pathlib import Path
            Path(path_root).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import sys
            exc_type, exc_value, exc_tb = sys.exc_info()
            import traceback
            stack_summary = traceback.extract_tb(exc_tb)
            last_entry = stack_summary[-1]
            file_name, line_number, func_name, text = last_entry
            import inspect
            print(f'{__name__}.{inspect.currentframe().f_code.co_name} ex occurred in {file_name}, line {line_number}, in {func_name}')
            print(f'Line:{text}')
            print(f'ex:{e}')

    def flush(self):
        """
        Releases existing running videos.
        Note: The content may be empty if data was created in another process.
        """
        # release the video writer associated to the dictionary
        for key_dict in self.internal_out:
            if self.internal_out[key_dict] is not None:
                self.internal_out[key_dict].release()

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

        Args:
        - **unique_id**: Unique identifier associated to this process
        - **key**: input unique identifier.
        - **internal_message**: message passed by the main process.
        - **message**: message to write on the image.
        - **input_data**: passed input to visualize.
        """
        try:
            # Check if the key was seen before, otherwise add it.
            if key not in self.internal_counter:
                self.internal_frames[key] = 0
                self.internal_counter[key] = 0
                self.internal_out[key] = None

            # Check the data type
            if isinstance(input_data, np.ndarray):
                #tensor_np = input_data
                print('EEE Wrong data type')
            if isinstance(input_data, torch.Tensor):
                if input_data == 'cpu':
                    tensor_data = input_data.detach().squeeze(0)
                else:
                    tensor_data = input_data.cpu().detach().squeeze(0)

            # Plot to figure
            if tensor_data.dim() == 2:
                fig = DataPlot.tensor_plot2D(tensor_data)
            else:
                minval=torch.min(tensor_data)
                maxval=torch.max(tensor_data)
                hist = torch.histc(tensor_data.cpu().detach(), bins=10, min=minval, max=maxval)
                fig = DataPlot.plot_1D(hist.cpu().detach(), minval, maxval)

            # plot the image to a numpy array
            image = DataPlot.plt2arr(fig)
            # Fix color and shape
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image = cv2.resize(image, self.shape_expected)
            # write a message
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50,50)
            fontScale = 0.5
            fontColor = (255,0,0)
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
                #fourcc = cv2.VideoWriter_fourcc(*'DIVX') # avi 
                self.internal_out[key] = cv2.VideoWriter(self.path_root + '/' + 
                                                    key + '_' + str(unique_id) + '_' + str(self.internal_counter[key]) + '_video.mp4',
                                                    fourcc, self.fps, self.shape_expected)

            # place the image in the video
            self.internal_out[key].write(image)
            plt.close()

            self.internal_frames[key] = self.internal_frames[key] + 1

        except Exception as e:
            import sys
            exc_type, exc_value, exc_tb = sys.exc_info()
            import traceback
            stack_summary = traceback.extract_tb(exc_tb)
            last_entry = stack_summary[-1]
            file_name, line_number, func_name, text = last_entry
            import inspect
            print(f'{__name__}.{inspect.currentframe().f_code.co_name} ex occurred in {file_name}, line {line_number}, in {func_name}')
            print(f'Line:{text}')
            print(f'ex:{e}')
