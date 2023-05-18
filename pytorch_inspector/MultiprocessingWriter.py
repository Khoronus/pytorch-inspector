import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

__all__ = ["MultiprocessingWriter"]

class MultiprocessingWriter(torch.multiprocessing.Process):
    """
    Information writer for multiprocessing operation.
    """    
    # Initialize the class with a file name
    def __init__(self, unique_id : int, event : torch.multiprocessing.Event, 
                 queue : torch.multiprocessing.Queue, keys : list, 
                 shape_expected : tuple, fps : float, maxframes : int, max_elapsed_time : float):
        """
        Args:
        - **unique_id**: Unique identifier associated to this process
        - **event**: Multiprocessing event
        - **queue**: Data structure used to receive data from the calling process.
        - **keys**: Accepted tensor keys identifier passed by the queue (see below).
        - **shape_expected**: Expected video output size.
        - **fps**: Expected video frame rate.
        - **maxframes**: Maximum number of frames for a video.
        - **max_elapsed_time**: Maximum time without receiving new data before timeout.

        The queue is expected to receive list of data in the format:
        [key(str),message(str),tensor(cpu/gpu)]
        A tensor name 'a_tensor' should have key 'a_tensor'. 
        """
        super().__init__() # Call the parent class constructor

        self.this_unique_id = unique_id
        self.event = event
        self.queue = queue

        self.keys = keys
        self.shape_expected = shape_expected
        self.fps = fps
        self.maxframes = maxframes
        self.max_elapsed_time = max_elapsed_time

    def run(self) -> None:
        """
        Function called when the process starts.
        The process automatically creates a video of the format:
        key + unique_id + '_' + internal_counter + '_video.mp4'
        where the internal_counter is a progressive number starting from 0.

        The data is passed from the calling process to this process via queue.
        The tensor shape is expected to be [B x W x H] where B (batch) is equal to 1.
        """

        queue = self.queue

        unique_id = self.this_unique_id
        shape_expected = self.shape_expected
        fps = self.fps
        maxframes = self.maxframes
        max_elapsed_time = self.max_elapsed_time
        internal_iterations = {}
        internal_counter = {}
        internal_out = {}
        for key in self.keys:
            internal_iterations[key] = 0
            internal_counter[key] = 0
            internal_out[key] = None

        def tensor_plot2D(key, message : str, tensor_data : torch.Tensor) -> None:
            """
            Plot a tensor image and write a message on the top. The image is then added to a video.
            Args:
            - **key**: tensor identifier.
            - **message**: message to write on the image.
            - **tensor_data**: tensor to visualize.
            """
            def plt2arr(fig, draw=True) -> np.array:
                """
                Convert a plot to numpy array.
                Args:
                - **fig**: Plot figure.
                - **draw**: Drwaw the figure to canvas?
                """
                if draw:
                    fig.canvas.draw()
                rgba_buf = fig.canvas.buffer_rgba()
                (w,h) = fig.canvas.get_width_height()
                rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
                return rgba_arr

            # Get the meshgrid size
            x = np.linspace(0, tensor_data.shape[1], tensor_data.shape[1])
            y = np.linspace(0, tensor_data.shape[0], tensor_data.shape[0])
            x, y = np.meshgrid(x, y)

            # Create a surface plot
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            # Convert tensor to numpy array
            tensor_np = tensor_data.numpy()   # tensor (pytorch)
            ax.plot_surface(x, y, tensor_np, cmap=cm.Spectral_r)

            # plot the image to a numpy array
            image = plt2arr(fig)
            # Fix color and shape
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image = cv2.resize(image, shape_expected)
            # write a message
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50,50)
            fontScale = 1
            fontColor = (255,0,0)
            thickness = 1
            lineType = 2
            cv2.putText(image, message, position, font, fontScale, fontColor, thickness, lineType) 

            # create a new video writer?
            if internal_out[key] is None or internal_iterations[key] == maxframes:
                #print(f'key out:{key}')
                internal_counter[key] = internal_counter[key] + 1
                internal_iterations[key] = 0
                if internal_out[key] is not None:
                    internal_out[key].release()
                # mp4
                internal_out[key] = cv2.VideoWriter(key + str(unique_id) + '_' + str(internal_counter[key]) + '_video.mp4',
                                                    0x7634706d, fps, shape_expected)

            # place the image in the video
            internal_out[key].write(image)
            plt.close()

            internal_iterations[key] = internal_iterations[key] + 1

        # loop until the event is set
        # or the timeout is reached
        start_time = time.time()        
        while not self.event.is_set():

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= max_elapsed_time:
                print(f'process stops. Elapsed time:{elapsed_time}/{max_elapsed_time}')
                break

            # the queue gets in non-blocking mode
            # if empty, continue the cycle to avoid deadlocks
            if queue is not None and queue.empty():
                time.sleep(1.0)
                continue

            try:
                # Get the obj from the queue
                content = queue.get_nowait()
                #content = queue.get()

                key = content[0]
                message = content[1]
                local_data = content[2]
            except Exception as e:
                print(f'Writer ex:{e}')

            if local_data is not None:
                # create a video
                if local_data.device == 'cpu':
                    data = local_data.detach().squeeze(0)
                else:
                    data = local_data.cpu().detach().squeeze(0)
                tensor_plot2D(key, message, data)
                # release the shared memory
                del local_data
                # reset the timer
                start_time = time.time()        

        # release the video writer associated to the dictionary
        for key_dict in internal_out:
            if internal_out[key_dict] is not None:
                internal_out[key_dict].release()
        print('done')

