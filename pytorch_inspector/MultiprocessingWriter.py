import atexit
from typing import Any, Optional
import time
import torch

from .DataStruct import ProcessInfoData

__all__ = ["MultiprocessingWriter"]

class MultiprocessingWriter(torch.multiprocessing.Process):
    """
    Information writer for multiprocessing operation.
    """    
    # Initialize the class with a file name
    def __init__(self, unique_id : int, event : torch.multiprocessing.Event, 
                 queue_from : torch.multiprocessing.Queue, 
                 queue_to : torch.multiprocessing.Queue, 
                 max_elapsed_time : float,
                 callback_onrun : Optional[Any],
                 callback_onclosing : Optional[Any]):
        """
        Args:
        - **unique_id**: Unique identifier associated to this process
        - **event**: Multiprocessing event
        - **queue_from**: Data structure used to receive data from the calling process.
        - **queue_to**: Data structure used to send data to the calling process.
        - **max_elapsed_time**: Maximum time without receiving new data before timeout.
                                Timeout is not used if the value is <= 0.
        - **callback_onrun**: Callback function on running process.
        - **callback_onclosing**: Callback function when the process terminates.

        The queue is expected to receive list of data in the format:
        [key(str),internal_message(str),message(str),tensor(cpu/gpu)]
        A tensor name 'a_tensor' should have key 'a_tensor'. 
        """
        super().__init__() # Call the parent class constructor

        self.this_unique_id = unique_id
        self.event = event
        self.queue_from = queue_from
        self.queue_to = queue_to

        self.max_elapsed_time = max_elapsed_time
        self.callback_onrun = callback_onrun
        self.callback_onclosing = callback_onclosing

        try:
            atexit.register(self.on_closing)
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

    def on_closing(self):
        """
        Function called after the program terminates. This does not guarantee that
        allocated data in other processes still exist.
        """
        if self.callback_onclosing is not None:
            self.callback_onclosing()

    def run(self) -> None:
        """
        Function called when the process starts.
        The process automatically wait for data passed via queue.
        If the callback is not None, it passes the data to the callback function.
        """
        try:

            queue_from = self.queue_from
            queue_to = self.queue_to
            max_elapsed_time = self.max_elapsed_time

            # get the current start method
            method = torch.multiprocessing.get_start_method()
            print(f'MultiprocessingWriting current start method:{method}')
            # Check if the current process is a daemon
            print(f'current process is a daemon:{torch.multiprocessing.current_process().daemon}')

            # synchronize with the main process that this process is ready 
            # with a dummy empty message.
            if method == 'spawn': 
                list_data = []
                queue_to.put(list_data)    

            # loop until the event is set
            # or the timeout is reached
            start_time = time.time()        
            while not self.event.is_set():

                current_time = time.time()
                elapsed_time = current_time - start_time
                if max_elapsed_time > 0 and elapsed_time >= max_elapsed_time:
                    print(f'process stops. Elapsed time:{elapsed_time}/{max_elapsed_time}')
                    break

                # the queue gets in non-blocking mode
                # if empty, continue the cycle to avoid deadlocks
                if queue_from is not None and queue_from.empty():
                    time.sleep(1.0)
                    continue

                # Get the obj from the queue
                content = queue_from.get_nowait()
                #content = queue.get()
                #key = content[0]
                #internal_message = content[1]
                #message = content[2]
                #local_data = content[3]
                shared_data = None
                if isinstance(content, ProcessInfoData):
                    key = content.name
                    internal_message = content.internal_message
                    message = content.message
                    shared_data = content.shared_data

                if shared_data is not None:
                    #print(f'key:{key} local_data:{local_data.shape}')
                    if self.callback_onrun is not None:
                        self.callback_onrun(self.this_unique_id, key, internal_message, message, shared_data)
                    # Share the information that the callback function with key
                    # if completed
                    list_data = []
                    list_data.append(key)
                    queue_to.put(list_data)    

                    # release the shared memory
                    #del local_data
                    # reset the timer
                    start_time = time.time()        

            # It may be redundant
            self.on_closing()

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


        print('MultiprocessingWriter.run done')

