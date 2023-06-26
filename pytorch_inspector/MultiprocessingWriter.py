import atexit
from typing import Any, Optional
import time
import torch

from pytorch_inspector.DataStruct import ProcessInfoData
from pytorch_inspector.utils.Decorators import *
from pytorch_inspector.utils.QueueOp import *

#__all__ = ["MultiprocessingWriter, MultiprocessingWriterFork, MultiprocessingWriterSpawn"]

class MultiprocessingCtx:
    """
    Available multiprocessing contexts.
    """    
    # Initialize the class with a file name
    ctx_fork = torch.multiprocessing.get_context('fork')
    ctx_spawn = torch.multiprocessing.get_context('spawn')

class MultiprocessingWriter:
    """
    Information writer for multiprocessing operation.
    """    
    # Initialize the class with a file name
    @exception_decorator
    def __init__(self, unique_id : int, event : torch.multiprocessing.Event, 
                 queue_from : torch.multiprocessing.Queue, 
                 queue_to : torch.multiprocessing.Queue, 
                 timeout : float,
                 callback_onrun : Optional[Any],
                 callback_onclosing : Optional[Any]):
        """
        Args:
        - **unique_id**: Unique identifier associated to this process
        - **event**: Multiprocessing event
        - **queue_from**: Data structure used to receive data from the calling process.
        - **queue_to**: Data structure used to send data to the calling process.
        - **timeout**: Maximum time without receiving new data before timeout.
                                Timeout is not used if the value is <= 0.
        - **callback_onrun**: Callback function on running process.
        - **callback_onclosing**: Callback function when the process terminates.

        The queue is expected to receive list of data in the format:
        [key(str),internal_message(str),message(str),tensor(cpu/gpu)]
        A tensor name 'a_tensor' should have key 'a_tensor'. 
        """
        super().__init__() # Call the parent class constructor
        print('MultiprocessingWriter')

        self.this_unique_id = unique_id
        self.event = event
        self.queue_from = queue_from
        self.queue_to = queue_to

        self.timeout = timeout
        self.callback_onrun = callback_onrun
        self.callback_onclosing = callback_onclosing

        atexit.register(self.on_closing)

    def on_closing(self):
        """
        Function called after the program terminates. This does not guarantee that
        allocated data in other processes still exist.
        """
        if self.callback_onclosing is not None:
            self.callback_onclosing()

    @exception_decorator
    def run(self) -> None:
        """
        Function called when the process starts.
        The process automatically wait for data passed via queue.
        If the callback is not None, it passes the data to the callback function.
        """
        queue_from = self.queue_from
        queue_to = self.queue_to
        timeout = self.timeout

        # get the current start method
        method = torch.multiprocessing.get_start_method()
        print(f'MultiprocessingWriting current start method:{method}')
        # Check if the current process is a daemon
        print(f'current process is a daemon:{torch.multiprocessing.current_process().daemon}')

        # synchronize with the main process that this process is ready 
        # with a dummy empty message.
        list_data = []
        queue_to.put(list_data)    

        # loop until the event is set
        # or the timeout is reached
        start_time = time.time()        
        while not self.event.is_set():
            current_time = time.time()
            elapsed_time = current_time - start_time
            if timeout > 0 and elapsed_time >= timeout:
                print(f'process stops. Elapsed time:{elapsed_time}/{timeout}')
                break

            # the queue gets in non-blocking mode
            # if empty, continue the cycle to avoid deadlocks
            if queue_from is not None and queue_from.empty():
                time.sleep(1.0)
                continue

            # Get the obj from the queue
            content = queue_from.get_nowait()

            shared_data = None
            if isinstance(content, ProcessInfoData):
                key = content.name
                internal_message = content.internal_message
                message = content.message
                shared_data = content.shared_data

            if shared_data is not None:
                if self.callback_onrun is not None:
                    #self.callback_onrun(self.this_unique_id, key, internal_message, message, shared_data)
                    self.callback_onrun(
                        **{
                            "unique_id":self.this_unique_id,
                            "key":key,
                            "internal_message":internal_message,
                            "message":message,
                            "input_data":shared_data,
                        }
                    )
                # Share the information that the callback function with key
                # if completed
                list_data = []
                list_data.append(key)
                queue_to.put(list_data)    
                # reset the timer
                start_time = time.time()        
        # It may be redundant
        self.on_closing()

        # Clear the queues
        QueueOp.clear(self.queue_from)
        QueueOp.clear(self.queue_to)

        print(f'MultiprocessingWriter.run {self.this_unique_id} done')

class MultiprocessingWriterFork(MultiprocessingCtx.ctx_fork.Process, MultiprocessingWriter):
    def __init__(self, *args, **kwargs):
        super().__init__() # Call the parent class constructor
        MultiprocessingWriter.__init__(self, *args, **kwargs)

    def run(self) -> None:
        MultiprocessingWriter.run(self)

class MultiprocessingWriterSpawn(MultiprocessingCtx.ctx_spawn.Process, MultiprocessingWriter):
    def __init__(self, *args, **kwargs):
        super().__init__() # Call the parent class constructor
        print('MultiprocessingWriterSpawn')
        MultiprocessingWriter.__init__(self, *args, **kwargs)

    def run(self) -> None:
        MultiprocessingWriter.run(self)