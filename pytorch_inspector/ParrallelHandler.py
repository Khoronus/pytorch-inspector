import torch

from .MultiprocessingWriter import MultiprocessingWriter

__all__ = ["ParrallelHandler"]

class SingletonMeta(type):
    """
    Meta Singleton
    """    
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ParrallelHandler(metaclass=SingletonMeta):
    """
    Singleton class to manage all processes instantiated by the calling program.
    """    
    def __init__(self):
        """
        Initialization keeps track of all the events created to close the processes once the program terminate.
        """    
        self.events = []

    def __del__(self):
        """
        Destructor tries to stop running processes.
        Note: The processes should be terminated if run as fork, since are of type daemon.
        """    
        self.stop()

    def stop(self):
        """
        Stop running processes.
        """    
        try:
            for event in self.events:
                if event is not None:
                    event.set()
        except Exception as e:
            print(f'ParallelHandler.stop ex:{e}')

    def parallel_start_process_spawn(self, rank, args):
        """
        Start a process spawn.
        Args:
        - **rank**: Rank of the process.
        - **args**: Arguments to initialize the MultiprocessingWriter.
        """    
        print(f'start_process[{rank}]_spawn args:{args}')
        # Create the process object inside the function
        obj = MultiprocessingWriter(*args)
        obj.start()

    def parallel_start_process(self, args):
        """
        Start a process fork.
        Args:
        - **args**: Arguments to initialize the MultiprocessingWriter.

        Returns:
        - The process
        """    
        print(f'start_process args:{args}')
        # Create the process object inside the function
        obj = MultiprocessingWriter(*args)
        obj.daemon = True
        obj.start()
        return obj

    def new_process(self, unique_id, keys):
        """
        Start a new process. The event, queue, and method are automatically initialized/selected.
        Args:
        - **unique_id**: Process unique identifier
        - **keys**: Accepted associated tensor keys

        Returns:
        - queue for the communication, context to track the process status.
        """    
        event = torch.multiprocessing.Event()
        # Create a queue object to share data between processes
        queue = torch.multiprocessing.Queue()

        # get the current start method
        method = torch.multiprocessing.get_start_method()
        print(f'current start method:{method}')

        # Pass the arguments for creating the process object as a tuple
        contextes = []
        args = (unique_id, event, queue, keys, (640,480), 20.0, 50, 120)
        if method == 'spawn':
            context = torch.multiprocessing.spawn(self.parallel_start_process_spawn, args=(args,), nprocs=1, join=False)
        elif method == 'fork':
            context = self.parallel_start_process(args=args)
        contextes.append(context)

        self.events.append(event)
        return queue, contextes

    def myobject(self, name, x, queue):
        """
        Wrapper to the torch hook handler. Additional arguments are passed.
        The current code try to record a tensor every N back propagation calls.
        The tensor is passed to CPU to reduce the memory consumption in GPU (it may slow the process).
        Args:
        - **name**: Tensor name (same used in the key)
        - **x**: Tensor object associated to the gradient (grad is calculated over x).
        - **queue**: Queue to exchange information to called processes.
 
        Returns:
        - hook for the backpropagation
        """    
        # register a tensor hook
        internal_counter = {name : 0}
        def myhook(grad):
            if internal_counter[name] % 20 == 0:
                list_data = []
                list_data.append(name)
                list_data.append(str(internal_counter[name]))
                list_data.append(x.cpu())#.clone().share_memory_())
                # Put the obj in the queue
                if queue.qsize() > 4:
                    pass
                else:
                    #print(f'myhook:{name} {unique_id}')
                    queue.put_nowait(list_data)    

                # The function adds also the gradient information
                list_data = []
                list_data.append(name + 'grad')
                list_data.append(str(internal_counter[name]))
                list_data.append(grad.cpu())#.clone().share_memory_())
                # Put the obj in the queue
                if queue.qsize() > 4:
                    pass
                else:
                    #print(f'myhook:{name} {unique_id}')
                    queue.put_nowait(list_data)    

            internal_counter[name] = internal_counter[name] + 1
            return
        return myhook
    
    def track_tensor(self, unique_id, list_tensors):
        """
        Automatically creates the hook and processes for the list of tensors to track.
        Args:
        - **unique_id**: Process unique identifier.
        - **list_tensors**: List of tensors to track {'a_tensor':a_tensor,'b_tensor':b_tensor}

        Returns:
        - queue for the communication, context to track the process status.
        """    
        try:
            #print(f'track_tensor:{list_tensors}')
            queue, contextes = None, None
            # Get the list of the names
            list_names = []
            for name, value in list_tensors.items():
                if isinstance(value, torch.Tensor) and value.requires_grad is False:
                    print(f'ParallelHandler.track_tensor warning:{name} requires_grad forced to True to support hook')
                    value.requires_grad_(True)
                #print(f'name:{name} value:{value}')
                list_names.append(name)
                list_names.append(name + 'grad')

            #print(f'# list_names:{len(list_names)}')

            # at least 1 tensor to track
            if len(list_names) > 0:
                # start the new process
                queue, contextes = self.new_process(unique_id, list_names)

                # create the hook
                for name, value in list_tensors.items():
                    value.register_hook(self.myobject(name, value, queue))
            else:
                print('ParallelHandler.track_tensor warning:unable to hook any tensor')

        except Exception as e:
            print(f'ParallelHandler.track_tensor ex:{e}')

        return queue, contextes
