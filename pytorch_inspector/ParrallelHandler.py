import atexit
from typing import Any, Optional
import torch

from pytorch_inspector.MultiprocessingWriter import MultiprocessingCtx, MultiprocessingWriterFork, MultiprocessingWriterSpawn
from pytorch_inspector.DataStruct import ProcessInfoData
from pytorch_inspector.utils.Decorators import *

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
    @exception_decorator
    def __init__(self, 
                 callback_onrun : Optional[Any],
                 callback_onclosing : Optional[Any],
                 frequency : float, timeout : float,
                 target_method : str,
                 daemon : bool = True):

        """
        Initialization keeps track of all the events created to close the processes once the program terminate.
        Args:
        - **callback_onrun**: Callback function on running process.
        - **callback_onclosing**: Callback function when the process terminates.
        - **frequency**: Backpropagation tensor sampling recording.
        - **timeout**: Maximum time without receiving new data before timeout.
        - **target_method**: Target method used to create a process (fork/spawn).
        - **daemon**: Create the new process as deamon (True/False).
        """    
        self.events = []
        # Container for the counter of active messages passed between all the processes
        # A new process will create a new dictionary of valid keys and total messages passed.
        # The counter is used to allow a maximum of 1 active message to be 
        # passed to a child process at time.
        self.active_messages_counter = []

        self.callback_onrun = callback_onrun
        self.callback_onclosing = callback_onclosing
        self.frequency = frequency
        self.timeout = timeout
        self.target_method = target_method
        self.daemon = daemon

        # Internal message passed to all the active processes
        self.internal_message = 'empty'
        # Pass the data to the active processes?
        self.pass_to_process = True 
        # Container with the create process information
        self.container_process_info = dict()
        # Unique interna ID for new created process (NO PID)
        self.internal_unique_id = 0
        # Unique status to define if the class is enabled.
        self.enabled = True
        atexit.register(self.stop)

    def __del__(self):
        """
        Destructor tries to stop running processes.
        Note: The processes should be terminated if run as fork, since are of type daemon.
        """    
        print('ParallelHandler.__del__')
        self.stop()

    @exception_decorator
    def stop(self) -> None:
        """
        Stop running processes.
        """    
        #print(f'ParallelHandler.stop:{self.events}')
        for event in self.events:
            if event is not None:
                event.set()

    def set_enabled(self, enabled) -> None:
        """
        It set the enable status of the class.
        The creation/attachment of new process is possible only if enable is True.
        Args:
        - **enabled**: If True the processes will be created. No, otherwise.
        """    
        self.enabled = enabled

    @exception_decorator
    def get_last_key(self):
        """
        It gets the last keys insert in the container process info.
        Returns:
        - It returns the last key if the container has at least 1 element. None otherwise.
        """    
        last_index = len(self.container_process_info) - 1
        if last_index < 0:
            return None
        return list(self.container_process_info.keys())[-1]

    @exception_decorator
    def get_internal_unique_id(self):
        """
        It gets the unique internal ID assigned to the process (not PID).
        Returns:
        - The last internal numerical ID assigned to last process. -1 if none assigned yet. 
        """    
        return self.internal_unique_id - 1

    def set_internal_message(self, internal_message : str) -> None:
        """
        Message passed to all the active processes while running.
        """
        self.internal_message = internal_message

    def set_pass_to_process(self, pass_to_process : bool) -> None:
        """
        Set if the data is passed to the process
        """
        self.pass_to_process = pass_to_process

    @exception_decorator
    def parallel_start_process(self, args, start_method : str, daemon : bool):
        """
        Start a new process as fork or spawn.
        Args:
        - **args**: Configuration arguments for the class Process.
        - **start_method**: Defines the starting method for the process (fork/spawn).
        - **daemon**: Create the new process as deamon (True/False).
        """    
        print(f'start_process args:{args}')
        # Create the process object inside the function
        if start_method == 'fork':
            obj = MultiprocessingWriterFork(*args)
            print(f'WARNING: Method {start_method} may cause a deadlock if run with lightning-gpu or other CUDA process. Spawn is recommended.')
        elif start_method == 'spawn':
            obj = MultiprocessingWriterSpawn(*args)
        else:
            raise ValueError('Invalid choice')
        obj.daemon=daemon
        obj.start()

    def new_process(self, unique_id : int, target_method : str, daemon : bool) -> None:
        """
        Start a new process. The event, queue, and method are automatically initialized/selected.
        Args:
        - **unique_id**: Process unique identifier
        - **target_method**: Target process method. Fork/Spawn are the possible choices. If None, it uses the start method.
                             Warning: Fork may cause deadlock in some cases (i.e. lightning).
        - **daemon**: Create the new process as deamon (True/False).
        Returns:
        - queue for the communication, context to track the process status.
        """    
        # get the current start method
        method = torch.multiprocessing.get_start_method()
        print(f'current start method:{method}')
        # get the selected method
        if target_method is not None:
            method = target_method
        print(f'current selected method:{method}')

        # Create a queue object to share data between processes
        # Important : The to/from must be reversed respect the process
        # to -> from (main process)  
        # from <- to (child process)
        #queue_to = torch.multiprocessing.Queue()
        #queue_from = torch.multiprocessing.Queue()
        #event = torch.multiprocessing.Event()
        if method == 'fork':
            queue_to =  MultiprocessingCtx.ctx_fork.Queue()
            queue_from = MultiprocessingCtx.ctx_fork.Queue()
            event = MultiprocessingCtx.ctx_fork.Event()
        elif method == 'spawn':
            queue_to =  MultiprocessingCtx.ctx_spawn.Queue()
            queue_from = MultiprocessingCtx.ctx_spawn.Queue()
            event = MultiprocessingCtx.ctx_spawn.Event()
        else:
            raise Exception(f'ParallelHandler.new_process: Method [{method}] is not supported (only spawn/fork).')

        # Pass the arguments for creating the process object as a tuple
        contexts = []
        args = (unique_id, event, queue_to, queue_from, self.timeout, self.callback_onrun, self.callback_onclosing)
        #if method == 'spawn':
        #    # Known issue github.com/pytorch/pytorch/issues/30461
        #    # torch.multiprocessing.spawn fails when join=False
        #    context = torch.multiprocessing.spawn(self.parallel_start_process_spawn, args=(args,), nprocs=1, join=False)
        #    # Wait a message from the process
        #    print('ParallelHandler.new_process:wait for the child process to be ready')
        #    data_from_process = queue_from.get()    # necessary for synchronizatino with spawn process
        #    print('ParallelHandler.new_process:child process is ready')
        #    #import time
        #    #time.sleep(3)  # <<< change this with some better solution. 2 ways queue?
        #elif method == 'fork':
        #    if torch.cuda.is_initialized():
        #        raise Exception('ParallelHandler.new_process: cannot fork if cuda is initialized. Please call before any cuda call (i.e. to(device)).')
        #    context = self.parallel_start_process_fork(args=args)
        #    # Wait a message from the process
        #    print('ParallelHandler.new_process:wait for the child process to be ready')
        #    data_from_process = queue_from.get()    # necessary for synchronizatino with spawn process
        #    print('ParallelHandler.new_process:child process is ready')
        #else:
        #    raise Exception(f'ParallelHandler.new_process: Method [{method}] is not supported (only spawn/fork).')

        # Fork may cause DEADLOCK with lightning, spawn may work normally
        context = self.parallel_start_process(args=args, start_method=method, daemon=daemon)
        print('ParallelHandler.new_process:wait for the child process to be ready')
        data_from_process = queue_from.get()    # necessary for synchronizatino with spawn process
        print('ParallelHandler.new_process:child process is ready')

        contexts.append(context)
        self.events.append(event)

        return queue_to, queue_from, contexts

    @exception_decorator
    def tensor_backpropagation_hook_wrapper(self, name, maxsize_queue, x, 
                                            queue_to, queue_from, active_messages_counter,
                                            callback_transform : Optional[Any]):
        """
        Wrapper to the torch hook handler. Additional arguments are passed.
        The current code try to record a tensor every N back propagation calls.
        The tensor is passed to CPU to reduce the memory consumption in GPU (it may slow the process).
        Args:
        - **name**: Tensor name (same used in the key)
        - **maxsize_queue**: maximum number of elements insert in the queue. It is based on the total number of tracked tensors 
                             in the process x 4.
        - **x**: Tensor object associated to the gradient (grad is calculated over x).
        - **queue_to**: Queue to exchange information to called processes.
        - **queue_from**: Queue to get information from called processes.
        - **active_messages_counter**: Container with the total number of messages passed to the child process for each key.
        - **callback_transform**: How to transform the passed tensor
         Returns:
        - hook for the backpropagation
        """    
        # register a tensor hook
        #print(f'tensor_backpropagation_hook_wrapper:{name}')
        # It counts how many times this hook has been called
        internal_counter = {name : 0}
        # Add an element to the queue only if the condition is true
        do_add_to_queue = True
        def hook(grad):
            try:
                if queue_from.qsize() > 0: 
                    # check if the counter should be decreased
                    content = queue_from.get_nowait()
                    #print(f'content:{content} active_messages:{active_messages}')
                    active_messages_counter[content[0]] -= 1
                    #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                pass

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: do_add_to_queue = True

            if internal_counter[name] % self.frequency == 0 and active_messages_counter[name] == 0:
                # put the tensor in the queue
                #print(f'{name} queue.qsize:{queue.qsize()}/{maxsize_queue}')
                if do_add_to_queue:
                    if queue_to.qsize() > maxsize_queue:
                        do_add_to_queue = False
                        pass
                    else:
                        active_messages_counter[name] += 1
                        if callback_transform is None:
                            shared_data=x.cpu().clone().detach()
                        else:
                            shared_data=callback_transform(x)
                        info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                    message=str(internal_counter[name]), shared_data=shared_data)
                        queue_to.put_nowait(info_data)    

            internal_counter[name] = internal_counter[name] + 1
            return
        return hook
    
    @exception_decorator
    def layer_backpropagation_hook_wrapper(self, name, maxsize_queue, 
                                           queue_to, queue_from, active_messages_counter, 
                                           callback_transform : Optional[Any]):
        """
        Wrapper to the torch hook handler for layers. Additional arguments are passed.
        The current code try to record information every N back propagation calls.
        Args:
        - **name**: Layer associated name (same used in the key)
        - **maxsize_queue**: maximum number of elements insert in the queue. It is based on the total number of tracked layers 
                             in the process x 2.
        - **queue_to**: Queue to exchange information to called processes.
        - **queue_from**: Queue to get information from called processes.
        - **active_messages_counter**: Container with the total number of messages passed to the child process for each key.
        - **callback_transform**: How to transform the passed tensor
        Returns:
        - hook for the backpropagation
        """    
        #print(f'layer_backpropagation_hook_wrapper:{name}')
        # It counts how many times this hook has been called
        internal_counter = {name : 0}
        # Total number of active messages passed to queue with the name
        #active_messages = {name : 0}
        # Add an element to the queue only if the condition is true
        do_add_to_queue = True
        # register a tensor hook
        def hook(module, grad_input, grad_output):
            try:
                if queue_from.qsize() > 0: 
                    # check if the counter should be decreased
                    content = queue_from.get_nowait()
                    #print(f'content:{content} active_messages:{active_messages}')
                    active_messages_counter[content[0]] -= 1
                    #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                pass

            if grad_output is None:
                return hook

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: 
                #print(f'Ado_add_to_queue:{do_add_to_queue}')
                do_add_to_queue = True
            #print(f'Ado_add_to_queue:{do_add_to_queue} -- {name} queue.qsize:{queue.qsize()}/{maxsize_queue}')
            if internal_counter[name] % self.frequency == 0 and active_messages_counter[name] == 0:
                # Put the obj in the queue
                if do_add_to_queue:
                    if queue_to.qsize() > maxsize_queue:
                        do_add_to_queue = False
                        pass
                    else:
                        active_messages_counter[name] += 1
                        if callback_transform is None:
                            shared_data=grad_output[0].cpu().clone().detach()
                        else:
                            shared_data=callback_transform(grad_output[0])
                        info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                    message=str(internal_counter[name]), shared_data=shared_data)
                        queue_to.put_nowait(info_data)    
        
            internal_counter[name] = internal_counter[name] + 1
        return hook

    @exception_decorator
    def model_forwardpropagation_hook_wrapper(self, name, maxsize_queue, 
                                              queue_to, queue_from, active_messages_counter,
                                              callback_transform : Optional[Any]):
        """
        Wrapper to the torch hook handler for layers. Additional arguments are passed.
        The current code try to record information every N forward propagation calls.
        Args:
        - **name**: Layer associated name (same used in the key)
        - **maxsize_queue**: maximum number of elements insert in the queue. It is based on the total number of tracked models 
                             in the process x 2.
        - **queue_to**: Queue to exchange information to called processes.
        - **queue_from**: Queue to get information from called processes.
        - **active_messages_counter**: Container with the total number of messages passed to the child process for each key.
        - **callback_transform**: How to transform the passed tensor
        Returns:
        - hook for the forward propagation
        """    
        #print(f'model_forwardpropagation_hook_wrapper:{name}')
        # It counts how many times this hook has been called
        internal_counter = {name : 0}
        # Add an element to the queue only if the condition is true
        do_add_to_queue = True
        def hook(module, input, output):

            # a new name found but not in the list
            if name not in active_messages_counter:
                active_messages_counter[name] = 0

            try:
                if queue_from.qsize() > 0: 
                    # check if the counter should be decreased
                    content = queue_from.get_nowait()
                    #print(f'content:{content} active_messages_counter:{active_messages_counter}')
                    active_messages_counter[content[0]] -= 1
                    #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                print(f'ex_queue_from:{e}')
                pass

            if output is None:
                return hook

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: 
                #print(f'Bdo_add_to_queue:{do_add_to_queue}')
                do_add_to_queue = True
            #print(f'do_add_to_queue:{do_add_to_queue} -- {name} queue.qsize:{queue_to.qsize()}/{maxsize_queue} pp:{self.pass_to_process} ic:{internal_counter[name]} f:{self.frequency} amc:{active_messages_counter[name]}')
            if self.pass_to_process and internal_counter[name] % self.frequency == 0 and active_messages_counter[name] == 0:
                #print(f'name:{name} size:{queue_to.qsize()} m:{maxsize_queue}')

                # Put the obj in the queue
                if do_add_to_queue:
                    if queue_to.qsize() > maxsize_queue:
                        do_add_to_queue = False
                        pass
                    else:
                        active_messages_counter[name] += 1

                        output_data = output
                        if isinstance(output, torch.Tensor) == False:
                            output_data = output[0]
                        #print(f'output_data:{output_data} grad:{g}')
                        if callback_transform is None:
                            shared_data=output_data.cpu().clone().detach()
                        else:
                            shared_data=callback_transform(output_data)                            
                        info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                    message=str(internal_counter[name]), shared_data=shared_data)
                        queue_to.put_nowait(info_data)    

            internal_counter[name] = internal_counter[name] + 1
        return hook

    @exception_decorator
    def call_process(self, list_names, unique_id_connect_to):
        """
        It creates or connect to a process.
        Args:
        - **list_names**: Container with the list of names to track
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        Returns:
        - The given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        # start the new process or connect to
        queue_to, queue_from, contexts = None, None, None
        # start the new process or connect to
        if unique_id_connect_to in self.container_process_info:
            unique_id = unique_id_connect_to
            queue_to = self.container_process_info[unique_id]['queue_to']
            queue_from = self.container_process_info[unique_id]['queue_from']
        else:
            unique_id = self.internal_unique_id
            self.internal_unique_id += 1
            #self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            queue_to, queue_from, contexts = self.new_process(unique_id, self.target_method, self.daemon)
            # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
            # collect the process information
            self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            # new counter for the passed names
            dict_counter = dict()
            #for i in range(len(list_names)):
            #    dict_counter[list_names[i]] = 0
            dict_counter = {list_names[i]:0 for i in range(len(list_names))}
            self.active_messages_counter.append(dict_counter)
        return unique_id, queue_to, queue_from, contexts

    @exception_decorator
    def track_tensor(self, unique_id_connect_to, list_tensors, callback_transform : Optional[Any]):#, tensors_hook_mode):
        """
        Automatically creates the hook and processes for the list of tensors to track.
        The tensors hook only backpropagation.
        Args:
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        - **list_tensors**: List of tensors to track {'a_tensor':a_tensor,'b_tensor':b_tensor}
        - **tensors_hook_mode**: Container with the type of hook associated to a tensor {'a_tensor':{b},'b_tensor':{b}}
                                 If no matching name is found, it uses the backpropagation.
                                 Supported mode:
                                 b : backpropagation
        - **callback_transform**: How to transform the passed tensor
        Returns:
        - The given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        if self.enabled == False:
            return None, None, None, None

        #print(f'track_tensor:{list_tensors}')
        queue_to, queue_from, contexts = None, None, None

        # Get the list of the names
        list_names = []
        for name, value in list_tensors.items():
            if isinstance(value, torch.Tensor) and value.requires_grad is False:
                print(f'ParallelHandler.track_tensor warning:{name} requires_grad forced to True to support hook')
                value.requires_grad_(True)
            #print(f'name:{name} value:{value}')
            list_names.append(name)

        #print(f'# list_names:{len(list_names)}')

        # at least 1 tensor to track
        if len(list_names) > 0:
            # start the new process or connect to
            #if unique_id_connect_to in self.container_process_info:
            #    unique_id = unique_id_connect_to
            #    queue_to = self.container_process_info[unique_id]['queue_to']
            #    queue_from = self.container_process_info[unique_id]['queue_from']
            #else:
            #    unique_id = self.internal_unique_id
            #    self.internal_unique_id += 1
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            #    queue_to, queue_from, contexts = self.new_process(unique_id, self.target_method, self.daemon)
            #    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
            #    # collect the process information
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            unique_id, queue_to, queue_from, contexts = self.call_process(list_names, unique_id_connect_to)

            # new counter for the passed names
            dict_counter = dict()
            #for i in range(len(list_names)):
            #    dict_counter[list_names[i]] = 0
            dict_counter = {list_names[i]:0 for i in range(len(list_names))}
            self.active_messages_counter.append(dict_counter)

            # create the hook
            for name, value in list_tensors.items():
                value.register_hook(self.tensor_backpropagation_hook_wrapper(name, len(list_names) * 4, value, 
                                                                                queue_to, queue_from, self.active_messages_counter[-1],
                                                                                callback_transform))
            return unique_id, queue_to, queue_from, contexts
        else:
            print('ParallelHandler.track_tensor warning:unable to hook any tensor')
        return None, None, None, None

    @exception_decorator
    def track_layer(self, unique_id_connect_to, list_layers, callback_transform : Optional[Any]):
        """
        Automatically creates the hook and processes for the list of layers to track.
        The layer hook only backpropagation.
        Args:
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        - **list_layers**: List of layers to track {'a_layer':a_layer,'b_layer':b_layer}
        - **callback_transform**: How to transform the passed tensor

        Returns:
        - the given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        if self.enabled == False:
            return None, None, None, None

        queue_to, queue_from, contexts = None, None, None
        # Get the list of the names
        list_names = []
        for name, value in list_layers.items():
            list_names.append(name)

        # at least 1 tensor to track
        if len(list_names) > 0:
            # start the new process or connect to
            #if unique_id_connect_to in self.container_process_info:
            #    unique_id = unique_id_connect_to
            #    queue_to = self.container_process_info[unique_id]['queue_to']
            #    queue_from = self.container_process_info[unique_id]['queue_from']
            #else:
            #    unique_id = self.internal_unique_id
            #    self.internal_unique_id += 1
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            #    queue_to, queue_from, contexts = self.new_process(unique_id, self.target_method, self.daemon)
            #    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
            #    # collect the process information
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            unique_id, queue_to, queue_from, contexts = self.call_process(list_names, unique_id_connect_to)

            # new counter for the passed names
            dict_counter = dict()
            #for i in range(len(list_names)):
            #    dict_counter[list_names[i]] = 0
            dict_counter = {list_names[i]:0 for i in range(len(list_names))}
            self.active_messages_counter.append(dict_counter)

            # create the hook
            for name, value in list_layers.items():
                value.register_full_backward_hook(self.layer_backpropagation_hook_wrapper(name, len(list_names) * 2, 
                                                                                            queue_to, queue_from, self.active_messages_counter[-1],
                                                                                            callback_transform))
            return unique_id, queue_to, queue_from, contexts

        else:
            print('ParallelHandler.track_layer warning:unable to hook any layer')

        return None, None, None, None

    @exception_decorator
    def track_model(self, unique_id_connect_to, list_models, callback_transform : Optional[Any]):
        """
        Automatically creates the hook and processes for the list of layers to track.
        The layer hook only forward propagation.
        Args:
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        - **list_models**: List of models to track {'a_model':a_model,'b_model':b_model}
        - **callback_transform**: How to transform the passed tensor

        Returns:
        - the given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        if self.enabled == False:
            return None, None, None, None

        #print(f'track_model:{list_models}')
        queue_to, queue_from, contexts = None, None, None
        # Get the list of the names
        list_names = []
        for name, value in list_models.items():
            list_names.append(name)

        # at least 1 tensor to track
        if len(list_names) > 0:
            # start the new process or connect to
            #if unique_id_connect_to in self.container_process_info:
            #    unique_id = unique_id_connect_to
            #    queue_to = self.container_process_info[unique_id]['queue_to']
            #    queue_from = self.container_process_info[unique_id]['queue_from']
            #else:
            #    unique_id = self.internal_unique_id
            #    self.internal_unique_id += 1
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            #    queue_to, queue_from, contexts = self.new_process(unique_id, self.target_method, self.daemon)
            #    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
            #    # collect the process information
            #    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}

            #    # new counter for the passed names
            #    dict_counter = dict()
            #    #for i in range(len(list_names)):
            #    #    dict_counter[list_names[i]] = 0
            #    dict_counter = {list_names[i]:0 for i in range(len(list_names))}
            #    self.active_messages_counter.append(dict_counter)
            unique_id, queue_to, queue_from, contexts = self.call_process(list_names, unique_id_connect_to)

            # create the hook
            for name, value in list_models.items():
                value.register_forward_hook(self.model_forwardpropagation_hook_wrapper(name, len(list_names) * 20, 
                                                                                        queue_to, queue_from, self.active_messages_counter[unique_id], #[-1],
                                                                                        callback_transform))
            return unique_id, queue_to, queue_from, contexts
        else:
            print('ParallelHandler.track_model warning:unable to hook any layer')

        return None, None, None, None
