import atexit
from typing import Any, Optional
import torch

from pytorch_inspector.MultiprocessingWriter import MultiprocessingCtx, MultiprocessingWriterFork, MultiprocessingWriterSpawn
from pytorch_inspector.DataStruct import ProcessInfoData
from pytorch_inspector.utils.Decorators import *
from pytorch_inspector.utils.MemoryOp import *

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
                 max_queue_size : int = 1000,
                 daemon : bool = True):

        """
        Initialization keeps track of all the events created to close the processes once the program terminate.
        Args:
        - **callback_onrun**: Callback function on running process.
        - **callback_onclosing**: Callback function when the process terminates.
        - **frequency**: Backpropagation tensor sampling recording.
        - **timeout**: Maximum time without receiving new data before timeout.
        - **target_method**: Target method used to create a process (fork/spawn).
        - **max_queue_size**: Maximum queue size (used to transfer data between processes).
        - **daemon**: Create the new process as deamon (True/False).

        When the number of elements in queue reaches max_queue_size, the queue is clean (all data read),
        and no more elements can be pushed to the queue.
        """    
        # Container with all the processes events
        self.events = []
        # Container with all the processes contexts
        self.contexts = []
        # Container for the counter of active messages passed between all the processes
        # A new process will create a new dictionary of valid keys and total messages passed.
        # The counter is used to allow a maximum of 1 active message to be 
        # passed to a child process at time.
        self.active_messages_counter = []
        # Container for the counter of messages passed between all the processes.
        self.passed_messages_counter = []

        self.callback_onrun = callback_onrun
        self.callback_onclosing = callback_onclosing
        self.frequency = frequency
        self.timeout = timeout
        self.target_method = target_method
        self.max_queue_size = max_queue_size
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
        # Pass data between processes only if on the same device (if True)
        self.same_device_only = False
        # How many times multiply the passed tensor memory size
        self.times_tensor_memory_size = 1.0
        # How many tensors can be pushed in a queue at time?
        # If True, the system will push a single tensor at time. It is guarantee that
        # all keys are passed the same number of times (n) or are processed one extra time (n+1) 
        self.push_only_single_tensor = False
        atexit.register(self.stop, check_is_alive=False)

    def __del__(self):
        """
        Destructor tries to stop running processes.
        Note: The processes should be terminated if run as fork, since are of type daemon.
        """    
        print('ParallelHandler.__del__')
        self.stop(check_is_alive=False)

    @exception_decorator
    def stop(self, check_is_alive : bool) -> None:
        """
        Stop running processes.
        Args:
        - **check_is_alive**: Check if running process are alive.
        """    
        #print(f'ParallelHandler.stop:{self.events}')
        for event in self.events:
            if event is not None:
                event.set()
        if check_is_alive:
            print('stop:check_is_alive')
            while self.is_alive():
                import time
                time.sleep(0.1)
            print('stop:check_is_alive done')

    @exception_decorator
    def is_alive(self) -> None:
        """
        Check if a process is alive.
        Returns:
        - It returns True if at least a process is alive. False otherwise.
        """    
        for allcontexts in self.contexts:
            for context in allcontexts:
                if context is not None:
                    if context.is_alive(): return True
        return False

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

    @exception_decorator
    def set_internal_message(self, internal_message : str) -> None:
        """
        Message passed to all the active processes while running.
        """
        self.internal_message = internal_message

    @exception_decorator
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
        Return:
        - It return an object to the new created process.
        """    
        print(f'start_process args:{args}')
        # Create the process object inside the function
        if start_method == 'fork':
            obj = MultiprocessingWriterFork(*args)
            print(f'WARNING: Method {start_method} may cause a deadlock if run with lightning-gpu or other CUDA process. Spawn is recommended.')
        elif start_method == 'spawn':
            obj = MultiprocessingWriterSpawn(*args)
        else:
            raise ValueError(f'parallel_start_process: Invalid choice:{start_method}')
        obj.daemon=daemon
        obj.start()
        return obj

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
        self.contexts.append(contexts)

        return queue_to, queue_from, contexts

    @exception_decorator
    def pass_tensor_to_queue(self, name, active_messages_counter,
                             passed_messages_counter, queue_to,
                             queue_from, internal_counter,
                             do_add_to_queue,
                             callback_transform,
                             tensor):
        """
        Wrapper to the torch hook handler for layers. Additional arguments are passed.
        The current code try to record information every N forward propagation calls.
        Args:
        - **name**: Layer associated name (same used in the key)
        - **active_messages_counter**: Container with the total number of active messages passed to the child process for each key.
        - **passed_messages_counter**: Container with the total number of messages passed to the child process for each key.
        - **queue_to**: Queue to exchange information to called processes.
        - **queue_from**: Queue to get information from called processes.
        - **internal_counter**: It counts how many times this hook has been called
        - **do_add_to_queue**: Add element to the queue.
        - **callback_transform**: How to transform the passed tensor.
        - **tensor**: Tensor to pass.
        """
        pushed_element = False

        # a new name found but not in the list
        if name not in active_messages_counter:
            active_messages_counter[name] = 0
        # a new name found but not in the list
        if name not in passed_messages_counter:
            passed_messages_counter[name] = 0

        try:
            #print(f'{name} queue_from.qsize:{queue_from.qsize()}/{maxsize_queue}')
            if queue_from.qsize() > 0: 
                # check if the counter should be decreased
                content = queue_from.get_nowait()
                active_messages_counter[content[0]] -= 1
                #print(f'content:{content} type:{type(content)}')
        except Exception as e:
            #print(f'tensor_backpropagation_hook_wrapper.ex_queue_from:{e}')
            pass

        #print(f'name:{name} ic:{internal_counter[name]} amc:{active_messages_counter[name]}')
        if internal_counter[name] % self.frequency == 0 and active_messages_counter[name] == 0:
            result = True
            # check if push only a tensor at time
            if self.push_only_single_tensor:
                # Get the current maximum value
                max_val = 0
                min_val = -1
                for element in passed_messages_counter:
                    if passed_messages_counter[element] > max_val: max_val = passed_messages_counter[element]
                    if min_val == -1 or passed_messages_counter[element] < min_val: min_val = passed_messages_counter[element]
                # check that no other active messages are running
                result = all(active_messages_counter[element] == 0 for element in active_messages_counter)
                # check that the current element did not already passed a message for processing
                if min_val != max_val and passed_messages_counter[name] == max_val: result = False
            #print(f'{name} min:{min_val} max:{max_val} pmc:{passed_messages_counter[name]} r:{result}')
            # Put the obj in the queue
            if result and do_add_to_queue:
                #print(f'{name} queue.qsize:{queue_to.qsize()}/{maxsize_queue}')

                if callback_transform is None:
                    shared_data = MemoryOp.assignTo(tensor, self.times_tensor_memory_size, self.same_device_only)
                    if shared_data != None: shared_data = shared_data.detach()
                else:
                    shared_data=callback_transform(tensor)
                # valid data
                if shared_data != None:
                    #print(f'>>> {name} min:{min_val} max:{max_val} pmc:{passed_messages_counter[name]} r:{result}')
                    #print(f'shared_data name:{name}')
                    active_messages_counter[name] += 1
                    passed_messages_counter[name] += 1
                    info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                message=str(internal_counter[name]), shared_data=shared_data)
                    queue_to.put_nowait(info_data)    
                    pushed_element = True

        internal_counter[name] = internal_counter[name] + 1        
        return pushed_element

    @exception_decorator
    def tensor_backpropagation_hook_wrapper(self, name, maxsize_queue, x, 
                                            queue_to, queue_from, active_messages_counter,
                                            passed_messages_counter,
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
        - **active_messages_counter**: Container with the total number of active messages passed to the child process for each key.
        - **passed_messages_counter**: Container with the total number of messages passed to the child process for each key.
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

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: do_add_to_queue = True
            if queue_to.qsize() > maxsize_queue: do_add_to_queue = False

            self.pass_tensor_to_queue(name, active_messages_counter,
                                      passed_messages_counter, queue_to,
                                      queue_from, internal_counter,
                                      do_add_to_queue,
                                      callback_transform, x)
        return hook
    
    @exception_decorator
    def layer_backpropagation_hook_wrapper(self, name, maxsize_queue, 
                                           queue_to, queue_from, active_messages_counter, 
                                           passed_messages_counter,
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
        - **active_messages_counter**: Container with the total number of active messages passed to the child process for each key.
        - **passed_messages_counter**: Container with the total number of messages passed to the child process for each key.
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

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: do_add_to_queue = True
            if queue_to.qsize() > maxsize_queue: do_add_to_queue = False

            self.pass_tensor_to_queue(name, active_messages_counter,
                                      passed_messages_counter, queue_to,
                                      queue_from, internal_counter,
                                      do_add_to_queue,
                                      callback_transform, grad_output[0])

        return hook

    @exception_decorator
    def model_forwardpropagation_hook_wrapper(self, name, maxsize_queue, 
                                              queue_to, queue_from, active_messages_counter,
                                              passed_messages_counter,
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
        - **active_messages_counter**: Container with the total number of active messages passed to the child process for each key.
        - **passed_messages_counter**: Container with the total number of messages passed to the child process for each key.
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

            if output is None: return hook

            nonlocal do_add_to_queue
            if queue_to.qsize() == 0: do_add_to_queue = True
            if queue_to.qsize() > maxsize_queue: do_add_to_queue = False

            # Get the output tensor data 
            output_data = output
            if isinstance(output, torch.Tensor) == False:
                output_data = output[0]

            self.pass_tensor_to_queue(name, active_messages_counter,
                                      passed_messages_counter, queue_to,
                                      queue_from, internal_counter,
                                      do_add_to_queue,
                                      callback_transform, output_data)
        return hook

    @exception_decorator
    def create_or_attachto_process(self, list_names : list, unique_id_connect_to : int):
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
            # add new names to the list of active messages
            for i in range(len(list_names)):
                self.active_messages_counter[unique_id][list_names[i]] = 0
                self.passed_messages_counter[unique_id][list_names[i]] = 0
            contexts = self.contexts[unique_id]
        else:
            unique_id = self.internal_unique_id
            self.internal_unique_id += 1
            #self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            queue_to, queue_from, contexts = self.new_process(unique_id, self.target_method, self.daemon)
            # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
            # collect the process information
            self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
            # new counter for the passed names
            dict_counter_active = dict()
            dict_counter_active = {list_names[i]:0 for i in range(len(list_names))}
            self.active_messages_counter.append(dict_counter_active)
            dict_counter_passed = dict()
            dict_counter_passed = {list_names[i]:0 for i in range(len(list_names))}
            self.passed_messages_counter.append(dict_counter_passed)

        return unique_id, queue_to, queue_from, contexts

    @exception_decorator
    def check_can_track(self):
        """
        Test if all the conditions are satisfied.
        Return
        - True if can track. False, otherwise. 
        """
        if self.enabled == False:
            return False
        return True
    
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
        if not self.check_can_track():
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
            unique_id, queue_to, queue_from, contexts = self.create_or_attachto_process(list_names, unique_id_connect_to)
            # create the hook [name of the associated tensor, and value (tensor)]
            for name, value in list_tensors.items():
                value.register_hook(self.tensor_backpropagation_hook_wrapper(name, self.max_queue_size,
                                                                             value, queue_to, queue_from, self.active_messages_counter[-1],
                                                                             self.passed_messages_counter[unique_id],
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
        if not self.check_can_track():
            return None, None, None, None

        queue_to, queue_from, contexts = None, None, None
        # Get the list of the names
        list_names = []
        for name, value in list_layers.items():
            list_names.append(name)

        # at least 1 tensor to track
        if len(list_names) > 0:
            unique_id, queue_to, queue_from, contexts = self.create_or_attachto_process(list_names, unique_id_connect_to)
            # create the hook
            for name, value in list_layers.items():
                value.register_full_backward_hook(self.layer_backpropagation_hook_wrapper(name, self.max_queue_size, 
                                                                                          queue_to, queue_from, self.active_messages_counter[-1],
                                                                                          self.passed_messages_counter[unique_id],
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
        if not self.check_can_track():
            return None, None, None, None

        #print(f'track_model:{list_models}')
        queue_to, queue_from, contexts = None, None, None
        # Get the list of the names
        list_names = []
        for name, value in list_models.items():
            list_names.append(name)

        # at least 1 tensor to track
        if len(list_names) > 0:
            unique_id, queue_to, queue_from, contexts = self.create_or_attachto_process(list_names, unique_id_connect_to)

            # create the hook
            for name, value in list_models.items():
                value.register_forward_hook(self.model_forwardpropagation_hook_wrapper(name, self.max_queue_size,
                                                                                       queue_to, queue_from, 
                                                                                       self.active_messages_counter[unique_id], 
                                                                                       self.passed_messages_counter[unique_id],
                                                                                       callback_transform))
            return unique_id, queue_to, queue_from, contexts
        else:
            print('ParallelHandler.track_model warning:unable to hook any layer')

        return None, None, None, None
