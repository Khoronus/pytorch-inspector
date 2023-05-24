import atexit
from typing import Any, Optional
import torch

from .MultiprocessingWriter import MultiprocessingWriter
from .DataStruct import ProcessInfoData

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
    def __init__(self, 
                 callback_onrun : Optional[Any],
                 callback_onclosing : Optional[Any],
                 frequency : float, max_elapsed_time : float):

        """
        Initialization keeps track of all the events created to close the processes once the program terminate.
        Args:
        - **callback_onrun**: Callback function on running process.
        - **callback_onclosing**: Callback function when the process terminates.
        - **frequency**: Backpropagation tensor sampling recording.
        - **max_elapsed_time**: Maximum time without receiving new data before timeout.
        """    
        self.events = []
        # Container for the counter of active messages passed between all the processes
        # A new process will create a new dictionary of valid keys and total messages passed.
        # The counter is used to allow a maximum of 1 active message to be 
        # passed to a child process at time.
        self.active_messages_counter = []

        self.max_elapsed_time = max_elapsed_time
        self.frequency = frequency
        self.callback_onrun = callback_onrun
        self.callback_onclosing = callback_onclosing

        # Internal message passed to all the active processes
        self.internal_message = 'empty'
        # Pass the data to the active processes?
        self.pass_to_process = True 
        # Container with the create process information
        self.container_process_info = dict()
        # Unique interna ID for new created process (NO PID)
        self.internal_unique_id = 0

        try:
            atexit.register(self.stop)
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

    def __del__(self):
        """
        Destructor tries to stop running processes.
        Note: The processes should be terminated if run as fork, since are of type daemon.
        """    
        print('ParallelHandler.__del__')
        self.stop()

    def stop(self) -> None:
        """
        Stop running processes.
        """    
        try:
            print(f'ParallelHandler.stop:{self.events}')
            for event in self.events:
                if event is not None:
                    event.set()
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

    def parallel_start_process_spawn(self, rank, args):
        """
        Start a process spawn.
        Args:
        - **rank**: Rank of the process.
        - **args**: Arguments to initialize the MultiprocessingWriter.
        Note:
        Spawn uses picle to serialize the arguments and pass them to the child processes.
        However, pickle cannot serialize daemon processes, so they are ignored by spawn.
        If the obj.daemon is True, it will never start and the program will run
        without running the code.
        """    
        print(f'start_process[{rank}]_spawn args:{args}')
        try:
            # Create the process object inside the function
            obj = MultiprocessingWriter(*args)
            obj.start()
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

    def parallel_start_process_fork(self, args):
        """
        Start a process fork.
        Args:
        - **args**: Arguments to initialize the MultiprocessingWriter.

        Returns:
        - The process
        """    
        try:
            print(f'start_process args:{args}')
            # Create the process object inside the function
            obj = MultiprocessingWriter(*args)
            obj.daemon = True
            obj.start()
            return obj
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
     

    def new_process(self, unique_id):
        """
        Start a new process. The event, queue, and method are automatically initialized/selected.
        Args:
        - **unique_id**: Process unique identifier

        Returns:
        - queue for the communication, context to track the process status.
        """    
        event = torch.multiprocessing.Event()
        # Create a queue object to share data between processes
        # Important : The to/from must be reversed respect the process
        # to -> from (main process)  
        # from <- to (child process)
        queue_to = torch.multiprocessing.Queue()
        queue_from = torch.multiprocessing.Queue()

        # get the current start method
        method = torch.multiprocessing.get_start_method()
        print(f'current start method:{method}')

        # Pass the arguments for creating the process object as a tuple
        contexts = []
        args = (unique_id, event, queue_to, queue_from, self.max_elapsed_time, self.callback_onrun, self.callback_onclosing)
        if method == 'spawn':
            # Known issue github.com/pytorch/pytorch/issues/30461
            # torch.multiprocessing.spawn fails when join=False
            context = torch.multiprocessing.spawn(self.parallel_start_process_spawn, args=(args,), nprocs=1, join=False)
            # Wait a message from the process
            print('ParallelHandler.new_process:wait for the child process to be ready')
            data_from_process = queue_from.get()    # necessary for synchronizatino with spawn process
            print('ParallelHandler.new_process:child process is ready')
            #import time
            #time.sleep(3)  # <<< change this with some better solution. 2 ways queue?
        elif method == 'fork':
            if torch.cuda.is_initialized():
                raise Exception('ParallelHandler.new_process: cannot fork if cuda is initialized. Please call before any cuda call (i.e. to(device)).')
            context = self.parallel_start_process_fork(args=args)
        else:
            raise Exception(f'ParallelHandler.new_process: Method [{method}] is not supported (only spawn/fork).')
        contexts.append(context)

        self.events.append(event)
        return queue_to, queue_from, contexts

    def tensor_backpropagation_hook_wrapper(self, name, maxsize_queue, x, queue_to, queue_from, active_messages_counter):
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
 
        Returns:
        - hook for the backpropagation
        """    
        # register a tensor hook
        #print(f'tensor_backpropagation_hook_wrapper:{name}')
        # It counts how many times this hook has been called
        internal_counter = {name : 0}
        active_messages = {name : 0}
        # Add an element to the queue only if the condition is true
        do_add_to_queue = True
        def hook(grad):
            try:
                # check if the counter should be decreased
                content = queue_from.get_nowait()
                #print(f'content:{content} active_messages:{active_messages}')
                active_messages_counter[content[0]] -= 1
                #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                pass

            try:
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
                            info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                        message=str(internal_counter[name]), shared_data=x.cpu().clone().detach())
                            queue_to.put_nowait(info_data)    
                            #list_data = []
                            #list_data.append(name)
                            #list_data.append(self.internal_message)
                            #list_data.append(str(internal_counter[name]))
                            #list_data.append(x.cpu().clone().detach())
                            #queue_to.put_nowait(list_data)    

                    # The function adds also the gradient information
                    #if queue.qsize() > maxsize_queue:
                    #    pass
                    #else:
                    #    list_data = []
                    #    list_data.append(name + 'grad')
                    #    list_data.append(self.internal_message)
                    #    list_data.append(str(internal_counter[name]))
                    #    list_data.append(grad)#.cpu().clone().detach())
                    #    queue.put_nowait(list_data)    

                internal_counter[name] = internal_counter[name] + 1
                return
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
        return hook
    
    def layer_backpropagation_hook_wrapper(self, name, maxsize_queue, queue_to, queue_from, active_messages_counter):
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
                # check if the counter should be decreased
                content = queue_from.get_nowait()
                #print(f'content:{content} active_messages:{active_messages}')
                active_messages_counter[content[0]] -= 1
                #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                pass

            try:
                if grad_output is None:
                    return
                
                #print(f'name:{name} active_messages:{active_messages}')

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
                            info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                        message=str(internal_counter[name]), shared_data=grad_output[0].cpu().clone().detach())
                            queue_to.put_nowait(info_data)    
                            #list_data = []
                            #list_data.append(name)
                            #list_data.append(self.internal_message)
                            #list_data.append(str(internal_counter[name]))
                            #list_data.append(grad_output[0].cpu().clone().detach())
                            #queue_to.put_nowait(list_data)    
            
                internal_counter[name] = internal_counter[name] + 1
                return
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
        return hook

    def model_forwardpropagation_hook_wrapper(self, name, maxsize_queue, queue_to, queue_from, active_messages_counter):
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

        Returns:
        - hook for the forward propagation
        """    
        #print(f'model_forwardpropagation_hook_wrapper:{name}')
        # It counts how many times this hook has been called
        internal_counter = {name : 0}
        # Add an element to the queue only if the condition is true
        do_add_to_queue = True
        def hook(module, input, output):
            try:
                # check if the counter should be decreased
                content = queue_from.get_nowait()
                #print(f'content:{content} active_messages:{active_messages}')
                active_messages_counter[content[0]] -= 1
                #print(f'content:{content} type:{type(content)}')
            except Exception as e:
                pass

            try:
                if output is None:
                    return

                nonlocal do_add_to_queue
                if queue_to.qsize() == 0: 
                    #print(f'Bdo_add_to_queue:{do_add_to_queue}')
                    do_add_to_queue = True
                #print(f'Bdo_add_to_queue:{do_add_to_queue} -- {name} queue.qsize:{queue.qsize()}/{maxsize_queue}')
                if self.pass_to_process and internal_counter[name] % self.frequency == 0 and active_messages_counter[name] == 0:
                    #print(f'layer hook:{name} grad_output:{len(grad_output)}')
                    # Put the obj in the queue
                    if do_add_to_queue:
                        if queue_to.qsize() > maxsize_queue:
                            do_add_to_queue = False
                            pass
                        else:
                            active_messages_counter[name] += 1
                            #list_data = []
                            #list_data.append(name)
                            #list_data.append(self.internal_message)
                            #list_data.append(str(internal_counter[name]))
                            #list_data.append(output.cpu().clone().detach())
                            #queue_to.put_nowait(list_data)    
                            info_data = ProcessInfoData(name=name, internal_message=self.internal_message, 
                                                        message=str(internal_counter[name]), shared_data=output.cpu().clone().detach())
                            queue_to.put_nowait(info_data)    

                internal_counter[name] = internal_counter[name] + 1
                return
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
        return hook

    def track_tensor(self, unique_id_connect_to, list_tensors):#, tensors_hook_mode):
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
        Returns:
        - the given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        try:
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
                if unique_id_connect_to in self.container_process_info:
                    unique_id = self.internal_unique_id
                    queue_to = self.container_process_info[unique_id]['queue_to']
                    queue_from = self.container_process_info[unique_id]['queue_from']
                else:
                    unique_id = self.internal_unique_id
                    self.internal_unique_id += 1
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
                    queue_to, queue_from, contexts = self.new_process(unique_id)
                    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
                    # collect the process information
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}

                # new counter for the passed names
                dict_counter = dict()
                #for i in range(len(list_names)):
                #    dict_counter[list_names[i]] = 0
                dict_counter = {list_names[i]:0 for i in range(len(list_names))}
                self.active_messages_counter.append(dict_counter)

                # create the hook
                for name, value in list_tensors.items():
                    value.register_hook(self.tensor_backpropagation_hook_wrapper(name, len(list_names) * 4, value, 
                                                                                 queue_to, queue_from, self.active_messages_counter[-1]))
            else:
                print('ParallelHandler.track_tensor warning:unable to hook any tensor')

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

        return unique_id, queue_to, queue_from, contexts


    def track_layer(self, unique_id_connect_to, list_layers):
        """
        Automatically creates the hook and processes for the list of layers to track.
        The layer hook only backpropagation.
        Args:
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        - **list_layers**: List of layers to track {'a_layer':a_layer,'b_layer':b_layer}

        Returns:
        - the given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        try:
            queue_to, queue_from, contexts = None, None, None
            # Get the list of the names
            list_names = []
            for name, value in list_layers.items():
                list_names.append(name)

            # at least 1 tensor to track
            if len(list_names) > 0:
                # start the new process or connect to
                if unique_id_connect_to in self.container_process_info:
                    unique_id = unique_id_connect_to
                    queue_to = self.container_process_info[unique_id]['queue_to']
                    queue_from = self.container_process_info[unique_id]['queue_from']
                else:
                    unique_id = self.internal_unique_id
                    self.internal_unique_id += 1
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
                    queue_to, queue_from, contexts = self.new_process(unique_id)
                    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
                    # collect the process information
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}

                # new counter for the passed names
                dict_counter = dict()
                #for i in range(len(list_names)):
                #    dict_counter[list_names[i]] = 0
                dict_counter = {list_names[i]:0 for i in range(len(list_names))}
                self.active_messages_counter.append(dict_counter)

                # create the hook
                for name, value in list_layers.items():
                    value.register_full_backward_hook(self.layer_backpropagation_hook_wrapper(name, len(list_names) * 2, 
                                                                                              queue_to, queue_from, self.active_messages_counter[-1]))
            else:
                print('ParallelHandler.track_layer warning:unable to hook any layer')

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

        return unique_id, queue_to, queue_from, contexts


    def track_model(self, unique_id_connect_to, list_models):
        """
        Automatically creates the hook and processes for the list of layers to track.
        The layer hook only forward propagation.
        Args:
        - **unique_id_connect_to**: Process unique identifier to connect to (if it exists).
        - **list_models**: List of models to track {'a_model':a_model,'b_model':b_model}

        Returns:
        - the given unique_id, queue for the communication (to,from), context to track the process status.
        """    
        try:

            #print(f'track_model:{list_models}')
            queue_to, queue_from, contexts = None, None, None
            # Get the list of the names
            list_names = []
            for name, value in list_models.items():
                list_names.append(name)

            # at least 1 tensor to track
            if len(list_names) > 0:
                # start the new process or connect to
                if unique_id_connect_to in self.container_process_info:
                    unique_id = unique_id_connect_to
                    queue_to = self.container_process_info[unique_id]['queue_to']
                    queue_from = self.container_process_info[unique_id]['queue_from']
                else:
                    unique_id = self.internal_unique_id
                    self.internal_unique_id += 1
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}
                    queue_to, queue_from, contexts = self.new_process(unique_id)
                    # contexts cannot be added due to the following error "cannot pickle 'weakref.ReferenceType' object"
                    # collect the process information
                    self.container_process_info[unique_id] = {'queue_to':queue_to, 'queue_from':queue_from}

                # new counter for the passed names
                dict_counter = dict()
                #for i in range(len(list_names)):
                #    dict_counter[list_names[i]] = 0
                dict_counter = {list_names[i]:0 for i in range(len(list_names))}
                self.active_messages_counter.append(dict_counter)

                # create the hook
                for name, value in list_models.items():
                    value.register_forward_hook(self.model_forwardpropagation_hook_wrapper(name, len(list_names) * 2, 
                                                                                           queue_to, queue_from, self.active_messages_counter[-1]))
            else:
                print('ParallelHandler.track_model warning:unable to hook any layer')

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

        return unique_id, queue_to, queue_from, contexts