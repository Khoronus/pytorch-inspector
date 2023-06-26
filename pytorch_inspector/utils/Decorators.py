"""
Collection of decorators.

This module contains the main decorators used for:
exception, wrapper of multiple processing (ParallelHandler)

Author: Unknown, Alessandro Moro
Date: 2023/06/21
"""
import functools
import inspect
import traceback
import sys
from typing import Any

from pytorch_inspector.utils.DictOp import *

def exception_decorator(func):
    """
    Decorator for the exception in code. It builds a wrapper around a calling function. 
    """
    @functools.wraps(func)
    #@traceback.print_exc()
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            stack_summary = traceback.extract_tb(exc_tb)
            # check if the function is a class method
            if inspect.ismethod(func):
                # skip the first argument in the traceback
                stack_summary = stack_summary[1:]
            last_entry = stack_summary[-1]
            file_name, line_number, func_name, text = last_entry
            print(f'{func.__name__} ex occurred in {inspect.getfile(func)}, line {func.__code__.co_firstlineno}, in {func_name}')
            # use print_exc to print the formatted traceback
            traceback.print_exc()
            raise
    return wrapper


def wrapper_multiple_process_decorator(func):
    """
    Decorator to run multiple processes from a splitted list
    """
    def track_multiple_process(*args, **kwargs):
        """
        Args:
        - **list_items**: Input dictionary with the list of items to track. If not defined, it uses the second argument in the function.
        - **num_process**: Split the dictionary in N many object (if possible).
        The wrapper automatically check if the list_items is defined as dictionary object or passed as argument.
        The dictionary is splitted in N objects where N is the size of the input dictionary.
        """
        if 'list_items' in kwargs:
            input_dict=kwargs['list_items']
        else:
            input_dict=args[2]

        # min number of process 1, max len of dictionary
        num_process = 1
        if 'num_process' in kwargs: num_process = kwargs['num_process']
        if num_process > len(input_dict): num_process = len(input_dict)
        if num_process <= 0: num_process = 1
        # split and call the function
        n_list_valid_backward = DictOp.split_dict(input_dict, n=num_process)
        for l in n_list_valid_backward:
            # get only the arguments that are valid in the function
            kwargs2 = {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}
            # temporary object in case it is necessary to modify the list of items to track
            # in the args (tuple type)
            tmp_tuple = args
            # Modify the arguments with the list of items
            if 'list_items' in kwargs2:
                kwargs2['list_items'] = l
            else:
                tmp_list = list(tmp_tuple)
                tmp_list[2] = l
                tmp_tuple = tuple(tmp_list)
            res = func(*tmp_tuple, **kwargs2)
        return res
    return track_multiple_process