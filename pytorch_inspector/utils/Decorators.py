import functools
import inspect
import traceback
import sys

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
