"""
Collection of functions and classes for operations on dictionaries.

Author: Unknown, Alessandro Moro
Date: 2023/06/21
"""

class DictOp:
    """
    Container with operations on dictionary
    """
    def split_dict(input_dict : dict, n : int) -> list:
        """
        Split a dictionary in n dictionaries.
        Args:
        - **input_dict**: Input dictionary.
        - **n**: Number of dictionaries expected in output
        Return
        - List of dictionaries. 
        """
        items = list(input_dict.items())
        return [dict(items[i::n]) for i in range(n)]
    
    def combine_args_kwargs(*args, **kwargs):
        """
        Combine args and kwargs
        Args:
        - **args**: List of arguments.
        - **kwargs**: Dictionary of arguments.
        Return
        - Combined dictionary.
        """
        all_args = {}
        for i, arg in enumerate(args):
            all_args[f"arg{i}"] = arg
        all_args.update(kwargs)
        return all_args