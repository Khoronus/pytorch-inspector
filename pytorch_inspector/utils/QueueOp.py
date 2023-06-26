"""
Collection of functions and classes for operations on queue.

Author: Unknown
Date: 2023/06/21
"""

from typing import Any, Optional

class QueueOp:
    """
    Operation on Queue. 
    """
    @staticmethod
    def clear(q : Optional[Any]) -> None:
        """
        Args:
        - **q**: Queue object
        Empty a queue by calling the get function.
        """
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass