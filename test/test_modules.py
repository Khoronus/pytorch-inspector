"""
Test unit for modules.

This module contains the main test unit for the library modules

Author: Unknown, Alessandro Moro
Date: 2023/06/21
"""
import unittest

import sys
sys.path.append('.')
from pytorch_inspector import DataRecorder
from pytorch_inspector import ParrallelHandler

class ModulesTestUnit(unittest.TestCase):

    def test_initialization(self):
        """
        It tests if the recorder and handler can be initialized.
        """
        try:
            dr = DataRecorder(shape_expected=(640,480), fps=20., maxframes=1000, path_root='output', 
                        colorBGR=(255,0,255), displayND_mode='pca')
            ph = ParrallelHandler(callback_onrun=dr.tensor_plot2D, callback_onclosing=dr.flush, 
                            frequency=20.0, timeout=120, max_queue_size=1000, target_method='spawn')
        except Exception:
            self.fail("test_initialization raised an error")

if __name__ == "__main__":
    unittest.main()