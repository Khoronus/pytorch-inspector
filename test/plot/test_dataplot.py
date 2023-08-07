import cv2
import torch

import sys
sys.path.append('.')
from pytorch_inspector.utils.DataPlot import DataPlot

def test_forceplot2d(tensor):
    fig = DataPlot.tensor_forceplot2D(tensor)
    # plot the image to a numpy array
    image = DataPlot.plt2arr(fig)
    # Fix color and shape
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))

    cv2.imshow('image', image)
    cv2.waitKey(0)

if __name__ == "__main__":
    tensor = torch.rand((1,140,11,1))
    test_forceplot2d(tensor)