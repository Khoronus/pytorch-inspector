import atexit
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
plt.ion()
plt.ioff()

__all__ = ["DataPlot"]

class DataPlot():
    """
    Collection of static functions to visualize data.
    """    

    @staticmethod
    def plt2arr(fig, draw=True) -> np.array:
        """
        Convert a plot to numpy array.
        Args:
        - **fig**: Plot figure.
        - **draw**: Drwaw the figure to canvas?
        """
        if draw:
            fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w,h) = fig.canvas.get_width_height()
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
        return rgba_arr
    

    @staticmethod
    def tensor_plot(tensor_data, fname_out):
        x = np.linspace(0, tensor_data.shape[2], tensor_data.shape[2])
        y = np.linspace(0, tensor_data.shape[1], tensor_data.shape[1])
        x, y = np.meshgrid(x, y)

        minid = min(4, tensor_data.shape[0])

        # Create a surface plot
        fig = plt.figure(dpi=300)
        for k in range(minid):
            ax = fig.add_subplot(2, 2, k + 1, projection='3d')
            # Convert tensor to numpy array
            tensor_np = tensor_data[k].numpy()
            ax.plot_surface(x, y, tensor_np, cmap=cm.Spectral_r)

        # Save the figure as image
        plt.savefig(fname_out)

    @staticmethod
    def tensor_plot_colormesh(tensor_data, fname_out):
        x = np.linspace(0, tensor_data.shape[2], tensor_data.shape[2] + 1)
        y = np.linspace(0, tensor_data.shape[1], tensor_data.shape[1] + 1)
        x, y = np.meshgrid(x, y)

        minid = min(4, tensor_data.shape[0])

        # Create a surface plot
        fig = plt.figure(dpi=300)
        for k in range(minid):
            ax = fig.add_subplot(2, 2, k + 1)
            # Convert tensor to numpy array
            tensor_np = tensor_data[k].numpy()
            # pcolormesh needs the pixel edges for x and y
            # and with default flat shading, Z needs to be evaluated at the pixel center
            plot = ax.pcolormesh(x, y, tensor_np, cmap='RdBu', shading='flat')
            plt.colorbar(plot)
        # Save the figure as image
        plt.savefig(fname_out)

    @staticmethod
    def tensor_plot2D(tensor_data):#, fname_out):
        x = np.linspace(0, tensor_data.shape[1], tensor_data.shape[1])
        y = np.linspace(0, tensor_data.shape[0], tensor_data.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a surface plot
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # Convert tensor to numpy array
        tensor_np = tensor_data.numpy()
        ax.plot_surface(x, y, tensor_np, cmap=cm.Spectral_r)
        return fig
        # Save the figure as image
        #plt.savefig('output/test.png')

        # Save the figure as image
        #plt.savefig(fname_out)

    @staticmethod
    def tensor_plot_colormesh2D(tensor_data, fname_out):
        x = np.linspace(0, tensor_data.shape[1], tensor_data.shape[1] + 1)
        y = np.linspace(0, tensor_data.shape[0], tensor_data.shape[0] + 1)
        x, y = np.meshgrid(x, y)

        # Create a surface plot
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        # Convert tensor to numpy array
        tensor_np = tensor_data.numpy()

        # pcolormesh needs the pixel edges for x and y
        # and with default flat shading, Z needs to be evaluated at the pixel center
        plot = ax.pcolormesh(x, y, tensor_np, cmap='RdBu', shading='flat')
        plt.colorbar(plot)

        # Save the figure as image
        plt.savefig(fname_out)

    @staticmethod
    def plot_1D(tensor_data, minv, maxv):#, fname_out):
        x = np.linspace(minv,maxv,num=tensor_data.shape[0])
        y = tensor_data.numpy()

        # Create a surface plot
        fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax2.plot(x,y)
        plt.tight_layout()
        return fig
        # Save the figure as image
        #plt.savefig(fname_out)
