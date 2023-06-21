"""
Collection of functions and classes to plot data on a figure.

This module contains the code for tensor data plot.

Author: Alessandro Moro
Date: 2023/06/21
"""
import atexit
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
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
        tensor_np = tensor_data.cpu().numpy()
        ax.plot_surface(x, y, tensor_np, cmap=cm.Spectral_r)
        return fig

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
        y = tensor_data.cpu().numpy()

        # Create a surface plot
        fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

        extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
        ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax2.plot(x,y)
        plt.tight_layout()
        return fig

    @staticmethod
    def tsne(X):
        X_flat = np.array(X).reshape(X.shape[0], -1)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        X_tsne = tsne.fit_transform(X_flat)
        return X_tsne

    @staticmethod
    def plot_tsne(tensor_data):
        X = tensor_data.numpy()
        X_tsne = DataPlot.tsne(X)
        # Create a surface plot
        fig = plt.figure(dpi=300)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        return fig

    @staticmethod
    def plot_pca(tensor_data):
        tensor = tensor_data
        # Reshape the tensor to have 2 dimensions
        tensor = tensor.reshape(tensor.size(0) * tensor.size(1), -1)
        # Center the tensor
        tensor -= tensor.mean(dim=0)
        # Compute the covariance matrix
        cov = torch.mm(tensor.t(), tensor) / (tensor.size(0) - 1)
        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eig(cov)
        eigenvalues = torch.abs(eigenvalues[:])
        # Sort the eigenvectors by decreasing eigenvalues
        _, idx = torch.sort(eigenvalues, descending=True)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        # Project the tensor onto the first two principal components
        pca_result = torch.mm(tensor, eigenvectors[:, :2].real)
        # Reshape the result to have the original dimensions
        #pca_result = pca_result.view(*tensor_data.shape[:-1], -1)
        # Create a surface plot
        fig = plt.figure(dpi=300)
        plt.scatter(pca_result[..., 0].cpu().numpy(), pca_result[..., 1].cpu().numpy())
        return fig

    @staticmethod
    def plot_pca_lowrank(tensor_data):
        # Flatten the tensor
        x = tensor_data.reshape(tensor_data.size(0), -1)
        # Compute PCA using pca_lowrank function
        U, S, V = torch.pca_lowrank(x)
        # Project the data onto the first two principal components
        x_pca = torch.mm(x - x.mean(0), V[:, :2])
        # Create a surface plot
        fig = plt.figure(dpi=300)
        plt.scatter(x_pca[:, 0].cpu().numpy(), x_pca[:, 1].cpu().numpy())
        return fig