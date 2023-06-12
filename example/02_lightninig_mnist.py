import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler, DataRecorder

# Define a LightningModule
class MNISTClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # Define your model architecture
        self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
        )

    def forward(self, x):
        # Define your forward pass
        return self.model(x)

    def configure_optimizers(self):
        # Define your optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        # Define your training step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Define your validation step
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

def test_training():
    # Prepare data loaders
    transform = transforms.ToTensor()
    train_dataset = MNIST('data/', train=True, download=True, transform=transform)
    val_dataset = MNIST('data/', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create a Trainer object
    trainer = pl.Trainer(max_epochs=5, devices=1, accelerator='gpu')

    # Create a LightningModule object
    model = MNISTClassifier()

    # Add a warning that cuda is initialized
    # Add internal unique_id
    print(f'cuda is initialized:{torch.cuda.is_initialized()}')
    dr = DataRecorder(shape_expected=(640,480), fps=20., maxframes=30, path_root='output', 
                      colorBGR=(255,0,255), displayND_mode='default')
    ph = ParrallelHandler(callback_onrun=dr.tensor_plot2D, callback_onclosing=dr.flush, 
                          frequency=5.0, timeout=30, max_queue_size=1000, target_method='spawn')
    id, queue_to, queue_from, context = ph.track_model(0, {'model': model}, callback_transform=None)

    # Start the training
    trainer.fit(model, train_loader, val_loader)

    # Stop the processes. Since they are running as daemon, no join is done.
    ph.stop(check_is_alive = True)

if __name__ == "__main__":

    # With Fork, process fork must be called before any CUDA device use/initialization
    # With Spawn, it is not a problem because it creates a new context.

    torch.multiprocessing.set_start_method("fork")
    #torch.multiprocessing.set_start_method("spawn")
    #torch.set_float32_matmul_precision('medium')# | 'high')
    print(f'cuda is initialized:{torch.cuda.is_initialized()}')
    test_training()
