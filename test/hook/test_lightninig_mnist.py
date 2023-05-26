import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

ctx_fork = torch.multiprocessing.get_context('fork')
ctx_spawn = torch.multiprocessing.get_context('spawn')

class MultiprocessingTest:
    """
    Information writer for multiprocessing operation.
    """    
    # Initialize the class with a file name
    def __init__(self):
        super().__init__() # Call the parent class constructor

    def run(self) -> None:
        print('MultiprocessingTest.run')
        import time
        iterations = 0
        while iterations < 100:
            time.sleep(0.1)
            iterations += 1

            print('If fork this line will be printed')
            tt = torch.randn(200,300, requires_grad=False)
            print('If fork the nex line will cause a deadlock')
            minval=torch.min(tt)
            print(f'tt.minval:{minval}')

class MultiprocessingTestFork(ctx_fork.Process, MultiprocessingTest):
    # Initialize the class with a file name
    def __init__(self):
        super().__init__() # Call the parent class constructor

    def run(self) -> None:
        MultiprocessingTest.run(self)

class MultiprocessingTestSpawn(ctx_spawn.Process, MultiprocessingTest):
    # Initialize the class with a file name
    def __init__(self):
        super().__init__() # Call the parent class constructor

    def run(self) -> None:
        MultiprocessingTest.run(self)

class ParrallelHandlerTest():
    def __init__(self):
        super().__init__() # Call the parent class constructor

    def parallel_start_process(self, args, start_method):
        try:
            # Create the process object inside the function
            if start_method == 'fork':
                obj = MultiprocessingTestFork(*args)
                print(f'WARNING: Method {start_method} may cause a deadlock if run with lightning-gpu or other CUDA process. Spawn is recommended.')
            elif start_method == 'spawn':
                obj = MultiprocessingTestSpawn(*args)
            else:
                raise ValueError('Invalid choice')
            obj.daemon=True
            obj.start()
            # Naive synchronization for the spawn process...
            import time
            time.sleep(3)
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

    def new_process(self):
        print('new_process')
        # Pass the arguments for creating the process object as a tuple
        contexts = []
        args = ()

        if torch.cuda.is_initialized():
            raise Exception('ParallelHandler.new_process: cannot fork if cuda is initialized. Please call before any cuda call (i.e. to(device)).')
        # DEADLOCK
        #context = self.parallel_start_process(args=args, start_method='fork')
        # WORK NORMALLY
        context = self.parallel_start_process(args=args, start_method='spawn')
        # Wait a message from the process
        print('ParallelHandler.new_process:child process is ready')

        contexts.append(context)
        return contexts

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
    ph = ParrallelHandlerTest()
    ph.new_process()

    # Start the training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":

    # With Fork, process fork must be called before any CUDA device use/initialization
    # With Spawn, it is not a problem because it creates a new context.

    torch.multiprocessing.set_start_method("fork")
    #torch.multiprocessing.set_start_method("spawn")
    #torch.set_float32_matmul_precision('medium')# | 'high')
    print(f'cuda is initialized:{torch.cuda.is_initialized()}')
    test_training()
