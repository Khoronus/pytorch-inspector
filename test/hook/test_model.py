# Example from:
# https://jamesmccaffrey.wordpress.com/2020/05/22/a-minimal-pytorch-complete-example/
import numpy as np
import torch as T

import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler

device = T.device("cuda:0")  # apply to Tensor or Module

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('F ==> Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    if input[0] is not None:
        print('input size:', input[0].size())
        print('output size:', output.data.size())
        print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('B ==> Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    if grad_input[0] is not None:
        print('grad_input size:', grad_input[0].size())
        print('grad_input norm:', grad_input[0].norm())
    if grad_output[0] is not None:
        print('grad_output size:', grad_output[0].size())
# -----------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(4, 7)  # 4-7-3
    self.oupt = T.nn.Linear(7, 3)
    # (initialize weights)
    #self.hid1.register_forward_hook(printnorm)
    #self.hid1.register_backward_hook(printgradnorm)

  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = self.oupt(z)  # no softmax. see CrossEntropyLoss() 
    return z

# -----------------------------------------------------------

def main():
  # 0. get started
  print("\nBegin minimal PyTorch Iris demo ")
  T.manual_seed(1)
  np.random.seed(1)
  
  # 1. set up training data
  print("\nLoading Iris train data ")

  train_x = np.array([
    [5.0, 3.5, 1.3, 0.3],
    [4.5, 2.3, 1.3, 0.3],
    [5.5, 2.6, 4.4, 1.2],
    [6.1, 3.0, 4.6, 1.4],
    [6.7, 3.1, 5.6, 2.4],
    [6.9, 3.1, 5.1, 2.3]], dtype=np.float32) 

  train_y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int_)

  print("\nTraining predictors:")
  print(train_x)
  print("\nTraining class labels: ")
  print(train_y)

  train_x = T.tensor(train_x, dtype=T.float32).to(device)
  train_y = T.tensor(train_y, dtype=T.long).to(device)

  # 2. create network
  net = Net().to(device)    # could use Sequential()
  print(vars(net))
  source = '_modules'
  print(vars(net)[source])
  net.register_forward_hook(printnorm)
  net.hid1.register_full_backward_hook(printgradnorm)
  net.oupt.register_full_backward_hook(printgradnorm)


  # 3. train model
  max_epochs = 1#00
  lrn_rate = 0.04
  loss_func = T.nn.CrossEntropyLoss()  # applies softmax()
  optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)

  print("\nStarting training ")
  net.train()
  indices = np.arange(6)
  for epoch in range(0, max_epochs):
    np.random.shuffle(indices)
    for i in indices:
      X = train_x[i].reshape(1,4)  # device inherited
      Y = train_y[i].reshape(1,)
      optimizer.zero_grad()
      oupt = net(X)
      loss_obj = loss_func(oupt, Y)
      loss_obj.backward()
      optimizer.step()
    # (monitor error)
  print("Done training ")

  # 4. (evaluate model accuracy)

  # 5. use model to make a prediction
  net.eval()
  print("\nPredicting species for [5.8, 2.8, 4.5, 1.3]: ")
  unk = np.array([[5.8, 2.8, 4.5, 1.3]], dtype=np.float32)
  unk = T.tensor(unk, dtype=T.float32).to(device) 
  logits = net(unk).to(device)
  probs = T.softmax(logits, dim=1)
  probs = probs.cpu().detach().numpy()  # allows printoptions

  np.set_printoptions(precision=4)
  print(probs)

  # 6. (save model)

  print("\nEnd Iris demo")

if __name__ == "__main__":
  main()
