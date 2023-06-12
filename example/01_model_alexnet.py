import inspect
import traceback
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


import sys
sys.path.append('.')
from pytorch_inspector import ParrallelHandler, ModelExplorer, DataRecorder, DataPlot

def test_training():
    print('>>>>> test_training <<<<<')
    #torch.multiprocessing.set_start_method("spawn")

    # Define the hyperparameters
    batch_size = 32 # adjust this according to your GPU memory
    num_epochs = 1 # adjust this according to your desired training time
    learning_rate = 0.01 # adjust this according to your optimization algorithm

    # Load the data and apply some transformations
    transform = transforms.Compose([
        transforms.Resize(224), # resize the image to 224x224 to make CIFAR10 image compatible with ImageNET
        transforms.ToTensor(), # convert the images to tensors
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.194, 0.2010)) # normalize the images using the mean and std of CIFAR10
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, ) # load the training data from a folder
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2) # create a data loader for the training data

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform, ) # load the test data from a folder
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2) # create a data loader for the test data

    # Define the model and load the pretrained weights
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT) # load the AlexNet model from torchvision.models
    model.eval() # set the model to evaluation mode

    # Replace the last layer of the model with a new one that matches the number of classes in your dataset
    num_classes = len(trainset.classes) # get the number of classes in your dataset
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes) # replace the last linear layer with a new one

    # Add a warning that cuda is initialized
    # Add internal unique_id
    print(f'cuda is initialized:{torch.cuda.is_initialized()}')
    dr0 = DataRecorder(shape_expected=(640,480), fps=20., maxframes=1000, path_root='output', 
                       colorBGR=(255,0,255), displayND_mode='pca')
    ph0 = ParrallelHandler(callback_onrun=dr0.tensor_plot2D, callback_onclosing=dr0.flush, 
                           frequency=20.0, timeout=120, max_queue_size=1000, target_method='spawn')
    ph = ParrallelHandler()
    ph.set_enabled(True)
    #ph = ParrallelHandler(callback=None, frequency=2.0, timeout=30.0, target_method='spawn')
    id, queue_to, queue_from, context = ph.track_model(0, {'model': model}, callback_transform=None)
    #ph.track_layer(1, {'features2_': model.features[2], 'features5_': model.features[5], 'features12_': model.features[12]})
    #ph.track_layer(2, {'avgpool_': model.avgpool})
    #ph.track_layer(3, {'classifier0_': model.classifier[0], 'classifier3_': model.classifier[3], 'classifier6_': model.classifier[6]}) # select 0,3,6
    # Input source to pass to the model, used to test the hook
    input_to_test = torch.randn(1, 3, 224, 224)

    file_path = 'output/list_valid_01_alexnet_backward.txt'
    if os.path.exists(file_path) and os.path.isfile(file_path):
        print('##################')
        print(f'load from file:{file_path}')
        list_layers_loaded = ModelExplorer.load_list_layers_name(file_path)
        list_valid_backward = ModelExplorer.get_hook_layers_fromlist(model, list_layers_loaded)
        print('##################')
        print(f'list_valid list_valid_backward:{list_valid_backward}')
        id, queue_to, queue_from, context = ph.track_layer(-1, list_valid_backward, callback_transform=None)
    else:
        list_valid_forward, list_valid_backward = ModelExplorer.get_hook_layers(model, [input_to_test])
        print('##################')
        print(f'list_valid list_valid_forward:{list_valid_forward}')
        print('##################')
        print(f'list_valid list_valid_backward:{list_valid_backward}')
        # Change the first value with:
        # -1 --> create a new process  
        # id --> attach to the created process
        id, queue_to, queue_from, context = ph.track_layer(-1, list_valid_backward, callback_transform=None)
        ModelExplorer.save_list_layers_name(list_valid_backward, file_path)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # check if GPU is available
    model.to(device) # move the model to GPU if available

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss() # use cross entropy loss for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # use stochastic gradient descent for optimization

    # Train the model
    for epoch in range(num_epochs): # loop over the epochs
        ph.set_internal_message('epoch:' + str(epoch))
        running_loss = 0.0 # initialize the running loss
        for i, data in enumerate(trainloader, 0): # loop over the batches
            inputs, labels = data # get the inputs and labels from the batch
            inputs = inputs.to(device) # move the inputs to GPU if available
            labels = labels.to(device) # move the labels to GPU if available

            optimizer.zero_grad() # zero the parameter gradients

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # compute the loss
            loss.backward() # backward pass
            optimizer.step() # update the parameters

            #hist = torch.histc(outputs, bins=100, min=torch.min(outputs).item(), max=torch.max(outputs).item())
            #DataPlot.plot_1D(hist.clone().cpu().detach(), torch.min(outputs).item(), torch.max(outputs).item(), 
            #                   fname_out='output/3.decoder_dist.png')

            running_loss += loss.item() # accumulate the loss
            if i % 200 == 199: # print statistics every 200 batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

            # Pass to the process only the last 50 batches in a set of 200.
            if i % 200 > 150:
                ph.set_pass_to_process(True)
            else:
                ph.set_pass_to_process(False)

    # Stop the processes. Since they are running as daemon, no join is done.
    ph.stop(check_is_alive = True)

    # Test the model on the test data and compute accuracy
    correct = 0 # initialize the number of correct predictions
    total = 0 # initialize the total number of predictions

    with torch.no_grad(): # disable gradient computation
        for data in testloader: # loop over the test batches
            images, labels = data # get the images and labels from the batch
            images = images.to(device) # move the images to GPU if available
            labels = labels.to(device) # move the labels to GPU if available

            outputs = model(images) # forward pass
            _, predicted = torch.max(outputs.data, 1) # get the predicted class labels
            correct += torch.sum((predicted == labels))
            total += torch.numel(labels)
        print(f'correct:{correct}/total:{total} accuracy:{correct / total}')


if __name__ == "__main__":

    # With Fork, process fork must be called before any CUDA device use/initialization
    # With Spawn, it is not a problem because it creates a new context.

    #torch.multiprocessing.set_start_method("fork")
    torch.multiprocessing.set_start_method("spawn")
    print(f'cuda is initialized:{torch.cuda.is_initialized()}')
    test_training()

