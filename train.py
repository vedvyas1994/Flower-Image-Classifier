#####------------------------train.py-------------------------------------------

#All library imports:
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
from collections import OrderedDict
from PIL import Image
import argparse
import json

# Defining various Arguments that we will be giving to the script:
parser = argparse.ArgumentParser(description = 'Parser for training script')
parser.add_argument('data_dir',
                    type = str,
                    help = 'Provide directory where data present. (Mandatory)')

parser.add_argument('--save_dir',
                    type = str,
                    help = 'Provide directory to save our checkpoint. (Optional)')

parser.add_argument('--arch',
                    type = str,
                    default = 'alexnet',
                    help = 'Vgg13 or Alexnet can be used. (Default: Alexnet)')

parser.add_argument('--learning_rate',
                    type = float,
                    default = 0.001,
                    help = 'Learning rate for model training (Default: 0.001)')

parser.add_argument('--hidden_units',
                    type = int,
                    default = 2048,
                    help = 'Hidden units in Classifier. (Default: 2048)')

parser.add_argument('--epochs',
                    type = int,
                    default = 7,
                    help = 'Number of epochs. (Default: 7)')

parser.add_argument('--GPU',
                    type = str,
                    default = 'cuda',
                    help = "Option whether to or not use GPU")

#setting values for data loading
args = parser.parse_args()


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Logic control depending upon GPU argument specified by user:
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

# Data Loading:
if data_dir:   # Program will load model only if mandatory argument data directory is specified
    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    valid_data_transforms = transforms.Compose([ transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    test_data_transforms = transforms.Compose([ transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)

# Label Mapping with categories:
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#len(cat_to_name)

# Defining function load_model() that will be used to select the model depending upon inputs given:
def load_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        #If hidden_units provided:
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        #if Hiddien_units not specified:
        else:
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    #If arch NOT specified as 'vgg13', use default Alexnet
    else:
        arch = 'alexnet'
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        # If Hidden_units specified:
        if hidden_units:
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        # IF not given:
        else:
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))

    model.classifier = classifier
    return model, arch


#Defining validtion() function that will be used while training:
def validation(model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

# Actually load model using load_model() function:
model, arch = load_model(args.arch, args.hidden_units)

# Actual Training of Model:
#Initializing criterion and Optimizer:
criterion = nn.NLLLoss()
#Now for setting up Optimizer, need to check if Learning Rate has been already specified or not:
if args.learning_rate:  #if Learning Rate provided
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
else:         #if NOT provided, use specify default
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)


model.to(device)
# Now setting up number of Epochs to be run. Check if already specified or NOT:
if args.epochs:     #if epochs give, use the same
    epochs = args.epochs
else:               #if not given, use default value = 7
    epochs = 7

print_every = 40
steps = 0

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () #where optimizer is working on classifier paramters only

        # Forward and backward passes
        outputs = model.forward (inputs) #calculating output
        loss = criterion (outputs, labels) #calculating loss
        loss.backward ()
        optimizer.step () #performs single optimization step

        running_loss += loss.item () # loss.item () returns scalar value of Loss function

        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("{} of {} Epochs.. ".format(e+1, epochs),
                  "Training Loss is: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss is: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy is: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0

            # Make sure training is back on
            model.train()

#Now that model has been created, save it
#switching back to normal CPU mode, as no need of GPU to save model
model.to('cpu')

#Check mapping between class name & predicted class before savig
model.class_to_idx = train_image_datasets.class_to_idx

# Create dictionary to be saved with all info:
checkpoint = {'classifier': model.classifier,
               'state_dict': model.state_dict(),
               'arch' : arch,
               'mapping' : model.class_to_idx  }

# Save using dictionary created above:
# Check if optional argument for saving directory given
if args.save_dir:  # If saving directory given, use it
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:              # If not given, use default
    torch.save (checkpoint, 'checkpoint.pth')
