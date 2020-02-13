#####------------------------predict.py-------------------------------------------

#importing required libraries:

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
import json
import argparse


# Defining various Arguments (optional & mandatory) that we will be giving to the script:
parser = argparse.ArgumentParser(description = "Parser for Prediction script")

parser.add_argument('image_dir', help = 'Provide path to image. (Mandatory)', type = str)

parser.add_argument('load_dir', help = 'Provide path to model checkpoint. (Mandatory)', type = str)

parser.add_argument('--GPU',
                    type = str,
                    default = 'cuda',
                    help = "Option whether to or not use GPU")

parser.add_argument ('--top_k',
                    type = int,
                    default = 3,
                    help = 'Top K most likely classes. (Optional)')

parser.add_argument ('--category_names',
                    type = str,
                    help = 'Mapping of categories to real names. JSON file name to be provided. (Optional)' )


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(file_path):
    checkpoint = torch.load(file_path)
    #Since model was saved with arch parameter, check for arch:
    # If arch value is alexnet use it:
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    # If arch value is null, use vgg13
    else:
        model = models.vgg13(pretrained = True)
    # Load various model parameters from saved checkpoint:
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['mapping']

    # As model has already been tuned, switch off model tuning
    for p in model.parameters():
        p.requires_grad = False
    return model


# Defining function that will actually process an image to make it ready for our model to predict
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image

# Defining predict function that will actually try and predict topkl probabilities and class of flower image as specified.
def predict(image_path, model, topkl, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)

    # Now to our feed forward model, we need to pass a tensor.
    # Thus, need to convert the numpy array to tensor
    if device == 'cuda':
        img = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(image).type(torch.FloatTensor)

    #As forward method is working with batches doing that we will have batch size = 1
    img = img.unsqueeze(dim =0)
    model.to(device)
    img.to(device)


    with torch.no_grad():
        # Pased image tensor to feedforward model
        model.eval()    #switching to evaluation mode so that dropout can function properly
        output = model.forward(img)

        # To get output as a probability
        output_prob = torch.exp(output)

        probs, indices = output_prob.topk(topkl)
        probs = probs.cpu()
        indices = indices.cpu()
        # COnvert both the above to numpy array:
        probs = probs.numpy()
        indices = indices.numpy()

        probs = probs.tolist()[0]
        indices = indices.tolist()[0]

        mapping = {val : key for key,val in model.class_to_idx.items()}

        classes = [mapping[item] for item in indices]
        classes = np.array(classes)

        return probs, classes


#Settigng values for data loading:
args = parser.parse_args()
file_path = args.image_dir

# Defining GPU or CPU:
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Load model from saved checkpoint:

model = load_model(args.load_dir)

#Define no of classes to be predicted. Default = 3
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 3

# Calculating probabilites and classes:
probs, classes = predict(file_path, model, nm_cl, device)

class_names = [cat_to_name[item] for item in classes]

for k in range (nm_cl):
     print("k: {}/{}.. ".format(k+1, nm_cl),
            "Flower name: {}.. ".format(class_names [k]),
            "Probability: {:.3f}..% ".format(probs [k]*100),
            )
