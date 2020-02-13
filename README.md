# Machine Learning Nanodegree
# Deep Learning
## Project 2: Flower Image Classifier
### By Vedavyas Kamath


### Overview
This project is to train an image classifier to recognize different species of flowers. This can be imagined as using something on a smartphone app that tells us the name of the flower camera maybe looking at. In practice we train this classifier, then export it for use in such applications. I will be using a dataset of 102 flower categories to train the model.

### Goal/Aim
To build an image classifier using PyTorch and train it with data for different species of flowers so that our model is able to predict i.e. recognize different species of flowers.


### Python Scripts
Developed 2 scripts as part of this project:

1. train.py: 
This script is responsible to create, tune and train our model on the data, and finally save it as a checkpoint so that it can be later used for prediction.

**Argument List:**
- `data_dir`: To provide directory where data present
- `--save_dir`: To provide directory where checkpoint will be saved.
- `--arch`: To provide the model architecture to be used (either of VGG13 or Alexnet)
- `--lrn`: To provide Learning rate for our model while training (Default : 0.001)
- `--hidden_units`: To specify hidden units to be added in our classifier (Default : 2048)
- `--epochs`: To specify number of epochs i.e. iterations for training the data (Default : 7)
- `--GPU`: To specify if GPU is to be used or not


2. predict.py: 
This script loads the model from saved checkpoint, after which it pre-processes the image and feeds it to the model for prediction.

**Argument List:**
- `image_dir`: To provide directory where images to be predicted are present
- `--load_dir`: To provide directory where saved checkpoint is to be loaded from
- `--GPU`: To specify if GPU is to be used or not
- `--top_k`: To specify the top 'k' number of probabilities along with its class name after prediction.
- `--category_names`: To specify mapping of categories to real names of.


### Software Requirements

This project requires **Python 3.x** and the following Python libraries installed:

- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [NumPy](http://www.numpy.org/)
- [PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html)
- [Argument Parser](https://docs.python.org/3/library/argparse.html)

