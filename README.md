# AI Flower Species Classifier

This repository contains a project that trains an AI model to classify flower species using a deep learning image classifier. The model is built using PyTorch and trained on a dataset containing images of 102 flower species. After training, the model can be used to predict the species of flowers in new images.

## Project Overview

This project is broken down into several key stages:

1. **Loading and Preprocessing the Data**
   - The dataset consists of images of flowers categorized into 102 species.
   - The dataset is divided into three sets:
     - **Training set**: Used to train the model.
     - **Validation set**: Used to tune model hyperparameters.
     - **Testing set**: Used to evaluate the final performance of the model.
   - The images are preprocessed by resizing to 224x224 pixels, normalized to the pre-trained network’s requirements, and augmented with random transformations such as rotation and horizontal flips.

2. **Building the Image Classifier**
   - A pre-trained model from the `torchvision.models` library (VGG16) is used as the feature extractor.
   - A custom fully connected neural network is added to classify the images into flower species.
   - The model is trained with backpropagation using a cross-entropy loss function and Adam optimizer.

3. **Training the Classifier**
   - The classifier is trained on the dataset, with regular evaluation on the validation set to tune hyperparameters.
   - Training continues until the accuracy is satisfactory (~70% validation accuracy is expected).

4. **Testing the Model**
   - After training, the model is tested on a holdout test set to verify its accuracy on unseen data.
   - The model should reach about 70% accuracy on the test set.

5. **Saving the Model**
   - Once the model is trained, it can be saved as a checkpoint for later use.
   - The model can be reloaded and used for inference or further training.

6. **Making Predictions**
   - A command-line interface or notebook function can be used to predict the class of a new flower image.
   - The prediction provides the top K classes along with the associated probabilities.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- Matplotlib
- Numpy
- JSON

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/flower-classifier.git
