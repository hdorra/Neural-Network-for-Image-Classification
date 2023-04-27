# Neural Network for Image Classification

This repository contains a PyTorch implementation of a simple neural network for image classification, specifically using the CIFAR-10 dataset. The neural network architecture consists of multiple fully connected layers with batch normalization and ReLU activation functions. Data augmentation techniques are also utilized to improve the model's performance.

## Dependencies

To run the code in this repository, you will need the following Python packages:

* pandas
* numpy
* matplotlib
* torch
* torchvision
* PIL
* scikit-learn
* tqdm

You can install them using the following command:
```
pip install pandas numpy matplotlib torch torchvision Pillow scikit-learn tqdm
```
## Usage

1. Clone this repository:
```
git clone https://github.com/yourusername/neural-network-image-classification.git
cd neural-network-image-classification
```

2. Run the main.py script to train and evaluate the neural network:
```
python main.py
```
## Data Augmentation

Data augmentation techniques, such as random horizontal flips and random rotations, are applied to the training data to improve the model's performance. These augmentations can help the model generalize better to unseen data.

## Training and Evaluation

The model is trained using the cross-entropy loss and the Adam optimizer. A learning rate scheduler is also used to adjust the learning rate during training. Early stopping is implemented to prevent overfitting, and the model with the best validation performance is saved.

## Results

The trained model's performance can be assessed by looking at the training and validation loss and accuracy values. The training and validation curves can be visualized using the matplotlib library.
The script will download the CIFAR-10 dataset if it is not already present and preprocess the data. It will then train the neural network using the training data and evaluate its performance on the validation data. Finally, the best model will be saved to a file named cifar_model_best_model.pt.

## Model Architecture

The neural network implemented in this repository consists of the following layers:

* Fully connected layer (input -> hidden)
* Batch normalization
* ReLU activation
* Fully connected layer (hidden -> hidden)
* Batch normalization
* ReLU activation
* Fully connected layer (hidden -> output)

The input layer size is determined by the input image size (32 x 32 x 3 for CIFAR-10), and the output layer size is determined by the number of classes (10 for CIFAR-10). The hidden layer size can be adjusted to experiment with different model complexities.
