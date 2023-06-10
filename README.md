# Siamese Network for Digit Recognition

This code implements a Siamese Convolutional Neural Network (CNN) in PyTorch for digit recognition using the MNIST dataset. The network takes two images of digits as input and outputs 1 if both images correspond to the same digit and 0 otherwise. Only 10% of the MNIST training dataset and 10% of the test dataset are used for training and testing, respectively.

## Architecture

The Siamese network architecture consists of two main parts: the shared convolutional layers and the fully connected layers.

### Convolutional Layers:
- `self.conv1`: The first convolutional layer with 1 input channel, 8 output channels, and a kernel size of 3x3.
- `self.conv2`: The second convolutional layer with 8 input channels, 16 output channels, and a kernel size of 3x3.
- `self.pool1`: Max pooling layer with a kernel size of 2x2 and stride 2.
- `self.pool2`: Max pooling layer with a kernel size of 3x3.
 
### Fully Connected Layers:
- `self.lin1`: First fully connected layer with 144 input features and 64 output features.
- `self.lin2`: Second fully connected layer with 64 input features and 32 output features.
- `self.lin3`: Output layer with 32 input features and 10 output features.

## Dataset Preparation

The `SiamDataset` class is responsible for preparing the dataset. It reads the MNIST training data from the "mnist_train.csv" file and the corresponding labels from the "trainMNISTlabels.csv" file. Only 10% of the data is used by adjusting the dataset length and randomly selecting images for each class.

The dataset is sorted based on the labels, and the images are stored in the `self.img` dictionary, where each key corresponds to a digit class and the values are lists of images belonging to that class.

The `__getitem__` method is implemented to randomly select a digit class and two images from that class for creating a positive pair. It also selects a different digit class and images from that class to create a negative pair. The labels for the positive and negative pairs are set accordingly.

## Training and Loss Calculation

The training loop runs for a specified number of epochs. In each epoch, the training dataloader provides pairs of images and labels. The network is optimized using the Adam optimizer and the Contrastive Loss function.

For each pair of images, the network computes the outputs of the shared convolutional layers for both images. Then, the Contrastive Loss function calculates the loss for positive pairs (images from the same class) and negative pairs (images from different classes).

The total loss is the sum of the losses for positive and negative pairs. The gradients are then calculated and used to update the network's parameters.

## Evaluation and Visualization

After training, a set of random image pairs is selected from the dataset for evaluation. For each pair, the Siamese network calculates the pairwise distance between the outputs of the shared convolutional layers.

The code then visualizes the image pairs along with their labels and the calculated distances. The resulting plot shows pairs of images, their labels, and the corresponding distances. The plot is saved as "Siamese_test.png" in the specified directory.

## Instructions for Running the Code

1. Install the required dependencies: numpy, pandas, matplotlib, and torch.
2. Make sure to have the MNIST training data file "mnist_train.csv" and the label file "trainMNISTlabels.csv" in the same directory as the script.
3. Run the code using a Python interpreter.
4. The training progress and loss will be displayed
5. After training, the plot with image pairs and distances are be saved as "Siamese_test.png" in the specified directory.


## Results

![image](https://github.com/Satyapalsinh10/Siamese_CNN_for_Digit_Comparison/assets/125583562/08b4be4b-af26-4820-b93c-d09131180e67)



