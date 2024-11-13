# Fashion MNIST Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) to classify fashion items from the Fashion MNIST dataset using PyTorch. The model classifies clothing items into 10 different categories including T-shirts, trousers, dresses, and more.

## Features

- Custom implementation of Fashion MNIST dataset loader
- CNN architecture with batch normalization and dropout
- Model checkpointing (saves best and last model states)
- Evaluation script with sample predictions
- Support for both CPU and CUDA training

## Project Structure

```
├── fashion_model.py    # Main model architecture and training script
├── evaluate.py         # Model evaluation script
└── README.md          # This file
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy

You can install the required packages using:

```bash
pip install torch torchvision numpy
```

## Model Architecture

The CNN architecture (`FashionNet`) consists of:
- 3 convolutional layers with batch normalization
- Max pooling layers
- 2 fully connected layers
- Dropout for regularization

## Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image associated with a label from 10 classes:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Usage

### Training

To train the model, run:

```bash
python fashion_model.py
```

The script will:
- Download the Fashion MNIST dataset if not present
- Train for 100 epochs
- Save checkpoints after each epoch
- Save the best model based on validation accuracy
- Print training progress and accuracy metrics

### Evaluation

To evaluate the trained model, run:

```bash
python evaluate.py
```

This will:
- Load the best saved model
- Evaluate on the test dataset
- Show overall accuracy
- Display sample predictions

## Model Checkpoints

The training process saves two types of checkpoints:
- `fashion_model_best.pt`: Model with the best validation accuracy
- `fashion_model_last.pt`: Model state after the most recent epoch

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Epoch number
- Accuracy achieved

## Performance

The model typically achieves accuracy over 90% on the test set after training.

