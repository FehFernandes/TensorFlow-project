# TensorFlow Handwritten Digit Recognition

This project implements a simple machine learning model using TensorFlow for handwritten digit recognition based on the MNIST dataset.

## Overview

The project demonstrates a complete machine learning workflow, from data preparation to model evaluation, focusing on recognizing handwritten digits (0-9).

## Features

### Setup and Initialization
- Imports required libraries (TensorFlow, NumPy, Matplotlib)
- Creates a timestamped PDF to save results

### Data Loading
- Utilizes the MNIST dataset (handwritten digits)
- Normalizes input data to values between 0 and 1

### Data Visualization
- Displays 5 example images from the dataset
- Saves these visualizations to PDF

### Neural Network Model
- Implements a simple neural network with:
  - A flatten layer to transform 28x28 images into vectors
  - A hidden layer with 128 neurons and ReLU activation
  - An output layer with 10 neurons (one for each digit 0-9)

### Training Process
- Trains the model for 5 epochs
- Records and displays accuracy and loss metrics for each epoch
- Uses callbacks to show progress

### Training Visualization
- Generates plots showing accuracy and loss evolution
- Compares performance between training and test sets

### Model Evaluation
- Calculates final accuracy on the test set
- Shows quantitative results

### Prediction Demonstration
- Selects 5 random images from the test set
- Makes predictions and displays results visually
- Shows the actual digit, prediction, and confidence level

### Confusion Matrix
- Generates a confusion matrix to evaluate model performance
- Visualizes correct predictions and errors across different classes (digits)

### Conclusion
- Finalizes the PDF with a summary
- Displays a message confirming TensorFlow is working correctly

## Requirements

- TensorFlow
- NumPy
- Matplotlib

## Output

The script generates a timestamped PDF containing all visualizations, training metrics, and evaluation results.
