# Neural network for predicting MNIST digits

This repository contains a neural network implemented in Python using Keras to predict handwritten digits from the MNIST dataset.

**Neural Network Architecture**
The neural network architecture used for this project is as follows:

```
model = keras.Sequential([
    keras.layers.Dense(256, input_shape=(784,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

I followed codebasics from their YouTube tutorial: [CodeBasics Neural Network For Handwritten Digits Classification](https://www.youtube.com/watch?v=iqQgED9vV7k&t=861s)

## Explanation
- The model consists of a Sequential container to which layers are added sequentially.
- The first layer is a Dense layer with 256 units and a 'relu' activation function, which takes the input shape of (784,) representing flattened 28x28 pixel images.
- A Dropout layer with a dropout rate of 0.5 is added to prevent overfitting.
- Two additional Dense layers with 128 units and 'relu' activation functions are added.
- BatchNormalization layer is added to normalize the activations of the previous layer at each batch.
- Finally, a Dense layer with 10 units and a 'softmax' activation function is added for the output layer, representing the 10 classes (digits 0-9).
- The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric.

# **Note:** This is just an example architecture and compilation. Feel free to experiment with different architectures, hyperparameters, and optimization techniques to improve model performance.

# Handwritten Digit Recognition using Neural Networks

## Overview

This repository contains a Python implementation of a neural network model for recognizing handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras.

## Project Structure

- `data/`: Contains the MNIST dataset.
- `models/`: Includes saved model weights and architectures.
- `src/`: Contains the source code for data preprocessing, model definition, training, and evaluation.
- `README.md`: This file.

## Code Implementation

The neural network model is implemented using TensorFlow and Keras. Below is an example of the model architecture and compilation:

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(256, input_shape=(784,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## Training Process and Performance Evaluation
The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. Performance metrics such as accuracy are used for evaluation. For detailed training and evaluation processes, refer to the source code in the src/ directory.

## Insights Gained and Future Improvements
Throughout the project, various insights were gained regarding neural network architectures, hyperparameters, and optimization techniques. Future improvements could include experimenting with different architectures, hyperparameters, and regularization techniques to further enhance model performance.

Feel free to explore the repository, try out the model, and contribute to its improvement!

