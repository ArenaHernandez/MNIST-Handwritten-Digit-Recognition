# MNIST-Handwritten-Digit-Recognition
Neural network for predicting MNIST digits
This repository contains a neural network implemented in Python using Keras to predict handwritten digits from the MNIST dataset.

*Neural Network Architecture*
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
