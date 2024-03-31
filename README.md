# Handwritten Digit Recognition using Neural Networks

## Overview
This repository contains a Python implementation of three neural network models for recognizing handwritten digits from the MNIST dataset. The models are built using TensorFlow and Keras.

## Project Structure

- `data/`: Contains the MNIST dataset.
- `models/`: Includes saved model weights and architectures.
- `src/`: Contains the source code for data preprocessing, model definition, training, and evaluation.
- `README.md`: This file.

## Model 1

This model consists of a simple architecture with one hidden layer.

### Architecture
- The model starts with a Dense layer of 256 units and a 'relu' activation function, taking input shape (784,) representing flattened 28x28 pixel images.
- A final Dense layer with 10 units and a 'softmax' activation function is added for the output layer, representing the 10 classes (digits 0-9).

### Compilation
- The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric.

### Training
- The model is trained with the specified configuration for 10 epochs.

```python
model1 = Sequential([
    Dense(256, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax')
])
model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.fit(X_train_flattened, y_train, epochs=10)
```

## Model 2

This model includes dropout regularization to prevent overfitting and batch normalization for improved convergence.

### Architecture
- The model starts with a Dense layer of 256 units and a 'relu' activation function, taking input shape (784,) representing flattened 28x28 pixel images.
- A Dropout layer with a dropout rate of 0.5 is added after the first Dense layer. Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting by forcing the model to learn redundant representations.
- Two additional Dense layers with 128 units and 'relu' activation functions are added.
- BatchNormalization layer is added after the third Dense layer to normalize the activations of the previous layer at each batch, which helps stabilize and accelerate the training process.

### Output Layer
- Finally, a Dense layer with 10 units and a 'softmax' activation function is added for the output layer, representing the 10 classes (digits 0-9).

### Compilation
- The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric.

### Training
- The model is trained with the specified configuration for 20 epochs, with a validation split of 20% to monitor the model's performance on unseen data during training.

```python
model2 = keras.Sequential([
    keras.layers.Dense(256, input_shape=(784,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2.fit(X_train_flattened, y_train, epochs=20, validation_split=0.2)
```

## Model 3

This model incorporates more advanced techniques like learning rate adjustment and early stopping.

### Architecture
- The model starts with a Dense layer of 256 units and a 'relu' activation function, taking input shape (784,) representing flattened 28x28 pixel images.
- A Dropout layer with a dropout rate of 0.5 is added after the first Dense layer to prevent overfitting.
- Two additional Dense layers with 128 units and 'relu' activation functions are added.
- BatchNormalization layer is added after the third Dense layer to normalize the activations of the previous layer at each batch, which helps stabilize and accelerate the training process.
- Finally, a Dense layer with 10 units and a 'softmax' activation function is added for the output layer, representing the 10 classes (digits 0-9).

### Compilation
- The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric.

### Training
- The model is trained for 20 epochs, with a validation split of 20% to monitor the model's performance on unseen data during training.

```python
model3 = Sequential([
    Dense(256, input_shape=(784,), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

optimizer = Adam(lr=0.001)  # Adjust learning rate
model3.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),  # Early stopping to prevent overfitting
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)  # Reduce learning rate on plateau
]

history = model3.fit(X_train_flattened, y_train, 
                    epochs=20, 
                    validation_split=0.2, 
                    callbacks=callbacks,
                    verbose=1)
```

## Note:
These models are just an example of architecture and compilation. Feel free to experiment with different architectures, hyperparameters, and optimization techniques to improve model performance. You are welcome to explore the repository, try out the model, and contribute to its improvement!


## Credits
I followed [codebasics'](https://www.youtube.com/@codebasics) YouTube tutorial for Neural Network For Handwritten Digits Classification: [link here](https://www.youtube.com/watch?v=iqQgED9vV7k&t=861s)



