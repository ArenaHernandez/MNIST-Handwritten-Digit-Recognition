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

# **Explanation:**
# - The model consists of a **Sequential** container to which layers are added sequentially.
# - The first layer is a **Dense** layer with 256 units and a '**relu**' activation function, which takes the input shape of **(784,)** representing flattened 28x28 pixel images.
# - A **Dropout** layer with a dropout rate of **0.5** is added to prevent overfitting.
# - Two additional **Dense** layers with 128 units and '**relu**' activation functions are added.
# - **BatchNormalization** layer is added to normalize the activations of the previous layer at each batch.
# - Finally, a **Dense** layer with 10 units and a '**softmax**' activation function is added for the output layer, representing the 10 classes (digits 0-9).
# - The model is compiled using the **Adam** optimizer, **sparse categorical crossentropy** loss function, and **accuracy** as the evaluation metric.

# **Note:** This is just an example architecture and compilation. Feel free to experiment with different architectures, hyperparameters, and optimization techniques to improve model performance.
