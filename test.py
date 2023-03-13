import numpy as np
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, TimeDistributed
from keras.models import Sequential
from keras import activations


model = Sequential([
    Conv2D(filters=6, kernel_size=7, strides=3, input_shape=(96, 96, 1)),
    Activation(activations.leaky_relu),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=12, kernel_size=4),
    Activation(activations.leaky_relu),
    MaxPooling2D(pool_size=2)
])
model2 = Sequential([
    TimeDistributed(model, input_shape=(3, 96, 96, 1)),
    Flatten(),
    Dense(216),
    Activation(activations.leaky_relu),
    Dense(4)
])

a = np.zeros((2, 2, 2))
a[0] = [[2, 2], [2, 2]]
print(a)
a[[0, 1]] = np.array([[2, 1], [1, 3]])
print(a)


