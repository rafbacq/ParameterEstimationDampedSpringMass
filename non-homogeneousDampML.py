import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import plotext as plt
import math
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as mpl
import sklearn
from sklearn.model_selection import train_test_split

from scipy.fft import fft, ifft


# Define training and validation data
dt = 0.5
steps = 60

# Setting random values of k  and m ;-;
k=1.0
m=1.0


class SpringMass:
    def __init__(self, k, b, m, d0, d1, d2, d3, d4, omega):
        self.k = k
        self.b = b
        self.m = m
        self.x = 1.0
        self.v = 0.0

        self.d0=d0
        self.d1=d1
        self.d2=d2
        self.d3=d3
        self.d4=d4
        self.omega=omega
        
    def d(self, t):
        omega = self.omega
        return self.d0 + self.d1 * np.cos(omega*t) + self.d2 * np.sin(omega*t) + self.d3 * np.cos(2*omega*t) + self.d4 * np.sin(2*omega*t)

    def accel(self, x, v, t):
        # mx'' + bx' + kx = d(t)
        # x'' = (d(t) - bx' - kx) / m
        return (self.d(t) - self.b * v - self.k * x) / self.m

    def step_rk4(self, t, dt):
        v1 = self.v
        x1 = self.x
        a1 = self.accel(self.x, self.v, t)

        v2 = v1 + (dt/2) * a1
        x2 = x1 + (dt/2) * v1
        a2 = self.accel(x2, v2, t + (dt/2))

        v3 = v1 + (dt/2) * a2
        x3 = x1 + (dt/2) * v1
        a3 = self.accel(x3, v3, t + (dt/2))

        v4 = v1 + dt * a3
        x4 = x1 + dt * v3
        a4 = self.accel(x4, v4, t + dt)

        self.v = self.v + dt * (a1 + a2 * 2 + a3 * 2 + a4) / 6
        self.x = self.x + dt * (v1 + v2 * 2 + v3 * 2 + v4) / 6

        return self.x


#generating values of b
def generateBValues(numVals, lowerBound=0.01, upperBound=3, randomize=False):
    if not randomize:
        values = [x * (upperBound - lowerBound) / (numVals - 1) + lowerBound for x in range(numVals)]
        random.shuffle(values)
        return values

    return [random.uniform(lowerBound, upperBound) for _ in range(numVals)]

# Create SpringMass instance and generate data
#b_values = generateBValues(2000)
#systems = [SpringMass(k, b, m) for b in b_values]
#x_train = np.array([[system.step_rk4((dt * a), dt) for a in range(steps)] for system in systems])
#y_train = np.array(b_values)

#b_values_test = generateBValues(1000, upperBound=5)
#systems_test = [SpringMass(k,b,m) for b in b_values_test]
#x_test = np.array([[system.step_rk4((dt * a), dt) for a in range(steps)] for system in systems_test])
#y_test = np.array(b_values_test)

# mx'' + bx' + kx = 0
# x'' = acceleration
# F = ma
# mx'' = acceleration force
# bx' = damping force
# kx = spring force
#
# mx'' + bx' + kx = d(t)
# d(t) = disturbance force

b_values = [0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.22, 2.50, 2.84, 3.34, 4.0, 5.0, 6.66, 10.0, 20.0, 40.0, 100.0]
#random.shuffle(b_values)
d0_values = [-1,0,1]
d1_values = [-1,0,1]
d2_values = [-1,0,1]
d3_values = [-1,0,1]
d4_values = [-1,0,1]
omega_values = [0.5, 0.7, 1.0, 1.3, 1.6]
systems = [SpringMass(k, b, m, d0, d1, d2, d3, d4, omega)
              for b in b_values
              for d0 in d0_values
              for d1 in d1_values
              for d2 in d2_values
              for d3 in d3_values
              for d4 in d4_values
              for omega in omega_values]
x_train = np.array([[system.step_rk4((dt * a), dt) for a in range(steps)] for system in systems])
y_train = np.array([[b, d0, d1, d2, d3, d4, omega]
              for b in b_values
              for d0 in d0_values
              for d1 in d1_values
              for d2 in d2_values
              for d3 in d3_values
              for d4 in d4_values
              for omega in omega_values])

d0_values_test = [-0.5, 0.5]
d1_values_test = [-0.5, 0.5]
d2_values_test = [-0.5, 0.5]
d3_values_test = [-0.5, 0.5]
d4_values_test = [-0.5, 0.5]
omega_values_test = [0.3, 0.8, 1.1, 1.4, 1.8]
b_values_test =  [round(random.uniform(0.04,2), 2) for _ in range(30)] + [round(random.uniform(2, 20),2) for _ in range(10)]

#random.shuffle(b_values)
systems_test = [SpringMass(k,b,m,d0,d1,d2,d3,d4,omega)
                for b in b_values_test
                for d0 in d0_values_test
                for d1 in d1_values_test
                for d2 in d2_values_test
                for d3 in d3_values_test
                for d4 in d4_values_test
                for omega in omega_values_test]
x_test = np.array([[system.step_rk4((dt * a), dt) for a in range(steps)] for system in systems_test])
y_test = np.array([[b, d0, d1, d2, d3, d4, omega]
              for b in b_values_test
              for d0 in d0_values_test
              for d1 in d1_values_test
              for d2 in d2_values_test
              for d3 in d3_values_test
              for d4 in d4_values_test
              for omega in omega_values_test])


# Making the model
def make_model(layer_sizes=[128,128], epochs=500, learning_rate=0.0005, batch_size=64):
    l = [layers.Dense(units=size, activation="relu", input_shape=[steps if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    # Filter out None values (since input_shape is only needed for the first layer)
    #l = [layer for layer in l if layer is not None]
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0003, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 40, factor=0.1)
    noise_layer = layers.GaussianNoise(0.08)
    model = keras.Sequential([
        noise_layer, 
        *l,
        #dropout did not help!
        #layers.Dense(units=32, activation="relu", input_shape=[1]),

        layers.Dense(units=7) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])  # puts together the model
    
    nn_history = model.fit(x_train, y_train, callbacks = [early_stopping, plateau_monitor], epochs=epochs, validation_split=0.2, batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.show() 

    print(model.evaluate(noise_layer(x_test), y_test))

    return model


#model = make_model(epochs=100, learning_rate=0.1)

# Visualizing the model
def visualize(model, inputs, expected):
    
    outputs = model.predict(inputs).flatten()
    s = sorted(zip(expected, outputs), key=lambda x: x[0])
    expected, outputs = zip(*s)
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    
    outputs = np.array(outputs)
    expected = np.array(expected)
    plt.plot(list(expected), list((outputs-expected)**2))
    plt.show()
#visualize(model, x_val, y_val)
