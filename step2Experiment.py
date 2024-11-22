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
import statistics
from scipy.fft import fft, ifft
import springmass_gen
import springmass_multi
import seaborn as sns
from keras import ops
from timeit import default_timer as timer
from tensorflow.keras.utils import register_keras_serializable



#TO-DO
#plot singular variable errors (damping, d0,d1,d2,d3,d4,omega)
#if find no special error patterns -->increase data and gaussian noise
#


# Define training and validation data
dt = 0.01 #blew up at 0.125
#blew up with 0.01
#use polynomial expansion in runge kutta to determine exact dt
input_size = 400
end_time = 60
steps = int(end_time / dt)
downsample_rate = math.ceil(int(steps / input_size))

# Setting random values of k  and m ;-;
#strut and vehicle values
k=38700.000
m=3108.9221

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
        omega = self.omega*math.sqrt(k/m)
        return self.d0 + self.d1 * np.cos((2*np.pi)*omega*t) + self.d2 * np.sin((2*np.pi)*omega*t) + self.d3 * np.cos((2*np.pi)*2*omega*t) + self.d4 * np.sin((2*np.pi)*2*omega*t)

    def accel(self, x, v, t):
        # mx'' + bx' + kx = d(t)
        # x'' = (d(t) - bx' - kx) / m
        #return (self.d(t) - self.b * v - self.k * x) / self.m
        return self.d(t) - (self.b*v) -(self.k*x)/(self.m)

    def step_rk4(self, t, dt):
        # d(t) = 0
        # m = 1
        # k = 1

        # at t = 0:
        # v1 = 0
        # x1 = 1
        v1 = self.v
        x1 = self.x
        # a1 = (-b v1 - k x1) / m
        a1 = self.accel(self.x, self.v, t)

        # v2 = v1 + (dt/2) * a1 = v1 + (dt/2) * (-b v1 - k x1) / m = v1 - (b*v1*v2 - k*x1*dt)/2m
        v2 = v1 + (dt/2) * a1
        # x2 = x1 + (dt/2) * v1 = x1 + (dt*v1)/2
        x2 = x1 + (dt/2) * v1
        # a2 = (-b v2 - k x2) = -b (v1 - (dt * b * v1 - dt * x1)) - k * x2
        a2 = self.accel(x2, v2, t + (dt/2))

        # v3 = v1 + 0.05 * 4 = 0.2
        v3 = v1 + (dt/2) * a2
        # original:
        # x3 = x1 + 0.05 * v1 = 1
        # fixed:
        # x3 = x1 + 0.05 * v2 = 1 + (-0.05 * 0.05) = .9975
        x3 = x1 + (dt/2) * v2
        # a3 = (-b v3 - k x3) / m = -(100*0.2 - 1*0.9975)/1 = -20 - .9975 = -20.9975
        a3 = self.accel(x3, v3, t + (dt/2))

        # v4 = v1 + 0.1 * -20.9975 = -2.09975
        v4 = v1 + dt * a3
        # x4 = x1 + 0.1 * 0.2 = 1.02
        x4 = x1 + dt * v3
        # a4 = (-b v4 - k x4) / m =  -(100*-2.09975 + 1.02)/1 = 208.955
        a4 = self.accel(x4, v4, t + dt)

        # v = v1 + 0.1 * (-1 + 2 * 4 + 2 * -20.9975 + 208.955) / 6 = 2.899
        self.v = self.v + dt * (a1 + a2 * 2 + a3 * 2 + a4) / 6
        # x = x1 + 0.1 * (0 + -0.05 * 2 + 0.2 * 2 + -2.09975) / 6 = 0.97
        self.x = self.x + dt * (v1 + v2 * 2 + v3 * 2 + v4) / 6
        #print(f"x: {self.x}, v: {self.v}, a1: {a1}, a2: {a2}, a3: {a3}, a4: {a4}")
        return self.x
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
# b_values= [round(random.uniform(0.04,2), 2) for _ in range(30)] + [round(random.uniform(2, 20),2) for _ in range(20)]
#plot using lograthmic scale

#x_train, y_train = springmass_gen.load_data("springmass_data_random_100k")
#x_train, y_train = springmass_gen.load_data("springmass_strut_data_random_100k")
x_train, y_train = springmass_gen.load_data("springmass_strut_5_data_random_100k")
x_test = x_train[:-20000]
y_test = y_train[:-20000]



#b_values_test =  [round(random.uniform(0.04,2), 2) for _ in range(30)] + [round(random.uniform(2, 20),2) for _ in range(10)]

"""
b_values = y_train.transpose()[0]
b_values_test = [b_values[i] + random.uniform(-0.04, 0.04) for i in range(10)]
#b_values_test = [0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.22, 2.50, 2.84, 3.34, 4.0, 5.0, 6.66, 10.0, 20.0]
d0_values_test = [-0.5, 0.5]
d1_values_test = [-0.5, 0.5]
d2_values_test = [-0.5, 0.5]
d3_values_test = [-0.5, 0.5]
d4_values_test = [-0.5, 0.5]
omega_values_test = [0.05, 0.1, 0.2, 0.275, 0.35, 0.45]

params_test = np.array(
    [[b, d0, d1, d2, d3, d4, omega]
        for b in b_values_test
        for d0 in d0_values_test
        for d1 in d1_values_test
        for d2 in d2_values_test
        for d3 in d3_values_test
        for d4 in d4_values_test
        for omega in omega_values_test])
b_system_values_test, d0_system_values_test, d1_system_values_test, d2_system_values_test, d3_system_values_test, d4_system_values_test, omega_system_values_test = params_test.transpose()
system_set_test = springmass_multi.SpringMass(
    k, b_system_values_test, m,
    d0_system_values_test,
    d1_system_values_test,
    d2_system_values_test,
    d3_system_values_test,
    d4_system_values_test,
    omega_system_values_test)

x_test = np.array([system_set_test.step_rk4(dt) for a in range(steps)][::downsample_rate]).transpose()
y_test = params_test
"""


# Making the model
def make_deep_disturbance_model(layer_sizes=[128,128,128,128,128], epochs=500, learning_rate=0.0005, batch_size=64):
    l = [layers.Dense(units=size, activation="relu", input_shape=[steps if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]


    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0003, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 40, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    model = keras.Sequential([
        noise_layer, 
        *l,
        layers.Dense(units=6) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_squared_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,1:], epochs=epochs, validation_split = 0.2, batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.show() 
    

    return model

def make_deep_damping_model(layer_sizes=[256,128,64,32,16], epochs=2000, learning_rate=0.001, batch_size=64): #trying funneling (e.g. 64,32,16)
    #idea to improve efficiency: increase learning rate
    start = timer()
    l = [layers.Dense(units=size, activation="relu", input_shape=[steps if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=100, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 50, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    model = keras.Sequential([
        noise_layer, 
        *l,
        layers.Dense(units=1) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_absolute_percentage_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,:1], epochs=epochs, validation_split = 0.2, callbacks=[early_stopping,plateau_monitor],batch_size=batch_size, verbose=1)  # trains model
    end = timer()
    print(end - start)
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE Loss')
    plt.show() 
    

    return model
    
def make_conv_disturbance_model(layer_sizes=[128,64], epochs=500, learning_rate=0.001, batch_size=64):
    l = [layers.Dense(units=size, activation="relu", input_shape=[None if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0003, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 40, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    n_filters = 64
    #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
    #pool_layer = layers.MaxPooling1D(pool_size = 2)
    '''layers.Conv1D(filters=n_filters, kernel_size=9),
        layers.MaxPooling1D(pool_size = 5),
        layers.Conv1D(filters=n_filters, kernel_size=9),
        layers.MaxPooling1D(pool_size = 5),'''
    model = keras.Sequential([
        noise_layer, 
        layers.Reshape((-1, 1)),
        #first version: only one Conv layer. Existent in modelDisturbCNN1 and modelDisturbCNN2
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #second version:2 conv layer 2 pool layer:
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #pool_layer = layers.maxPooling1dlayer(kernel_size = 2, stride=1)
       
        layers.Conv1D(filters=64, kernel_size=3),
        layers.MaxPooling1D(pool_size = 2),
        layers.Conv1D(filters=128, kernel_size=3),
        layers.MaxPooling1D(pool_size = 2),
        layers.Conv1D(filters=256, kernel_size=3),
        layers.MaxPooling1D(pool_size = 2),
        layers.Flatten(),
        *l,
        layers.Dense(units=6) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_squared_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,1:], epochs=epochs, validation_split = 0.2, batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.show() 

    return model

def make_conv_damping_model(layer_sizes=[64,32,16], epochs=1000, learning_rate=0.001, batch_size=64): #way too high right now test some examples to compare
    #expect: no way to get this high (98)?
    l = [layers.Dense(units=size, activation="relu", input_shape=[None if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0003, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 40, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    conv_layer = layers.Conv1D(filters=32, kernel_size=3)
    conv_layer2 = layers.Conv1D(filters = 64, kernel_size=3)
    model = keras.Sequential([
        noise_layer, 
        layers.Reshape((-1, 1)), #this may not work lowkey
        conv_layer,
        layers.MaxPooling1D(pool_size = 2),
        conv_layer2,
        layers.MaxPooling1D(pool_size = 2),
        layers.Flatten(),
        *l,
        layers.Dense(units=6) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_absolute_percentage_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,1:], epochs=epochs, validation_split = 0.2, batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE Loss')
    plt.show() 

    return model
def make_LSTM_damping_model(layer_sizes=[64,32,16], epochs=1000, learning_rate=0.001, batch_size=64):
    l = [layers.Dense(units=size, activation="relu", input_shape=[None if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0003, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 40, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    
    model = keras.Sequential([
        noise_layer, 
        
        #first version: only one Conv layer. Existent in modelDisturbCNN1 and modelDisturbCNN2
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #second version:2 conv layer 2 pool layer:
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #pool_layer = layers.maxPooling1dlayer(kernel_size = 2, stride=1)
        layers.Reshape((-1, 1)),
        layers.LSTM(128, return_sequences=True, activation = "tanh"),
        layers.LSTM(64, return_sequences=True, activation = "tanh"),
        layers.LSTM(32, return_sequences= True, activation = "tanh"),
        layers.LSTM(16, return_sequences=False, activation = "tanh"),
       
        *l,
        layers.Dense(units=1) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_absolute_percentage_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,:1], epochs=epochs, validation_split = 0.2, batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE Loss')
    plt.show() 

    return model

@register_keras_serializable()
def combinedError(y_true, y_estimated):
    y_errorMAPE = (abs(y_true[:,:1]-y_estimated[:,:1])/y_true[:,:1])*100
    y_errorMSE = ops.mean(ops.square(y_true[:,1:]-y_estimated[:,1:]),axis=-1)
    return y_errorMAPE*y_errorMSE

def make_deep_Overall_model(layer_sizes=[512,256,128,64,32,16], epochs=2000, learning_rate=0.001, batch_size=64): #trying funneling (e.g. 64,32,16)
    #idea to improve efficiency: increase learning rate
    l = [layers.Dense(units=size, activation="relu", input_shape=[steps if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=100, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 50, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    model = keras.Sequential([
        noise_layer, 
        *l,
        layers.Dense(units=7) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss=combinedError)  # puts together the model
    
    nn_history = model.fit(x_train, y_train, epochs=epochs, validation_split = 0.2, callbacks=[early_stopping,plateau_monitor],batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show() 
    

    return model
    
def make_LSTM_overall_model(layer_sizes=[128,64,32,16], epochs=1000, learning_rate=0.001, batch_size=64):
    l = [layers.Dense(units=size, activation="relu", input_shape=[None if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0001, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 25, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    
    model = keras.Sequential([
        noise_layer, 
        
        #first version: only one Conv layer. Existent in modelDisturbCNN1 and modelDisturbCNN2
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #second version:2 conv layer 2 pool layer:
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #pool_layer = layers.maxPooling1dlayer(kernel_size = 2, stride=1)
        layers.Reshape((-1, 1)),
        #layers.LSTM(256, return_sequences=True, activation = "tanh"),
        layers.LSTM(128, return_sequences=True, activation = "tanh"),
        layers.LSTM(64, return_sequences= True, activation = "tanh"),
        layers.LSTM(32, return_sequences=True, activation = "tanh"),
        layers.LSTM(16, return_sequences=False, activation = "tanh"),
       
        *l,
        layers.Dense(units=7) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss=combinedError)  # puts together the model
    
    nn_history = model.fit(x_train, y_train, epochs=epochs, validation_split = 0.2, callbacks=[early_stopping,plateau_monitor],batch_size=batch_size, verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show() 

    return model
def make_LSTM_disturbance_model(layer_sizes=[128,64,32], epochs=1000, learning_rate=0.001, batch_size=64):
    start = timer()
    l = [layers.Dense(units=size, activation="relu", input_shape=[None if n == 0 else layer_sizes[n - 1]]) for n, size in enumerate(layer_sizes)]

    early_stopping = callbacks.EarlyStopping(
        min_delta=0.0001, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    plateau_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 25, factor=0.1)
    noise_layer = layers.GaussianNoise(0.05)
    
    model = keras.Sequential([
        noise_layer, 
        
        #first version: only one Conv layer. Existent in modelDisturbCNN1 and modelDisturbCNN2
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #second version:2 conv layer 2 pool layer:
        #conv_layer = layers.Conv1D(filters=n_filters, kernel_size=3)
        #pool_layer = layers.maxPooling1dlayer(kernel_size = 2, stride=1)
        layers.Reshape((-1, 1)),
        #layers.LSTM(256, return_sequences=True, activation = "tanh"),
        #layers.LSTM(256, return_sequences=True, activation = "tanh"),
        layers.LSTM(128, return_sequences=True, activation = "tanh"),
        layers.LSTM(64, return_sequences= True, activation = "tanh"),
        layers.LSTM(32, return_sequences=False, activation = "tanh"),
       
        *l,
        layers.Dense(units=6) #1 unit for each output we are trying to recieve 
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    model.compile(optimizer=opt, loss='mean_squared_error')  # puts together the model
    
    nn_history = model.fit(x_train, y_train[:,1:], epochs=epochs, validation_split = 0.2, callbacks=[early_stopping,plateau_monitor],batch_size=batch_size, verbose=1)  # trains model
    end = timer()
    print(end - start)
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.show() 

    return model

# Visualizing the model
def visualizeGeneral(modelDAMPING, modelDISTURBANCE, inputs, expected):

    #inputs = [[position history... of first system],[position history... of second system]]
    #predicted & truth = [[b, d0, d1,d2,d3,d4, omega of system one], [b, d0, d1,d2,d3,d4, omega of system two]]
    
    #outputs = model.predict(inputs).flatten()
    #s = sorted(zip(expected, outputs), key=lambda x: x[0])
    #expected, outputs = zip(*s)
    #plt.clear_data()
    #plt.clear_color()
    #plt.clear_figure() 
    #plt.plot_size(80, 25)
    
    #outputs = np.array(outputs)
    #expected = np.array(expected)
    #plt.plot(list(expected), list((outputs-expected)**2))
    #plt.show()
    outputsDAMPING = modelDAMPING.predict(inputs) #array of arrays
    outputsDISTURBANCE = modelDISTURBANCE.predict(inputs)
    outputs = np.concatenate((outputsDAMPING, outputsDISTURBANCE), axis=1)
    #[[damping, d0, d1, d2, d3,d4, omega]]

    twoNormErrors = []
    infinityNorm = []
    p2Norm = []
    for system in range(len(outputs)): #caculating twoNorm
        #e = [outputs[system][n] - expected[system[n]] for n in range(len(outputs[system])]
        e = outputs[system] - expected[system]
        for error in range(len(e)):
            e[error] = e[error]**2
        twoNormErrors.append(math.sqrt(sum(e)))
    for system in range(len(outputs)):#calculating infinityNorm
        e= []
        e = np.abs(outputs[system]-expected[system])
        maxE = e.max()
        infinityNorm.append(maxE)
    for system in range(len(outputs)): #caluclating p2Norm
        p = []
        p = expected[system]
        for pNorm in range(len(p)):
            p[pNorm] = p[pNorm]**2
        p2Norm.append(math.sqrt(sum(p)))
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.scatter(list(p2Norm), list(twoNormErrors))
    plt.title("Two Norm Error over P2 Norm")
    plt.xlabel("p2Norm")
    plt.ylabel("twoNorm")
    plt.show()
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.scatter(list(p2Norm), list(infinityNorm))
    plt.title("Infinity Norm over P2 Norm")
    plt.xlabel("p2Norm")
    plt.ylabel("infinityNorm")
    plt.show()
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    twoNormOverp2Norm = [i / j for i, j in zip(twoNormErrors, p2Norm)]
    infinityNormOverp2Norm = [i / j for i, j in zip(infinityNorm, p2Norm)]
    plt.scatter(list(twoNormOverp2Norm), list(infinityNormOverp2Norm))
    plt.title("(Infinity Norm/P2 Norm) over (Two Norm Error/P2 Norm)")
    plt.xlabel("TwoNormError/P2Norm")
    plt.ylabel("InfinityNorm/P2Norm")
    plt.show()
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    infinityNormOver2Norm = [i / j for i, j in zip(infinityNorm, twoNormErrors)]
    plt.scatter(list(p2Norm), list(infinityNormOver2Norm))
    plt.title("(Infinity Norm/2Norm) over (P2 Norm)")
    plt.xlabel("P2 Norm")
    plt.ylabel("infinityNorm/2Norm")
    plt.show()
    '''
    mpl.figure(figsize=(10, 6))
    mpl.plot(p2Norm, twoNormErrors)
    mpl.xlabel('P2 Norm')
    mpl.ylabel('Two Norm Errors')
    mpl.show()

    mpl.figure(figsize=(10, 6))
    mpl.plot(p2Norm, infinityNorm)
    mpl.xlabel('P2 Norm')
    mpl.ylabel('Infinity Norm Errors')
    mpl.show()

    twoNormOverp2Norm = [i / j for i, j in zip(twoNormErrors, p2Norm)]
    infinityNormOverp2Norm = [i / j for i, j in zip(infinityNorm, p2Norm)]

    mpl.figure(figsize=(10, 6))
    mpl.plot(twoNormOverp2Norm, infinityNormOverp2Norm)
    mpl.xlabel('Two Norm / P2 Norm')
    mpl.ylabel('Infinity Norm / P2 Norm')
    mpl.show()
    '''
    

#visualize(model, x_val, y_val)

def visualizeSpecific(model, inputs, expected, typeError): #note: need to plot using matplotlib cuz this graph looks stanky asf
    outputs = np.array(model.predict(inputs))
    expected = np.array(expected)
    labels_mse = ["D0", "D1", "D2", "D3", "D4", "omega"]
    error_label = "Error"

    for i in range(len(outputs[0])):
        error = []
        true_values_plot = []
        
        # Calculate error and prepare true values for plotting
        for j in range(len(outputs)):
            if typeError == "mse":
                error.append((outputs[j][i] - expected[j][i]) ** 2)
            elif typeError == "mape":
                error.append(abs((expected[j][i] - outputs[j][i]) / expected[j][i]))
            true_values_plot.append(expected[j][i])
        
        meanError = statistics.mean(error)
        
        # Clear figure and plot
        plt.clear_figure()
        plt.clear_data()
        ylabel=""
        xlabel=""
        # Set labels based on typeError and index
        if typeError == "mse":
            ylabel = f"{error_label} of {labels_mse[i]}"
            xlabel = f"True Values of {labels_mse[i]}"
        elif typeError == "mape":
            ylabel = "Error of Damping"
            xlabel = "True Values of B"
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(true_values_plot, error)
        plt.show()           
            

        
        
        
    
                
                
                
                
            
        


    #error idea: for each value of b plot error (predicted vs expected) of differnet values 
    '''UDTruth_b =[]
    CDTruth_b = []
    ODTruth_b= []

    UDPredicted_b =[]
    CDPredicted_b = []
    ODPredicted_b = []
    for array in outputs:
        b = array[0]
        zeta = b/(2(math.sqrt(m*k)))
        if(zeta>0 and zeta<1):
            UDPredicted_b.append(zeta)
        elif(zeta==1):
            CDPredicted_b.append(zeta)
        elif(zeta>1):
            ODPredicted_b.append(zeta)

    for array in expected:
        b = array[0]
        zeta = b/(2(math.sqrt(m*k)))
        if(zeta>0 and zeta<1):
            UDTruth_b.append(zeta)
        elif(zeta==1):
            CDTruth_b.append(zeta)
        elif(zeta>1):
            ODTruth_b.append(zeta)
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    '''
    
    
    

    #graph each paramter and see what the error (mse for disturbance and mape for damping)
    #expected vs predicted
    #plot the error of the predicted value vs expected value
    #should tell you two things: tell you the total error in that particular variable, but will also give you idea of whether the vlaue of that parameter gives the most error



#model = make_model()
#visualize(model, x_test, y_test)
#printErrors(model, x_test, y_test)
