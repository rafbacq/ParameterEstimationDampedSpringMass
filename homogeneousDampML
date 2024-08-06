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
class SpringMass:
    def __init__(self, k, b, m):
        self.k = k
        self.b = b
        self.m = m
        self.x = 1.0
        self.v = 0.0

    def accel(self, x, v):
        return (-self.b * v - self.k * x) / self.m

    def step_rk4(self, dt):
        v1 = self.v
        x1 = self.x
        a1 = self.accel(self.x, self.v)

        v2 = v1 + (dt/2) * a1
        x2 = x1 + (dt/2) * v1
        a2 = self.accel(x2, v2)

        v3 = v1 + (dt/2) * a2
        x3 = x1 + (dt/2) * v1
        a3 = self.accel(x3, v3)

        v4 = v1 + dt * a3
        x4 = x1 + dt * v3
        a4 = self.accel(x4, v4)

        self.v = self.v + dt * (a1 + a2 * 2 + a3 * 2 + a4) / 6
        self.x = self.x + dt * (v1 + v2 * 2 + v3 * 2 + v4) / 6

    def pos(self, t):
        mu = self.b / (2 * self.m)
        omega_not = math.sqrt(self.k / self.m) # omega_not = 1 if k = 1 and m = 1
        if(mu**2 -omega_not**2>0):
            #gamma = math.sqrt(mu**2-omega_not**2)
            #alpha = mu-gamma
            #beta = mu+gamma
            #return (self.x/(2*gamma))((alpha**2)*math.exp(-alpha*t)-(beta**2)*math.exp(-beta*t))
            gamma = math.sqrt(mu**2-omega_not**2)
            alpha = mu+gamma
            beta = mu-gamma
            return (self.x/(2*gamma))*((alpha)*math.exp(-(beta)*t)-(beta)*math.exp(-alpha*t))
        elif(mu**2-omega_not**2==0):
            return self.x*(math.exp(-mu*t))+ self.x*mu*t*(math.exp(-mu*t))
        elif(mu**2-omega_not**2<0):
            gamma = math.sqrt(omega_not**2 - mu**2)
            phi = math.atan2(-mu, gamma)
            a = self.x * (math.sqrt((mu/gamma)**2 + 1))
            return a * (math.exp(-mu * t) * math.cos(gamma * t + phi))

# Define training and validation data
dt = 0.5
steps = 120
#for dt=0.5
#steps=120 -->up until 7 it good
#steps=200 -->up until 4 its good up until 12 its alright

# Setting random values of k  and m ;-;
k=1.0
m=1.0

#generating values of b
def generateBValues(numVals, lowerBound=0.01, upperBound=3, randomize=False):
    if geometric:
        lowerBound = max(lowerBound, 0.01)
        factor = (upperBound / lowerBound)**(1.0/numVals)
        values = [factor ** i for i in range(numVals)] * lowerBound
        random.shuffle(values)
        return values

    if not randomize:
        values = [x * (upperBound - lowerBound) / (numVals - 1) + lowerBound for x in range(numVals)]
        random.shuffle(values)
        return values

    return [random.uniform(lowerBound, upperBound) for _ in range(numVals)]

# Create SpringMass instance and generate data
b_values = [0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.22, 2.50, 2.84, 3.34, 4.0, 5.0, 6.66, 10.0, 20.0, 40.0, 100.0]
#random.shuffle(b_values)
systems = [SpringMass(k, b, m) for b in b_values]
x_train = np.array([[system.pos(dt * a) for a in range(steps)] for system in systems])
y_train = np.array(b_values)

b_values_test =  [round(random.uniform(0.04,2), 2) for _ in range(30)] + [round(random.uniform(2, 20),2) for _ in range(10)]
#random.shuffle(b_values)
systems_test = [SpringMass(k,b,m) for b in b_values_test]
x_test = np.array([[system.pos(dt * a) for a in range(steps)] for system in systems_test])
y_test = np.array(b_values_test)




# Making the model
def make_model(layer_sizes=[16], epochs=500, learning_rate=0.0005, batch_size=64):
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
        layers.Dense(units=1)
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)  # sets training algorithm to correct weights
    #mean_absolute_percentage_error = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')

    model.compile(optimizer=opt, loss='mean_absolute_percentage_error', metrics=['mean_absolute_percentage_error'])  # puts together the model
    
    nn_history = model.fit(
        x_train, y_train,
        #callbacks = [early_stopping, plateau_monitor],
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        verbose=1)  # trains model
    
    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(80, 25)
    plt.plot(nn_history.history['loss'], label='Training Loss')
    plt.plot(nn_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.show() 

    print(model.evaluate(noise_layer(x_test), y_test))
    tf.keras.utils.plot_model(model, to_file = "lenet.png", show_shapes = True)

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
    #plt.plot(list(expected), list((outputs-expected)**2))
    plt.plot(list(expected), list(100*(expected - outputs) / expected))#mean absolute percentage error
    plt.show()
#visualize(model, x_val, y_val)


