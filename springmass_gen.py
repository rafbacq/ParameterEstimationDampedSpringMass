import math
import numpy as np
import springmass_multi

def gen_data(n_systems,
             input_size=400,
             end_time=60,
             b_range=(0.04, 20),
             d0_range=(-1, 1),
             d1_range=(-1, 1),
             d2_range=(-1, 1),
             d3_range=(-1, 1),
             d4_range=(-1, 1),
             omega_range=(0.075, 0.5)):
    #k = 1.0
    #m = 1.0
    k=38700.000 #--> vehicle
    m= 3108.9221
    
    
    dt = 0.01
    steps = int(end_time / dt)
    downsample_rate = math.ceil(int(steps / input_size))

    # generate parameter lists for set of systems
    rng = np.random.default_rng(seed=42)
    b_values = rng.uniform(low=math.sqrt(b_range[0]), high=math.sqrt(b_range[1]), size=n_systems) ** 2
    d0_values = rng.uniform(low=d0_range[0], high=d0_range[1], size=n_systems)
    d1_values = rng.uniform(low=d1_range[0], high=d1_range[1], size=n_systems)
    d2_values = rng.uniform(low=d2_range[0], high=d2_range[1], size=n_systems)
    d3_values = rng.uniform(low=d3_range[0], high=d3_range[1], size=n_systems)
    d4_values = rng.uniform(low=d4_range[0], high=d4_range[1], size=n_systems)
    omega_values = rng.uniform(low=omega_range[0], high=omega_range[1], size=n_systems)

    params = np.array([b_values, d0_values, d1_values, d2_values, d3_values, d4_values, omega_values]).transpose()
    system_set = springmass_multi.SpringMass(
        k, b_values, m, d0_values, d1_values, d2_values, d3_values, d4_values, omega_values)

    x_data = []
    for i in range(steps):
        if i % (steps / 100) == 0:
            print(i)
        x_data.append(system_set.step_rk4(dt))
    x = np.array(x_data[::downsample_rate]).transpose()
    y = params
    return x, y

def save_data(file_prefix, x, y):
    np.save(file_prefix + "_x.npy", x)
    np.save(file_prefix + "_y.npy", y)

def load_data(file_prefix):
    x = np.load(file_prefix + "_x.npy")
    y = np.load(file_prefix + "_y.npy")
    return x, y
