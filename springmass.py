import numpy as np
import plotext as plt
import math

# F = ma = -bx' - kx
# a = x'' = dv/dt = F/m
# v = x' dx/dt

class SpringMass:
    def __init__(self, k, b, m):
        self.k = k
        self.b = b
        self.m = m
        self.x = 1.0
        self.v = 0.0

    def generateSignal(self, t, timesteps):
        #freqs = np.linspace(0.1, 10, timesteps) #will generate array of evenly values
        #signal = np.sum([np.sin(2 * np.pi * f * t) + np.cos(2*np.pi*f*t) for f in freqs])
        #print(signal)
        #return signal

        omega = 1
        d0=0
        d1=1
        d2=2
        d3=3
        d4=4
        t = 0

        stepSignal = (8 *np.pi)/15


        signals = [] 

        while(t != 8*3.1415):
            signals.append(d0 + d1*np.cos(omega*t) + d2* np.sin(omega*t) + d3*np.cos(2*omega*t) + d4*np.sin(2*omega*t))
            t+=stepSignal
        return signal
        

    def d(self, t): #fft algorithm on disturbance

        steps=60
        signal = self.generateSignal(t, steps)


        fft_result = fft(signal)
        

        #ifft_result = ifft(fft_result)
        return np.real(fft_result)

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

    def step_euler(self, dt):
        a = self.accel(self.x, self.v)
        self.x = self.x + dt * self.v
        self.v = self.v + dt * a

    def step_midpoint(self, dt):
        a = self.accel(self.x,self.v)
        v_m = self.v + (dt/2)*a
        x_m = self.x + (dt/2)*self.v
        a_m = self.accel(x_m, v_m)

        self.x = self.x + dt*v_m
        self.v = self.v + dt*a_m

    '''def step_rk4(self, dt):
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
    '''

    def pos(self, t):
        # returns position x(t) given t
        # mu = b / 2m
        # omega_0^2 = k / m
        # gamma = sqrt(omega_0^2 - mu^2)
        # phi = atan(-mu / gamma)
        #     = atan2(-mu, gamma)
        # 𝜙 = atan(-𝜇/𝛾)
        # a = h * sqrt(1 + (mu/gamma)^2)
        # x = a * e^(mu*t) * cos(gamma * t + phi)
        mu = self.b / (2 * self.m)
        omega_not = math.sqrt(self.k / self.m)
        if(mu**2 -omega_not**2>0):
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

def run():
    system_r = SpringMass(1, 3, 1) #k=1 b=3 m=1
    system_actual = SpringMass(1, 3, 1)

    dt = 0.1
    steps = 60
    positions_r = []
    positions_actual = [system_actual.pos(n * dt) for n in range(steps)]
    for t in range(steps):
        positions_r.append(system_r.x)
        system_r.step_rk4((dt*t), dt)

    plt.clear_data()
    plt.clear_color()
    plt.clear_figure()
    plt.plot_size(120,25)
    plt.plot(positions_r, label='runge') # blue
    #plt.plot(positions_actual, label = 'actual pos') # green
    plt.show()


#to determine damping condition need to solve for zeta
#zeta = b/(2(sqrt(mk)))
#zeta=1 -->critically damped
#0<zeta<1 --> under-damped
# zeta>1 -->overdamped 

# mu = b / 2m
# omega_0 = sqrt(k / m)
# zeta = mu / omega_0 = (b / 2m) / (sqrt(k / m))
# zeta = b / (2m * sqrt(k / m))
# zeta = b / (2 * sqrt(km))
