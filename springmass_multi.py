import numpy as np

class SpringMass:
    def __init__(self, k, b, m, d0, d1, d2, d3, d4, omega):
        a = np.array([0])
        for param in k, b, m, d0, d1, d2, d3, d4, omega:
            a = np.broadcast(a, param)
        print(a.shape)
        self.k = k
        self.b = b
        self.m = m
        self.x = np.ones(a.shape)
        self.v = np.zeros(a.shape)
        self.t = np.zeros(a.shape)

        self.d0=d0
        self.d1=d1
        self.d2=d2
        self.d3=d3
        self.d4=d4
        self.omega=omega
        
    def d(self, t):
        ft = self.omega * t
        return (
            self.d0 +
            self.d1 * np.cos(ft) + self.d2 * np.sin(ft) +
            self.d3 * np.cos(ft * 2) + self.d4 * np.sin(ft * 2)
        )

    def accel(self, x, v, t):
        # mx'' + bx' + kx = d(t)
        # x'' = (d(t) - bx' - kx) / m
        return (self.d(t) - self.b * v - self.k * x) / self.m

    def step_rk4(self, dt):
        t = self.t
        v1 = self.v
        x1 = self.x
        a1 = self.accel(self.x, self.v, t)

        v2 = v1 + (dt/2) * a1
        x2 = x1 + (dt/2) * v1
        a2 = self.accel(x2, v2, t + (dt/2))

        v3 = v1 + (dt/2) * a2
        x3 = x1 + (dt/2) * v2
        a3 = self.accel(x3, v3, t + (dt/2))

        v4 = v1 + dt * a3
        x4 = x1 + dt * v3
        a4 = self.accel(x4, v4, t + dt)

        self.v = self.v + dt * (a1 + a2 * 2 + a3 * 2 + a4) / 6
        self.x = self.x + dt * (v1 + v2 * 2 + v3 * 2 + v4) / 6
        self.t = t + dt

        return self.x