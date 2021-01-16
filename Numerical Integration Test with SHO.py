import numpy as np

m = 2
gamma = 0
y_0 = 3
#parameters

y = y_0
z = 0
t = 0
dt = np.pi / 100000
#Initial values and timestep

while t < (2*np.pi):
    ms = m**2
    dy = z*dt
    dz = (-gamma*z - ms*y) * dt

    z = z + dz
    y = y + dy
    t = t + dt
    print(z,y,t)