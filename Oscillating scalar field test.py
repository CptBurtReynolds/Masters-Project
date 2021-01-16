import numpy as np
import matplotlib.pyplot as mplt
#importing required modules

m = 2
ms = m**2
H = 0.1*m
phi_0 = 3
a_0 = 0.01
#parameters

phi = phi_0
a = a_0
x = 0
t = 0
dt = 0.001
#Initial values and timestep

array_a = []
array_eng_dens = []
#establishing arrays for plotting figures

while t <  3:
    dphi = x*dt
    dx = (-3*H*x - ms*phi) * dt
    da = a*H*dt
    #solving scalar field equation

    eng_dens = 0.5*(x*x + ms * phi * phi)
    pressure = 0.5*(x*x - ms * phi * phi)
    #calculating relavent quantities

    array_a.append(a)
    array_eng_dens.append(eng_dens)
    #placing values into arrays so they can be plotted later

    x = x + dx
    phi = phi + dphi
    a = a + da
    t = t + dt
    #progressing timestep and updating values
    print(x,phi,a,eng_dens,pressure,t)

mplt.plot(array_a, array_eng_dens)
mplt.xlabel('Scale Factor, a')
mplt.ylabel('Energy Density, rho')
mplt.legend()
mplt.show()