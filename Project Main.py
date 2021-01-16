import numpy as np
import matplotlib.pyplot as mplt
import scipy.integrate as integrate
#importing required modules

#################### Constants ###########################
M_pl = 2.43*10**(18)        #Planck mass in GeV
M_pl_2 = M_pl**2            #Planck mass squared
M_pl_3 = M_pl**3            #Planck mass cubed
M_pl_4 = M_pl**4            #Planck mass to the power of 4
Omega_m0 = 0.32             #Present matter density fraction
Omega_de0 = 0.68            #Present dark energy density fraction
rho_c = 3.7468*10**(-47)    #Critical energy density in GeV^4
rho_m0 = Omega_m0 * rho_c   #Present matter energy density in GeV^4
rho_de0 = Omega_de0 * rho_c #Present dark energy energy density in GeV^4
n = 3                       #Dominant energy component (matter)
a_0 = 1                     #Present scale factor
##########################################################

#################### Input Values ########################
#lambdo = 5                 #Model parameter
z_i = 20                    #Initial redshift
z_c = 10                    #Redshift at crossover
eta = 0.5                     #Model parameter after crossover
##########################################################

################# Derived parameters #####################
a_c = 1/(z_c + 1) #Scale factor at crossover
lambdo = ((n*(2*Omega_de0 + (a_0/a_c)**3 *Omega_m0))/(2*Omega_de0))**(1/2)
#First step of calculating initial conditions; obtaining the model parameter lamda (lambdo) or the redshift at crossover, depending on choice of input
phi_c = M_pl/lambdo * (4*np.log(M_pl) - np.log(rho_de0))
#Second step; obtaining the scalar field value at crossover
a_i = 1/(z_i +1) #Initial scale factor
phi_i = M_pl/lambdo * (4*np.log(M_pl) + np.log(2*((lambdo**2)/n - 1)) - np.log((a_0/a_i)**3 * rho_m0))
#Third step; obtaining initial scalar field value, must be less than crossover field value
x_i = (((a_0/a_i)**3 * rho_m0)/(((lambdo**2)/n) -1))**(0.5)
#Final step; obtain initial rate of change for the scalar field
##########################################################

################# Deriving initial time ##################
rho_m_i = (a_0/a_i)**3 * rho_m0
#Initial matter energy density
rho_phi_i = 2 * M_pl_4 * np.exp(-lambdo*phi_i/M_pl)
#Initial scalar energy density
H_i = ((rho_m_i+rho_phi_i)/(3*M_pl_2))**(1/2)
t_i = 2/(n*H_i)
#Initial time derived with Hubble paramter
##########################################################

############### Initial values and timestep ##############
phi = phi_i
a = a_i
x = x_i
t = t_i
dt = 0.001 * t
##########################################################

################### Arrays for plotting ##################
array_a = []
array_phi = []
array_H = []
array_omega = []
array_physical_distance = []
array_luminosity_distance = []
##########################################################

############### Functions for Integrals ##################
def physical_distance(a):
    return 1/(H * a**2)
##########################################################



while a < a_0:
    ############### Pre-Transition loop ##################
    if phi < phi_c:
        scalar_potential = M_pl_4 * np.exp(-lambdo*phi/M_pl)
        energy_density_phi = x**2 /2 + scalar_potential
        pressure_phi = x**2 /2 - scalar_potential
        energy_density_m = (a_0/a)**3 * rho_m0
        H = ((energy_density_m+energy_density_phi)/(3*M_pl_2))**(1/2)
        d_z = integrate.quad(physical_distance, a, 1)
        d_L = (1/a) * d_z[0] 
        #calculating values from inputs, these will be used to update the infintesimals

        dphi = x*dt
        dx = (-6*x/(n*t) + lambdo*M_pl_3*np.exp(-lambdo*phi/M_pl)) * dt
        da = (2*a/(n*t)) * dt
        #solving scalar field equation pre-transition

        array_a.append(a)
        array_phi.append(phi)
        array_omega.append(pressure_phi/energy_density_phi)
        array_physical_distance.append(d_z[0])
        array_luminosity_distance.append(d_L)
        #placing values into arrays so they can be plotted later

        x = x + dx
        phi = phi + dphi
        a = a + da
        t = t + dt
        #progressing timestep and updating values
    ######################################################

    ############### Post-transition loop #################
    else:
        scalar_potential = M_pl_4 * np.exp(-lambdo*phi_c/M_pl) * np.exp(lambdo*eta*(phi - phi_c)/M_pl)
        energy_density_phi = x**2 /2 + scalar_potential
        pressure_phi = x**2 /2 - scalar_potential
        energy_density_m = (a_0/a)**3 * rho_m0
        H = ((energy_density_m+energy_density_phi)/(3*M_pl_2))**(1/2)
        d_z = integrate.quad(physical_distance, a, 1)
        d_L = (1/a) * d_z[0]
        #calculating values from inputs, these will be used to update the infintesimals

        dphi = x*dt
        dx = (-3*H*x - (eta*lambdo/M_pl)*scalar_potential) * dt
        da = a*H*dt
        #solving scalar field equation post-transition

        array_a.append(a)
        array_phi.append(phi)
        array_omega.append(pressure_phi/energy_density_phi)
        array_physical_distance.append(d_z[0])
        array_luminosity_distance.append(d_L)
        #placing values into arrays so they can be plotted later

        x = x + dx
        phi = phi + dphi
        a = a + da
        t = t + dt
        #progressing timestep and updating values
    ######################################################


mplt.plot(array_a, array_phi)
mplt.xlabel('Scale Factor, a')
mplt.ylabel('Phi')
mplt.legend()
mplt.show()

mplt.plot(array_a, array_omega)
mplt.xlabel('Scale Factor, a')
mplt.ylabel('Barotrpoic Parameter, w')
mplt.legend()
mplt.show()

mplt.plot(array_a, array_physical_distance)
mplt.xlabel('Scale Factor, a')
mplt.ylabel('Physical Distance, d(a)')
mplt.legend()
mplt.show()

mplt.plot(array_a, array_luminosity_distance)
mplt.xlabel('Scale Factor, a')
mplt.ylabel('Luminosity Distance, d_L(a)')
mplt.legend()
mplt.show()