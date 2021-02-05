# importing required modules
import numpy as np
import matplotlib.pyplot as mplt
import scipy.integrate as integrate

# Constants
M_PL = 2.43*10**(18) # Planck mass in GeV
M_PL_2 = M_PL**2 # Planck mass squared
M_PL_3 = M_PL**3 # Planck mass cubed
M_PL_4 = M_PL**4 # Planck mass to the power of 4
OMEGA_M0 = 0.32 # Present matter density fraction
OMEGA_DE0 = 0.68 # Present dark energy density fraction
RHO_C = 3.7468*10**(-47) # Critical energy density in GeV^4
RHO_M0 = OMEGA_M0 * RHO_C # Present matter energy density in GeV^4
RHO_DE0 = OMEGA_DE0 * RHO_C # Present dark energy energy density in GeV^4
N = 3 # Dominant energy component (matter)

# Input Values
#lambdo = 5 # Model parameter
Z_I = 20 # Initial redshift
Z_C = 10 # Redshift at crossover
ETA = 0.5 # Model parameter after crossover


def main():
    """Program to ...."""

    a_0 = 1 # Present scale factor
    lambdo, phi_c, a, phi, x = derive_parameters(a_0) # Derived parameters
    t = derive_initial_time(a, lambdo, phi, a_0) # Deriving initial time

    # Initial values and timestep
    da = 0.0001 * a
    dt = (N*t/(2*a)) * da

    # Structure to store values for plottting later
    values = PlottingArrays()

    # TODO: Commnet on why you are doing this
    while a < a_0:
        if phi < phi_c:
            values, dt, x, phi, a, t = pre_transition(values, lambdo, phi, x, a_0, a, dt, t, da)
        else:
            values, dt, x, phi, a, t = post_transition(values, lambdo, phi_c, phi, x, a_0, a, dt, da, t)

    values.calculate_physical_Luminosity_distance() # TODO: Comment on why this needs to be done after loop
    plotGraphs(values) # Plot graphs from calculated values

def derive_parameters(a_0):
    """Derives initial parameters from global variables"""

    a_c = 1/(Z_C + 1) #Scale factor at crossover

    # First step of calculating initial conditions; obtaining the model parameter lamda (lambdo) or the redshift at crossover, depending on choice of input
    lambdo = ((N*(2*OMEGA_DE0 + (a_0/a_c)**3 *OMEGA_M0))/(2*OMEGA_DE0))**(1/2)

    # Second step; obtaining the scalar field value at crossover
    phi_c = M_PL/lambdo * (4*np.log(M_PL) - np.log(RHO_DE0))

    # Third step; obtaining initial scalar field value, must be less than crossover field value
    a_i = 1/(Z_I +1) #Initial scale factor
    phi_i = M_PL/lambdo * (4*np.log(M_PL) + np.log(2*((lambdo**2)/N - 1)) - np.log((a_0/a_i)**3 * RHO_M0))

    # Final step; obtain initial rate of change for the scalar field
    x_i = (((a_0/a_i)**3 * RHO_M0)/(((lambdo**2)/N) -1))**(0.5)
    
    return lambdo, phi_c, a_i, phi_i, x_i


def derive_initial_time(a_i, lambdo, phi_i, a_0):
    """Derives initial time parameters from global variables and passed in values"""

    rho_m_i = (a_0/a_i)**3 * RHO_M0 # Initial matter energy density
    rho_phi_i = 2 * M_PL_4 * np.exp(-lambdo*phi_i/M_PL) # Initial scalar energy density

    # Initial time derived with Hubble paramter
    H_i = ((rho_m_i+rho_phi_i)/(3*M_PL_2))**(1/2)
    t_i = 2/(N*H_i)
    
    return t_i

def pre_transition(values, lambdo, phi, x, a_0, a, dt, t, da):
    """Calculate .... pre-transition"""
    # TODO: What is being calculated here?

    # Calculating values from inputs, these will be used to update the infintesimals
    scalar_potential = M_PL_4 * np.exp(-lambdo*phi/M_PL)
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**3 * RHO_M0
    barotropic_parameter = pressure_phi/energy_density_phi
    H = ((energy_density_m+energy_density_phi)/(3*M_PL_2))**(1/2)
    lcdm_H = ((RHO_M0 * a**(-3) + RHO_DE0)/(3*M_PL_2))**(1/2)
    
    # Solving scalar field equation pre-transition
    dphi = x*dt
    dx = (-6*x/(N*t) + lambdo*M_PL_3*np.exp(-lambdo*phi/M_PL)) * dt
    dt = (N*t/(2*a)) * da
    
    # Placing values into arrays so they can be plotted later
    values.a.append(a)
    values.phi.append(phi)
    values.z.append((1/a) - 1)
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.lcdm_integrand.append(values.lcdm_integrand[-1] + (da/(lcdm_H * a**2)))
    values.Omega.append(energy_density_phi/(energy_density_phi+energy_density_m))
    
    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    a = a + da
    t = t + dt
    
    return values, dt, x, phi, a, t


def post_transition(values, lambdo, phi_c, phi, x, a_0, a, dt, da, t):
    """Calculate .... post-transition"""
    # TODO: What is being calculated here?

    # Calculating values from inputs, these will be used to update the infintesimals
    scalar_potential = M_PL_4 * np.exp(-lambdo*phi_c/M_PL) * np.exp(lambdo*ETA*(phi - phi_c)/M_PL)
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**3 * RHO_M0
    barotropic_parameter = pressure_phi/energy_density_phi
    H = ((energy_density_m+energy_density_phi)/(3*M_PL_2))**(1/2)
    lcdm_H = ((RHO_M0 * a**(-3) + RHO_DE0)/(3*M_PL_2))**(1/2)
    
    # Solving scalar field equation post-transition
    dphi = x*dt
    dx = (-3*H*x - (ETA*lambdo/M_PL)*scalar_potential) * dt
    dt = (1/(a*H))*da
    
    # Placing values into arrays so they can be plotted later
    values.a.append(a)
    values.phi.append(phi)
    values.z.append((1/a) - 1)
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.lcdm_integrand.append(values.lcdm_integrand[-1] + (da/(lcdm_H * a**2)))
    values.Omega.append(energy_density_phi/(energy_density_phi+energy_density_m))
    
    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    a = a + da
    t = t + dt
    
    return values, dt, x, phi, a, t


def plotGraphs(values):
    """Plots Graph from given values"""

    mplt.plot(values.z, values.phi, label = 'STFQ', color = 'r')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Phi')
    mplt.legend()
    mplt.show()

    mplt.plot(values.z, values.barotropic_parameter, label = 'STFQ', color = 'r')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Barotrpoic Parameter, w')
    mplt.legend()
    mplt.show()

    mplt.plot(values.z, values.physical_distance, label = 'STFQ', color = 'r')
    mplt.plot(values.z, values.lcdm_physical_distance, label = 'lcdm', color = 'c')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Physical Distance, d(z)')
    mplt.legend()
    mplt.show()

    mplt.plot(values.z, values.luminosity_distance, label = 'STFQ', color = 'r')
    mplt.plot(values.z, values.lcdm_luminosity_distance, label = 'lcdm', color = 'c')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Luminosity Distance, d_L(z)')
    mplt.legend()
    mplt.show()


class PlottingArrays:
    """Stores all array values needed to plot graphs"""
    def __init__(self):
        self.a = []
        self.z = []
        self.phi = []
        self.barotropic_parameter = []
        self.physical_distance_integrand = [0]
        self.lcdm_integrand = [0]
        self.physical_distance = []
        self.lcdm_physical_distance = []
        self.luminosity_distance = []
        self.lcdm_luminosity_distance = []
        self.Omega = []

    def calculate_physical_Luminosity_distance(self):
        """Calculate ...."""
        # TODO: What is being calculated here?
        # TODO: Added some comments here becuase this is a big ol chunk

        d_z_i = self.physical_distance_integrand[-1]
        self.physical_distance_integrand.pop()
        for val in self.physical_distance_integrand:
            self.physical_distance.append(d_z_i - val)

        for val_phys, val_a in zip(self.physical_distance,self.a):
            self.luminosity_distance.append(1/val_a * val_phys)

        lcdm_d_z_i = self.lcdm_integrand[-1]
        self.lcdm_integrand.pop()
        for val in self.lcdm_integrand:
            self.lcdm_physical_distance.append(lcdm_d_z_i - val)

        for val_phys, val_a in zip(self.lcdm_physical_distance,self.a):
            self.lcdm_luminosity_distance.append(1/val_a * val_phys)


if __name__ == "__main__":
    main()