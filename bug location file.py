# Importing required modules
import numpy as np
import matplotlib.pyplot as mplt
import scipy.integrate as integrate

# Constants
CONSTANT_H_0 = 1.580*10**(-42) # Hubble Constant in GeV
CONSTANT_H_0_2 = CONSTANT_H_0**2 # Hubble constant squared
CONSTANT_M_pl = 2.43*10**(18) # Planck mass in GeV
CONSTANT_M_pl_2 = CONSTANT_M_pl**2 # Planck mass squared
CONSTANT_M_pl_3 = CONSTANT_M_pl**3 # Planck mass cubed
CONSTANT_M_pl_4 = CONSTANT_M_pl**4 # Planck mass to the power of 4
CONSTANT_Omega_m0 = 0.32 # Present matter density fraction
CONSTANT_Omega_de0 = 0.68 # Present dark energy density fraction
CONSTANT_rho_c = 3.7468*10**(-47) # Critical energy density in GeV^4
CONSTANT_rho_m0 = CONSTANT_Omega_m0 * CONSTANT_rho_c # Present matter energy density in GeV^4
CONSTANT_rho_de0 = CONSTANT_Omega_de0 * CONSTANT_rho_c # Present dark energy energy density in GeV^4
CONSTANT_n = 3 # Dominant energy component (matter)
CONSTANT_crossover_scale = 1.0/(CONSTANT_H_0*(1.0 - CONSTANT_Omega_de0)) # Crossover scale for DGP model

# Input Values
Z_I = 20 # Initial redshift
Z_C = 10.0 # Redshift at crossover
ETA = 0.5 # Model parameter after crossover

def main():
    """Program to model the evolution of the STFQ Dark Energy model across a redshift range."""

    a_0 = 1 # Present scale factor
    lambdo, phi_c, a, phi, x = derive_parameters(a_0) # Derived parameters for scalar field analysis and the scale factor
    t = derive_initial_time(a, lambdo, phi, a_0) # Deriving initial time

    # Initial values and timestep
    da = 0.0001 * a
    dt = (CONSTANT_n*t/(2*a)) * da

    # Growth factor analysis initial values
    G=G_LCDM=G_DGP = 1.0*a
    y=y_LCDM=y_DGP = da/dt

    # Structure to store values for plottting later
    values = PlottingArrays()
    
    # Loop for solving the scalar field equation and the density perturbation equation
    while a < a_0:
        
        # Solving the scalar field equation for STFQ and calculating Hubble parameters
        if phi < phi_c:
            values, H, H_LCDM, x, phi = pre_transition(values, lambdo, phi, x, a_0, a, da, t, dt)
        else:
            values, H, H_LCDM, x, phi = post_transition(values, lambdo, phi_c, phi, x, a_0, a, da, dt)
        # Solving density perturbation equations and updating timesteps
        values, G, y, G_LCDM, y_LCDM, G_DGP, y_DGP = perturbation_equation(values, H, H_LCDM, G, y, G_LCDM, y_LCDM, G_DGP, y_DGP, a_0, a, da, dt)
        values, a, t, dt = timestep(values, a, da, t)

    # Calculates the fractional luminosity distance for STFQ and LCDM. Done post-loop due to form of the physical distance intergral
    values.calculate_physical_Luminosity_distance() 
    # Calculates the fractional growth from the STFQ and LCDM f(z)G(z) values
    values.calculate_fractional_growth() 
    # Plot graphs from calculated values
    plotGraphs(values) 

def derive_parameters(a_0):
    """Derives initial parameters from global variables (constants and input parameters)"""

    a_c = 1/(Z_C + 1) #Scale factor at crossover

    # First step of calculating initial conditions; obtaining the model parameter lamda (lambdo) or the redshift at crossover, depending on choice of input
    lambdo = ((CONSTANT_n*(2*CONSTANT_Omega_de0 + (a_0/a_c)**3 *CONSTANT_Omega_m0))/(2*CONSTANT_Omega_de0))**(1/2)

    # Second step; obtaining the scalar field value at crossover
    phi_c = CONSTANT_M_pl/lambdo * (4*np.log(CONSTANT_M_pl) - np.log(CONSTANT_rho_de0))

    # Third step; obtaining initial scalar field value, must be less than crossover field value
    a_i = 1/(Z_I +1) #Initial scale factor
    phi_i = CONSTANT_M_pl/lambdo * (4*np.log(CONSTANT_M_pl) + np.log(2*((lambdo**2)/CONSTANT_n- 1)) - np.log((a_0/a_i)**3 * CONSTANT_rho_m0))

    # Final step; obtain initial rate of change for the scalar field
    x_i = (((a_0/a_i)**3 * CONSTANT_rho_m0)/(((lambdo**2)/CONSTANT_n) -1))**(0.5)
    
    return lambdo, phi_c, a_i, phi_i, x_i

def derive_initial_time(a_i, lambdo, phi_i, a_0):
    """Derives initial time parameters from global variables and input parameters"""

    rho_m_i = (a_0/a_i)**3 * CONSTANT_rho_m0 # Initial matter energy density
    rho_phi_i = 2 * CONSTANT_M_pl_4 * np.exp(-lambdo*phi_i/CONSTANT_M_pl) # Initial scalar energy density

    # Initial time derived with Hubble paramter
    H_i = ((rho_m_i+rho_phi_i)/(3*CONSTANT_M_pl_2))**(1/2)
    t_i = 2/(CONSTANT_n*H_i)
    
    return t_i

def pre_transition(values, lambdo, phi, x, a_0, a, da, t, dt):
    """Solves the scalar field equation pre-transition"""

    # Calculating energy densities, pressure, the barotropic parameter, and the Hubble parameter from the pre-transition scalar potential
    scalar_potential = CONSTANT_M_pl_4 * np.exp(-lambdo*phi/CONSTANT_M_pl)
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**3 * CONSTANT_rho_m0
    barotropic_parameter = pressure_phi/energy_density_phi
    H = ((energy_density_m+energy_density_phi)/(3*CONSTANT_M_pl_2))**(1/2)

    # Calculating the Hubble parameter for LCDM
    H_LCDM = ((CONSTANT_rho_m0 * a**(-3) + CONSTANT_rho_de0)/(3*CONSTANT_M_pl_2))**(1/2)

    # Solving scalar field equation pre-transition
    dphi = x*dt
    dx = (-3*H*x + (lambdo/CONSTANT_M_pl)*scalar_potential) * dt
    
    # Placing values into arrays so they can be plotted later
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.LCDM_integrand.append(values.LCDM_integrand[-1] + (da/(H_LCDM * a**2)))

    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    
    return values, H, H_LCDM, x, phi

def post_transition(values, lambdo, phi_c, phi, x, a_0, a, da, dt):
    """Solves the scalar field equation post-transition for STFQ"""

    # Calculating energy densities, pressure, the barotropic parameter, and the Hubble parameter from the post-transition scalar potential
    scalar_potential = CONSTANT_M_pl_4 * np.exp(-lambdo*phi_c/CONSTANT_M_pl) * np.exp(lambdo*ETA*(phi - phi_c)/CONSTANT_M_pl)
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**3 * CONSTANT_rho_m0
    barotropic_parameter = pressure_phi/energy_density_phi
    H = ((energy_density_m+energy_density_phi)/(3*CONSTANT_M_pl_2))**(1/2)

    # Calculating the LCDM Hubble parameter
    H_LCDM = ((CONSTANT_rho_m0 * a**(-3) + CONSTANT_rho_de0)/(3*CONSTANT_M_pl_2))**(1/2)
    
    # Solving scalar field equation post-transition
    dphi = x*dt
    dx = (-3*H*x - (ETA*lambdo/CONSTANT_M_pl)*scalar_potential) * dt

    # Placing values into arrays so they can be plotted later
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.LCDM_integrand.append(values.LCDM_integrand[-1] + (da/(H_LCDM * a**2)))

    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    
    return values, H, H_LCDM, x, phi

def perturbation_equation(values, H, H_LCDM, G, y, G_LCDM, y_LCDM, G_DGP, y_DGP, a_0, a, da, dt):
    """Solves the density perturbation equation for a variety of models"""

    # Solving density perturbation equation for STFQ
    energy_density_m = (a_0/a)**3 * CONSTANT_rho_m0
    dG = y*dt
    dy = (energy_density_m*G/(2.0*CONSTANT_M_pl_2) - 2*H*y)*dt
    # Solving density perturbation equation for LCDM
    dG_LCDM = y_LCDM*dt
    dy_LCDM = (energy_density_m*G_LCDM/(2.0*CONSTANT_M_pl_2) - 2*H_LCDM*y_LCDM)*dt
    # Calculating the Hubble parameter for DGP
    small_bombs = (CONSTANT_crossover_scale**(-2.0) + (4.0*energy_density_m)/(3.0*CONSTANT_M_pl_2))**(0.5)
    H_DGP = 0.5*(CONSTANT_crossover_scale**(-1.0) + small_bombs)
    Hdot_DGP = (-energy_density_m*H_DGP)/(CONSTANT_M_pl_2*small_bombs)
    beta = 1- 2.0*CONSTANT_crossover_scale*H_DGP*(1.0 + Hdot_DGP/(3.0*(H_DGP)**(2.0)))
    # Solving density pertubation equation for DGP
    dG_DGP = y_DGP*dt
    dy_DGP = ((energy_density_m*G_DGP/(2.0*CONSTANT_M_pl_2))*(1.0 + 1.0/(3.0*beta)) - 2*H_DGP*y_DGP)*dt

    # Appending values into arrays
    values.fG.append(a*(dG/da))
    values.fG_LCDM.append(a*(dG_LCDM/da))
    values.fG_DGP.append(a*(dG_DGP/da))

    # Updating variables
    G = G + dG
    y = y + dy
    G_LCDM = G_LCDM + dG_LCDM
    y_LCDM = y_LCDM + dy_LCDM
    G_DGP = G_DGP + dG_DGP
    y_DGP = y_LCDM + dy_DGP
    
    return values, G, y, G_LCDM, y_LCDM, G_DGP, y_DGP

def timestep(values, a, da, t):
    """Updates the timestep variables"""

    # Appending values into arrays
    values.a.append(a)
    values.z.append((1/a) - 1)

    # Updating the timesteps
    dt = (CONSTANT_n*t/(2*a)) * da
    a = a + da
    t = t + dt

    return values, a, t, dt
    
def plotGraphs(values):
    """Plots graphs from prior calculated values"""

    # Barotropic parameter against redshift
    mplt.plot(values.z, values.barotropic_parameter, label = 'STFQ', color = 'r')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Barotrpoic Parameter, w')
    mplt.legend()
    mplt.show()

    # Fractional luminosity distance against redshift
    mplt.plot(values.z, values.fractional_luminosity_distance, label = 'Fractional dL', color = 'r')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Fractional Luminosity Distance')
    mplt.legend()
    mplt.show()

    mplt.plot(values.z, values.fG, label = 'STFQ', color = 'r')
    mplt.plot(values.z, values.fG_LCDM, label = 'LCDM', color = 'c')
    mplt.plot(values.z, values.fG_DGP, label = 'DGP', color = 'g')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Fractional Growth, f(z)G(z)')
    mplt.legend()
    mplt.show()


    # Fractional Growth Factor G against redshift
    mplt.plot(values.z, values.fractional_growth, label = 'Fractional fG', color = 'r')
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Fractional Growth, f(z)G(z)')
    mplt.legend()
    mplt.show()

class PlottingArrays:
    """Stores all array values needed to plot graphs"""
    def __init__(self):
        self.a = []
        self.z = []
        self.barotropic_parameter = []
        
        self.physical_distance_integrand = [0]
        self.LCDM_integrand = [0]
        self.physical_distance = []
        self.LCDM_physical_distance = []
        self.luminosity_distance = []
        self.LCDM_luminosity_distance = []
        self.fractional_luminosity_distance = []

        self.fG = []
        self.fG_LCDM = []
        self.fG_DGP = []
        self.fractional_growth = []
        self.growth_index = []
        self.growth_index_LCDM = []
        self.growth_index_DGP = []

    def calculate_physical_Luminosity_distance(self):
        """Calculate physical and then luminosity distances numerically"""

        # Calculating the physical distance for STFQ
        d_z_i = self.physical_distance_integrand[-1]
        self.physical_distance_integrand.pop()
        for val in self.physical_distance_integrand:
            self.physical_distance.append(d_z_i - val)

        # Calculating the luminosity distance for STFQ
        for val_phys, val_a in zip(self.physical_distance,self.a):
            self.luminosity_distance.append(1/val_a * val_phys)

        # Calculating the physical distance for LCDM
        LCDM_d_z_i = self.LCDM_integrand[-1]
        self.LCDM_integrand.pop()
        for val in self.LCDM_integrand:
            self.LCDM_physical_distance.append(LCDM_d_z_i - val)

        # Calculating the luminosity distance for LCDM
        for val_phys, val_a in zip(self.LCDM_physical_distance,self.a):
            self.LCDM_luminosity_distance.append(1/val_a * val_phys)

        # Calculating the fractional luminosity distance between STFQ and LCDM
        for val_STFQ, val_LCDM in zip(self.luminosity_distance,self.LCDM_luminosity_distance):
            self.fractional_luminosity_distance.append((val_STFQ-val_LCDM)/(val_LCDM))
    
    def calculate_fractional_growth(self):
        """Calculate the fractional growth quantity f(z)G(z)"""
        
        for val_STFQ, val_LCDM in zip(self.fG, self.fG_LCDM):
            self.fractional_growth.append((val_STFQ-val_LCDM)/(val_LCDM))

if __name__ == "__main__":
    main()