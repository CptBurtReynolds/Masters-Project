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

# Input Values
#lambdo = 5 # Model parameter
Z_I = 20 # Initial redshift
Z_C = 10 # Redshift at crossover
ETA = 0.5 # Model parameter after crossover


def main():
    """Program to model the evolution of the STFQ Dark Energy model across a redshift range."""

    a_0 = 1 # Present scale factor
    lambdo, phi_c, a, phi, x = derive_parameters(a_0) # Derived parameters
    t = derive_initial_time(a, lambdo, phi, a_0) # Deriving initial time

    # Initial values and timestep
    da = 0.0001 * a
    dt = (CONSTANT_n*t/(2*a)) * da

    # Growth factor analysis initial values
    G, y, G_LCDM, y_LCDM = derive_inital_growth_parameters(a, da, dt)

    # Structure to store values for plottting later
    values = PlottingArrays()
    
    # Loop for solving the scalar field equation and the density perturbation equation
    while a < a_0:
        #print(y, y_LCDM)
        if phi < phi_c:
            values, dt, x, phi, G, y, G_LCDM, y_LCDM, a, t = pre_transition(values, lambdo, phi, x, G, y, G_LCDM, y_LCDM, a_0, a, dt, t, da)
        else:
            values, dt, x, phi, G, y, G_LCDM, y_LCDM, a, t = post_transition(values, lambdo, phi_c, phi, x, G, y, G_LCDM, y_LCDM, a_0, a, dt, da, t)

    values.calculate_physical_Luminosity_distance() # Calculate the physical and luminosity distances. Done post-loop due to form of intergral
    values.calculate_fractional_growth() # Calculates the fractional growth from the STFQ and LCDM f(z)G(z) values

    print(len(values.a))
    #for i in range(5000,10000):
    #    print(values.a[i], values.G_LCDM[i], values.y_LCDM[i])
    plotGraphs(values) # Plot graphs from calculated values

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

def derive_inital_growth_parameters(a, da, dt):
    G_i = 1*a
    y_i = da/dt

    return G_i, y_i, G_i, y_i

def derive_initial_time(a_i, lambdo, phi_i, a_0):
    """Derives initial time parameters from global variables and input parameters"""

    rho_m_i = (a_0/a_i)**3 * CONSTANT_rho_m0 # Initial matter energy density
    rho_phi_i = 2 * CONSTANT_M_pl_4 * np.exp(-lambdo*phi_i/CONSTANT_M_pl) # Initial scalar energy density

    # Initial time derived with Hubble paramter
    H_i = ((rho_m_i+rho_phi_i)/(3*CONSTANT_M_pl_2))**(1/2)
    t_i = 2/(CONSTANT_n*H_i)
    
    return t_i

def pre_transition(values, lambdo, phi, x, G, y, G_LCDM, y_LCDM, a_0, a, dt, t, da):
    """Solve the scalar field and density purterbation equations pre-transition"""

    # Calculating values from inputs, these will be used to update the infintesimals
    scalar_potential = CONSTANT_M_pl_4 * np.exp(-lambdo*phi/CONSTANT_M_pl)
    # Calculating Energy densities, pressure and the barotropic parameter
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**(3.0) * CONSTANT_rho_m0
    barotropic_parameter = pressure_phi/energy_density_phi
    # Calculating the Hubble parameter for STFQ and LCDM
    H = ((energy_density_m+energy_density_phi)/(3*CONSTANT_M_pl_2))**(1/2)
    H_LCDM = ((CONSTANT_rho_m0 * a**(-3.0) + CONSTANT_rho_de0)/(3.0*CONSTANT_M_pl_2))**(1.0/2.0)
    
    # Solving scalar field equation pre-transition
    dphi = x*dt
    dx = (-6*x/(CONSTANT_n*t) + lambdo*CONSTANT_M_pl_3*np.exp(-lambdo*phi/CONSTANT_M_pl)) * dt
    
    # Solving density perturbation equation pre-transition for both STFQ and LCDM
    dG = y*dt
    dy = ((3/2)*CONSTANT_Omega_m0*CONSTANT_H_0_2*(1/a**3)*G - 2*H*y)*dt
    dG_LCDM = y_LCDM*dt
    dy_LCDM = ((3.0/2.0)*CONSTANT_Omega_m0*CONSTANT_H_0_2*(1.0/a**(3.0))*G_LCDM - 2.0*H_LCDM*y_LCDM)*dt
    
    # Placing values into arrays so they can be plotted later
    values.a.append(a)
    values.z.append((1/a) - 1)
    values.phi.append(phi)
    values.G.append(G)
    values.y.append(y)
    values.t.append(t)
    values.G_LCDM.append(G_LCDM)
    values.y_LCDM.append(y_LCDM)
    values.f.append((a/G)*(dG/da))
    values.f_LCDM.append((a/G_LCDM)*(dG_LCDM/da))
    values.fG.append(a*(dG/da))
    values.fG_LCDM.append(a*(dG_LCDM/da))
    values.H_LCDM.append(H_LCDM)
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.LCDM_integrand.append(values.LCDM_integrand[-1] + (da/(H_LCDM * a**2)))
    values.Omega.append(energy_density_phi/(energy_density_phi+energy_density_m))

    # Updating the timestamp
    dt = (CONSTANT_n*t/(2*a)) * da

    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    y = y + dy
    G = G + dG
    y_LCDM = y_LCDM + dy_LCDM
    G_LCDM = G_LCDM + dG_LCDM
    a = a + da
    t = t + dt
    
    return values, dt, x, phi, G, y, G_LCDM, y_LCDM, a, t


def post_transition(values, lambdo, phi_c, phi, x, G, y, G_LCDM, y_LCDM, a_0, a, dt, da, t):
    """Solve the scalar field and density purterbation equations post-transition"""

    # Calculating values from inputs, these will be used to update the infintesimals
    scalar_potential = CONSTANT_M_pl_4 * np.exp(-lambdo*phi_c/CONSTANT_M_pl) * np.exp(lambdo*ETA*(phi - phi_c)/CONSTANT_M_pl)
    # Calculating Energy densities, pressures and the barotropic parameter
    energy_density_phi = x**2 /2 + scalar_potential
    pressure_phi = x**2 /2 - scalar_potential
    energy_density_m = (a_0/a)**(3.0) * CONSTANT_rho_m0
    barotropic_parameter = pressure_phi/energy_density_phi
    # Calculating the Hubble parameter for STFQ and LCDM
    H = ((energy_density_m+energy_density_phi)/(3*CONSTANT_M_pl_2))**(1/2)
    H_LCDM = ((CONSTANT_rho_m0 * a**(-3.0) + CONSTANT_rho_de0)/(3.0*CONSTANT_M_pl_2))**(1.0/2.0)
    
    # Solving scalar field equation post-transition
    dphi = x*dt
    dx = (-3*H*x - (ETA*lambdo/CONSTANT_M_pl)*scalar_potential) * dt
    
    # Solving density perturbation equation pre-transition
    dG = y*dt
    dy = ((3/2)*CONSTANT_Omega_m0*CONSTANT_H_0_2*(1/a**3)*G - 2*H*y)*dt

    dG_LCDM = y_LCDM*dt
    dy_LCDM = ((3.0/2.0)*CONSTANT_Omega_m0*CONSTANT_H_0_2*(1.0/a**(3.0))*G_LCDM - 2.0*H_LCDM*y_LCDM)*dt

    # Placing values into arrays so they can be plotted later
    values.a.append(a)
    values.z.append((1/a) - 1)
    values.phi.append(phi)
    values.G.append(G)
    values.y.append(y)
    values.t.append(t)
    values.G_LCDM.append(G_LCDM)
    values.y_LCDM.append(y_LCDM)
    values.f.append((a/G)*(dG/da))
    values.f_LCDM.append((a/G_LCDM)*(dG_LCDM/da))
    values.fG.append(a*(dG/da))
    values.fG_LCDM.append(a*(dG_LCDM/da))
    values.H_LCDM.append(H_LCDM)
    values.barotropic_parameter.append(barotropic_parameter)
    values.physical_distance_integrand.append(values.physical_distance_integrand[-1] + (da/(H * a**2)))
    values.LCDM_integrand.append(values.LCDM_integrand[-1] + (da/(H_LCDM * a**2)))
    values.Omega.append(energy_density_phi/(energy_density_phi+energy_density_m))
    
    # Updating the timestamp
    dt = (CONSTANT_n*t/(2*a)) * da

    # Progressing timestep and updating values
    x = x + dx
    phi = phi + dphi
    y = y + dy
    G = G + dG
    y_LCDM = y_LCDM + dy_LCDM
    G_LCDM = G_LCDM + dG_LCDM
    a = a + da
    t = t + dt
    
    return values, dt, x, phi, G, y, G_LCDM, y_LCDM, a, t


def plotGraphs(values):
    """Plots Graph from given values"""

    #mplt.plot(values.a, values.y, label = 'STFQ', color = 'r')
    mplt.plot(values.a, values.G_LCDM, label = 'G', color = 'c')
    mplt.plot(values.a, values.y_LCDM, label = 'dG/dt, y', color = 'r')
    mplt.xlabel('Scale Factor, a')
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
        self.phi = []
        self.G = []
        self.G_LCDM = []
        self.y = []
        self.y_LCDM = []
        self.t = []
        self.f = []
        self.f_LCDM = []
        self.fG = []
        self.fG_LCDM = []
        self.fractional_growth = []
        self.growth_index = []
        self.H_LCDM = []
        self.barotropic_parameter = []
        self.physical_distance_integrand = [0]
        self.LCDM_integrand = [0]
        self.physical_distance = []
        self.LCDM_physical_distance = []
        self.luminosity_distance = []
        self.LCDM_luminosity_distance = []
        self.fractional_luminosity_distance = []
        self.Omega = []

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