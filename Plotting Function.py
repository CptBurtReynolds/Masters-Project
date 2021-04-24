import numpy as np
import matplotlib.pyplot as mplt

mode = 4

# barotropic
if mode == 0:


    zc5_eta01 = np.load("Barotropic zc5, eta 0.1.npy")
    zc5_eta05 = np.load("Barotropic zc5, eta 0.5.npy")
    zc5_eta1 = np.load("Barotropic zc5, eta 1.0.npy")

    zc10_eta01 = np.load("Barotropic zc10, eta 0.1.npy")
    zc10_eta05 = np.load("Barotropic zc10, eta 0.5.npy")
    zc10_eta1 = np.load("Barotropic zc10, eta 1.0.npy")


    mplt.plot(zc5_eta01[0], zc5_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc5_eta05[0], zc5_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc5_eta1[0], zc5_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    mplt.xlim(-0.01, 10)
    mplt.ylim(-1.01, 0.01)
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Barotrpoic Parameter, w')
    mplt.legend()
    mplt.show()

    mplt.plot(zc10_eta01[0], zc10_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc10_eta05[0], zc10_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc10_eta1[0], zc10_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    mplt.xlim(-0.01, 10)
    mplt.ylim(-1.01, 0.01)
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Barotrpoic Parameter, w')
    mplt.legend()
    mplt.show()

# potential
elif mode == 1:
    zc10_eta01 = np.load("potentialfrac_zc10,eta0.1.npy")
    zc10_eta05 = np.load("potentialfrac_zc10,eta0.5.npy")
    zc10_eta1 = np.load("potentialfrac_zc10,eta1.0.npy")

    mplt.plot(zc10_eta01[0], zc10_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc10_eta05[0], zc10_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc10_eta1[0], zc10_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    mplt.xlim(0.99, 1.01)
    mplt.ylim(0.5, 7)
    mplt.xlabel('STSQ Field, phi/phi_c')
    mplt.ylabel('STSQ Potential, V/V_c')
    mplt.legend()
    mplt.show()

# frac lum dis
elif mode == 2:
    zc10_eta01 = np.load("lumdisfrac_zc10,eta0.1.npy")
    zc10_eta025 = np.load("lumdisfrac_zc10,eta0.25.npy")
    zc10_eta05 = np.load("lumdisfrac_zc10,eta0.5.npy")
    zc10_eta075 = np.load("lumdisfrac_zc10,eta0.75.npy")
    zc10_eta1 = np.load("lumdisfrac_zc10,eta1.0.npy")

    mplt.plot(zc10_eta01[0], zc10_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc10_eta025[0], zc10_eta025[1], label = 'eta = 0.25', linestyle = 'solid', color = 'green')
    mplt.plot(zc10_eta05[0], zc10_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc10_eta075[0], zc10_eta075[1], label = 'eta = 0.75', linestyle = '-.', color = 'purple')
    mplt.plot(zc10_eta1[0], zc10_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    #mplt.xlim(0, 3)
    mplt.ylim(-0.0036, 0)
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Fractional Luminosity Distance, delta d_L/d_L')
    mplt.legend()
    mplt.show()

# growth index
elif mode == 3:
    zc10_eta01 = np.load("growthindex_zc10,eta0.1.npy")
    zc10_eta05 = np.load("growthindex_zc10,eta0.5.npy")
    zc10_eta1 = np.load("growthindex_zc10,eta1.0.npy")
    zc10_LCDM = np.load("growthindex_LCDM.npy")
    zc10_DGP = np.load("growthindex_DGP.npy")
    zc10_DGP_nobeta = np.load("growthindex_DGP_nobeta.npy")
    
    mplt.plot(zc10_eta01[0], zc10_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc10_eta05[0], zc10_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc10_eta1[0], zc10_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    mplt.plot(zc10_LCDM[0], zc10_LCDM[1], label = 'LCDM', linestyle = '-.', color = 'purple')
    mplt.plot(zc10_DGP[0], zc10_DGP[1], label = 'DGP', linestyle = 'solid', color = 'green')
    mplt.plot(zc10_DGP_nobeta[0], zc10_DGP_nobeta[1], label = 'DGP, 1/3beta = 0', linestyle = ':', color = 'cyan')

    mplt.xlabel('Redshift, z')
    mplt.ylabel('Growth Index, gamma')
    mplt.legend()
    mplt.show()

# frac growth
elif mode == 4:
    zc10_eta01 = np.load("growthfrac_zc10,eta0.1.npy")
    zc10_eta025 = np.load("growthfrac_zc10,eta0.25.npy")
    zc10_eta05 = np.load("growthfrac_zc10,eta0.5.npy")
    zc10_eta075 = np.load("growthfrac_zc10,eta0.75.npy")
    zc10_eta1 = np.load("growthfrac_zc10,eta1.0.npy")

    mplt.plot(zc10_eta01[0], zc10_eta01[1], label = 'eta = 0.1', linestyle = 'solid', color = 'blue')
    mplt.plot(zc10_eta025[0], zc10_eta025[1], label = 'eta = 0.25', linestyle = 'solid', color = 'green')
    mplt.plot(zc10_eta05[0], zc10_eta05[1], label = 'eta = 0.5', linestyle = 'dashed', color = 'red')
    mplt.plot(zc10_eta075[0], zc10_eta075[1], label = 'eta = 0.75', linestyle = '-.', color = 'purple')
    mplt.plot(zc10_eta1[0], zc10_eta1[1], label = 'eta = 1.0', linestyle = ':', color = 'orange')
    mplt.xlim(0, 3)
    #mplt.ylim(-0.0036, 0)
    mplt.xlabel('Redshift, z')
    mplt.ylabel('Fractional growth, delta fG/fG')
    mplt.legend()
    mplt.show()
