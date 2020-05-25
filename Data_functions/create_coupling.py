# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:02:47 2016

@author: cdroin
"""

""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""

### Import external modules
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.style.use('ggplot')

""""""""""""""""""""" FUNCTION """""""""""""""""""""

def build_coupling_array_from_2D_gaussian(l_t_coor, l_mat_var, domainx, domainy, l_amp):
    l_res = np.zeros( (len(domainx),len(domainy)))
    for t_coor, mat_var, amp in zip(l_t_coor, l_mat_var, l_amp):
        for i, phase1 in enumerate(domainx):
            for j, phase2 in enumerate(domainy):
                l_res[i,j] += amp * st.multivariate_normal.pdf( [phase1, phase2], mean=t_coor, cov=mat_var)
                l_res[i,j] += amp * st.multivariate_normal.pdf( [phase1, phase2], mean=(t_coor[0]+2*np.pi,  t_coor[1] ), cov=mat_var)
                l_res[i,j] += amp * st.multivariate_normal.pdf( [phase1, phase2], mean=(t_coor[0],  t_coor[1]+2*np.pi ), cov=mat_var)
                l_res[i,j] += amp * st.multivariate_normal.pdf( [phase1, phase2], mean=(t_coor[0]+2*np.pi,  t_coor[1]+2*np.pi ), cov=mat_var)
    return l_res

""""""""""""""""""""" TEST """""""""""""""""""""

if __name__ == '__main__':
    domain_theta = np.linspace(0,2*np.pi,50, endpoint = False)
    domain_phi = np.linspace(0,2*np.pi,50, endpoint = False)

    t_coor_1 = (1.5,4.5)
    mat_var_1 = np.array( [[0.5, 0 ], [0, 0.5]]   )
    t_coor_2 = (5,2)
    mat_var_2 = np.array( [[0.5, 0 ], [0, 0.5]]   )
    #t_coor_3 = (5,3)
    #mat_var_3 = np.array( [[0.5, 0 ], [0., 0.5]]        )
    l_amp_1 = [-1,1]#, -2]

    F = build_coupling_array_from_2D_gaussian(l_t_coor = [t_coor_1, t_coor_2], l_mat_var = [mat_var_1, mat_var_2], domainx = domain_theta, domainy = domain_phi, l_amp = l_amp_1)


    plt.pcolor(F.T,cmap='bwr', vmin=-0.3, vmax=0.3)
    plt.colorbar()
    plt.xlabel("theta")
    plt.ylabel("phi")
    plt.show()
    plt.close()

    plt.imshow(F.T, origin='lower', interpolation = "bessel" )
    plt.colorbar()
    plt.xlabel("theta")
    plt.ylabel("phi")
    plt.show()
    plt.close()
