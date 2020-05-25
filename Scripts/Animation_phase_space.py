# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:02:47 2016

@author: cdroin
"""

""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import scipy.stats as st
import sys
import os
from Data_functions.create_coupling import build_coupling_array_from_2D_gaussian
import seaborn as sn
sn.set_style("whitegrid", {'grid.color': 'white',  'xtick.direction': 'out', 'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
            'ytick.color': '.15', 'ytick.direction': 'out', 'ytick.major.size': 6.0, 'ytick.minor.size': 3.0, 'axes.grid' : False})
""""""""""""""""""""" PARAMETERS DEFINITION """""""""""""""""""""
current_palette = sn.color_palette()

N = 1000*4
TF = 100*2
tspan = np.linspace(0,TF,N)
dt = TF/N
theta_1 = 0
theta_2 = 0
l_theta_1=[theta_1]
l_theta_2=[theta_2]

T_theta_1 = 24.1/10
T_theta_2 = 22./2/10


w_1 = 2*np.pi/T_theta_1
w_2 = 2*np.pi/T_theta_2

K1 = 3*12#10.
K2 = -1*12#-5
resolution = 50
domain_theta = np.linspace(0,2*np.pi,resolution)
F = build_coupling_array_from_2D_gaussian( [(np.pi/2, np.pi/2), (3*np.pi/2, 3*np.pi/2)], [[[0.5,0],[0,0.5]], [[0.5,0],[0,0.5]]], domain_theta, domain_theta, [K1,K2])


sigma = 1
""" SIMULATE SYSTEM """
for t in tspan:
    theta_1+=w_1 * dt\
            + K1*st.multivariate_normal.pdf( [theta_1%(2*np.pi), theta_2%(2*np.pi)], mean=(np.pi/2, np.pi/2), cov=[[0.5,0],[0,0.5]])*dt\
            + K2*st.multivariate_normal.pdf( [theta_1%(2*np.pi), theta_2%(2*np.pi)], mean=(3*np.pi/2, 3*np.pi/2), cov=[[0.5,0],[0,0.5]])*dt\
            + np.random.normal(0,dt)*sigma
    #theta_2+=(w_2 + K*np.sin(theta_1-theta_2))*scale
    theta_2+=w_2 *dt+ np.random.normal(0,dt)*sigma
    l_theta_1.append(theta_1%(2*np.pi))
    l_theta_2.append(theta_2%(2*np.pi))

#delete first half of the simulation (to start on steady state)
tspan = tspan[int(len(tspan)/2):]
l_theta_1 = l_theta_1[int(len(tspan)/2):]
l_theta_2 = l_theta_2[int(len(tspan)/2):]

#mask domain_theta
abs_d_data_x = np.abs(np.diff(l_theta_1))
mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
masked_data_x = np.array([x if not m else np.nan for x,m in zip(l_theta_1, mask_x)  ])
x = masked_data_x

abs_d_data_y = np.abs(np.diff(l_theta_2))
mask_y = np.hstack([ abs_d_data_y > abs_d_data_y.mean()+3*abs_d_data_y.std(), [False]])
masked_data_y = np.array([x if not m else np.nan for x,m in zip(l_theta_2, mask_y)  ])
y = masked_data_y
""""""""""""""""""""" PLOT DEFINITION """""""""""""""""""""
fig, ax_arr = plt.subplots(figsize=(10, 10))
#plt.subplots_adjust(bottom=0.25)

#ax_arr.contourf(domain_theta, domain_theta, K*F, vmin = -2, vmax = 2, cmap='bwr')
ax_arr.imshow(F.T, vmin = -8, vmax = 8, cmap='bwr',interpolation='spline16', origin='lower', extent=[0, 2*np.pi,0, 2*np.pi])

# set up lines and points
line, =  ax_arr.plot([], [],  '-', c=current_palette[1], lw = 4)
line2, =  ax_arr.plot([], [],  '--', c=current_palette[1], lw = 1, alpha = 0.4)

pts, = ax_arr.plot([], [],  'o', c=current_palette[1], markersize = 6)

ax_arr.set_xlim((0, 2*np.pi-0.1))
ax_arr.set_ylim((0, 2*np.pi-0.1))
ax_arr.set_xlabel(r'$\phi_1$')
ax_arr.set_ylabel(r'$\phi_2$')
plt.tight_layout()
""""""""""""""""""""" ANIMATION FUNCTIONS """""""""""""""""""""

def update(num):
    line.set_data(x[max(0,num-10):num], y[max(0,num-10):num])
    line2.set_data(x[:num], y[:num])
    pts.set_data(x[num-1], y[num-1])
    #line.axes.axis([0, 24,12,18])
    return line, line2, pts

anim = animation.FuncAnimation(fig, update, frames=len(x), interval=10, blit=True)
#anim.save('../Results/Animation_phase_space.mp4', writer = 'ffmpeg', fps=60, extra_args=['-vcodec','libx264'])#, dpi = 250)

plt.show()
