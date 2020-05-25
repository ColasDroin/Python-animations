"""
Created on Fri Oct 14 10:02:47 2019

@author: cdroin
"""

""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import interpolate
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation



import seaborn as sn
sn.set_style("whitegrid", {'grid.color': 'white',  'xtick.direction': 'out', 'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
            'ytick.color': '.15', 'ytick.direction': 'out', 'ytick.major.size': 6.0, 'ytick.minor.size': 3.0, 'axes.grid' : False})

current_palette = sn.color_palette()

def waveform(theta, a, b, sigma = 0.1):
    return ((np.cos(theta)+1)/2)**1.6*a+b+np.random.normal(0,sigma)

TF = 48
N = TF*2
tspan = np.concatenate(( np.linspace(0,TF,N), np.linspace(0,TF,N), np.linspace(0,TF,N), np.linspace(0,TF,N) ))
dt = TF/N
theta = 0
l_theta=[theta]
l_s = [waveform(theta, 1, 0)]
T_theta = 24.
w = 2*np.pi/T_theta


sigma = 0.05
a= 1.5
b = +0.5
""" SIMULATE SYSTEM """
for t in tspan[1:]:
    theta+=w * dt+ np.random.normal(0,dt)*sigma
    a += -1/35*(a-1)*dt+ np.random.normal(0,dt)*sigma
    b += -1/35*(b-0.1)*dt+ np.random.normal(0,dt)*sigma
    sig = 0.1
    l_theta.append(theta%(2*np.pi))
    l_s.append(waveform(theta, a, b, sig))

#delete first half of the simulation (to start on steady state)
#tspan = tspan[int(len(tspan)/2):]
#l_theta = l_theta[int(len(tspan)/2):]

#mask domain_theta
abs_d_data_x = np.abs(np.diff(tspan))
mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
masked_data_x = np.array([x if not m else np.nan for x,m in zip(tspan, mask_x)  ])
tspan = masked_data_x

fig, ax = plt.subplots(figsize = (15,5))
line, = ax.plot(tspan, l_s, color = current_palette[1])
pts, = ax.plot(tspan, l_s, 'o', color = current_palette[1])
sn.despine(offset=5, trim=True)


def update(num, tspan, l_s, line, pts):
    line.set_data(tspan[max(0,num-50):num], l_s[max(0,num-50):num])
    pts.set_data(tspan[num-1], l_s[num-1])
    line.axes.axis([0, 48,-0.5,2.5])
    return line, pts

anim = animation.FuncAnimation(fig, update, frames=len(tspan), fargs=[tspan, l_s, line, pts], interval=30, blit=True)
#HTML(anim.to_html5_video())


anim.save('../Results/Animation_stochastic_trajectories.mp4', writer = 'ffmpeg', fps=60, extra_args=['-vcodec','libx264'], dpi = 150)


plt.show()
