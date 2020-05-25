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
""""""""""""""""""""" ANIMATION """""""""""""""""""""
dic_reg_atg = pickle.load( open( "../Data_functions/dic_reg_atg.p", "rb" ) )
dic_atg = pickle.load( open( "../Data_functions/dic_atg.p", "rb" ) )

w = 2*np.pi/24
Xt = np.array(list(range(0,24,2))*4)
Xt_pred = np.array(list(np.linspace(0,24,100))*2)
l_gene = ['arntl', 'clock', 'cry1', 'per1', 'nr1d1']
l_reg =[dic_reg_atg[gene] for gene in l_gene]
l_pred = [[float(B[0])]*len(Xt_pred)+B[1]*np.cos(w*Xt_pred)+B[2]*np.sin(w*Xt_pred) for B in l_reg]

x = Xt_pred
fig, ax = plt.subplots()
for i, gene in enumerate(l_gene):
    plt.plot(Xt, dic_atg[gene].flatten('F'), '.', color = current_palette[i], label = l_gene[i])
data = x
abs_d_data = np.abs(np.diff(data))
mask = np.hstack([ abs_d_data > abs_d_data.mean()+3*abs_d_data.std(), [False]])
x = np.ma.MaskedArray(data, mask)

l_line = [ax.plot(x, y, color = current_palette[i])[0] for i,y in enumerate(l_pred)]
l_pts = [ax.plot(x, y, 'o', color = current_palette[i])[0] for i,y in enumerate(l_pred)]
sn.despine(offset=5, trim=True)
plt.legend(bbox_to_anchor=(0., -0.0, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)

def update(num, x, l_pred, l_line):
    for pts, line, y in zip(l_pts, l_line, l_pred):
        line.set_data(x[max(0,num-50):num], y[max(0,num-50):num])
        pts.set_data(x[num], y[num])
    line.axes.axis([0, 24,12,18])
    return line,

anim = animation.FuncAnimation(fig, update, frames=len(x), fargs=[x, l_pred, l_line], interval=30, blit=True)

#anim.save('../Results/Animation_trajectories_atger.mp4', writer = 'ffmpeg', fps=30, extra_args=['-vcodec','libx264'], dpi = 250)

plt.show()
#plt.close()
