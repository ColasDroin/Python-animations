"""
Created on Fri Oct 14 10:02:47 2016

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

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.fourier import fourier_ellipsoid
from scipy.ndimage.filters import laplace

sys.path.insert(0, os.path.realpath('..'))
from Data_functions.create_coupling import build_coupling_array_from_2D_gaussian


import seaborn as sn
sn.set_style("whitegrid", {'grid.color': 'white',  'xtick.direction': 'out', 'xtick.major.size': 6.0, 'xtick.minor.size': 3.0,
            'ytick.color': '.15', 'ytick.direction': 'out', 'ytick.major.size': 6.0, 'ytick.minor.size': 3.0, 'axes.grid' : False})

""""""""""""""""""""" RESIZE AND PLOT COUPLING FUNCTION """""""""""""""""""""
n_torus = 50
domain_theta = np.linspace(0,2*np.pi,n_torus)
F = build_coupling_array_from_2D_gaussian( [(np.pi/2, np.pi/2), (3*np.pi/2, 3*np.pi/2)], [[[0.5,0],[0,0.5]], [[0.5,0],[0,0.5]]], domain_theta, domain_theta, [3,-1])

Fp = interpolate.interp2d(np.linspace(0,2*np.pi, F.shape[0], endpoint = False), np.linspace(0,2*np.pi, F.shape[1], endpoint = False), F)
Fp = Fp(np.linspace(0,2*np.pi, n_torus, endpoint = False), np.linspace(0,2*np.pi, n_torus, endpoint = False))


""""""""""""""""""""" CREATE TORUS """""""""""""""""
theta = np.linspace(0, 2.*np.pi, n_torus)
phi = np.linspace(0, 2.*np.pi, n_torus)
theta, phi = np.meshgrid(theta, phi)
c, a = 20, 10
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)+30

norm = plt.Normalize(vmin=-0.6,vmax=0.6)
colors_tore = plt.cm.bwr(norm(Fp))


""""""""""""""""""""" SET FIG AND 3D AXIS """""""""""""""""
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')


# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)


""""""""""""""""""""" ANIMATION FUNCTIONS """""""""""""""""

# initialization function: plot the background of each frame
def init():
    return None

# animation function.  This will be called sequentially with the frame number
def animate(i):
    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return None

""""""""""""""""""""" LAUNCH ANIMATION """""""""""""""""
# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=30)

ax.plot_surface(x, y, z, rstride=1, cstride=1, edgecolors='w', facecolors=colors_tore, shade = False, alpha=None, antialiased=False)
#plt.tight_layout()

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('../Results/torus.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
