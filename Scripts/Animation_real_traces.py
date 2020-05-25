"""
Created on Fri Oct 14 10:02:47 2016

@author: cdroin

NB : This animation requires dependencies from the coupling project, which
can be found here: https://github.com/ColasDroin/CouplingHMM
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
sys.path.insert(0, os.path.realpath('../Classes'))

from Functions.create_hidden_variables import create_hidden_variables
from Functions.signal_model import signal_model

from Classes.StateVar import StateVar
from Classes.HMMsim import HMMsim


""""""""""""""""""""" CHOOSE DATA """""""""""""""""""""
temperature = None
cell = 'NIH3T3'


""""""""""""""""""""" LOAD PARAMETERS """""""""""""""""""""
with open('../Parameters/Real/opt_parameters_div_'+str(temperature)+"_"+cell+'.p', 'rb') as f:
    l_parameters = [dt, sigma_em_circadian, W, pi,
    N_theta, std_theta, period_theta, l_boundaries_theta, w_theta,
    N_phi, std_phi, period_phi, l_boundaries_phi, w_phi,
    N_amplitude_theta, mu_amplitude_theta, std_amplitude_theta,  gamma_amplitude_theta, l_boundaries_amplitude_theta,
    N_background_theta, mu_background_theta, std_background_theta, gamma_background_theta, l_boundaries_background_theta,
    F] = pickle.load(f)

""""""""""""""""""""" SET ANIMATION PARAMETERS """""""""""""""""""""
N_trajectories = 20
n_torus = N_theta

""""""""""""""""""""" RESIZE AND PLOT COUPLING FUNCTION """""""""""""""""""""
Fp = interpolate.interp2d(np.linspace(0,2*np.pi, F.shape[0], endpoint = False), np.linspace(0,2*np.pi, F.shape[1], endpoint = False), F)
Fp = Fp(np.linspace(0,2*np.pi, n_torus, endpoint = False), np.linspace(0,2*np.pi, n_torus, endpoint = False))


""""""""""""""""""""" SMOOTH AND PLOT COUPLING FUNCTION """""""""""""""""""""
F_t = myfft = np.fft.rfft2(Fp)
F_mag = np.abs(np.fft.fftshift(F_t))
F_phase = np.angle(np.fft.fftshift(F_t))
F_t = np.fft.fftshift(F_t)

tenth = np.percentile( F_mag.flatten(), 99.2)
ll_idx_coef = [(i,j) for i in range(F_mag.shape[0]) for j in range(F_mag.shape[1]) if F_mag[i,j]<tenth]
print("remaining coef:" , F_mag.shape[0]*F_mag.shape[1]-len(ll_idx_coef))

for (i,j) in ll_idx_coef:
    F_t[i,j] = 0
    F_mag[i,j] = 0
    F_phase[i,j] = 0

Fp = np.fft.irfft2(np.fft.ifftshift(F_t) )


""""""""""""""""""""" CREATE HIDDEN VARIABLES """""""""""""""""""""
theta_var_coupled, amplitude_var, background_var = create_hidden_variables(l_parameters = l_parameters )
l_var = [theta_var_coupled, amplitude_var, background_var]
domain_theta = theta_var_coupled.domain
domain_phi = theta_var_coupled.codomain
""""""""""""""""""""" SIMULATE TRACES """""""""""""""""""""
sim = HMMsim(  l_var, signal_model , sigma_em_circadian,  waveform = W , dt=0.5*dt, uniform = True )
ll_t_l_xi, ll_t_obs  =  sim.simulate_n_traces(nb_traces=N_trajectories, tf=100)

""""""""""""""""""""" REORDER VARIABLES """""""""""""""""
ll_obs_circadian = []
ll_obs_nucleus = []
lll_xi_circadian = []
lll_xi_nucleus = []
for idx, (l_t_l_xi, l_t_obs) in enumerate(zip(ll_t_l_xi, ll_t_obs)):
    ll_xi_circadian = [ t_l_xi[0] for t_l_xi in l_t_l_xi   ]
    ll_xi_nucleus = [ t_l_xi[1] for t_l_xi in l_t_l_xi   ]
    l_obs_circadian = np.array(l_t_obs)[:,0]
    l_obs_nucleus = np.array(l_t_obs)[:,1]
    ll_obs_circadian.append(l_obs_circadian)
    ll_obs_nucleus.append(l_obs_nucleus)
    lll_xi_circadian.append(ll_xi_circadian)
    lll_xi_nucleus.append(ll_xi_nucleus)

""""""""""""""""""""" CREATE PHASE COORDINATES """""""""""""""""
x_t = []
for ll_x_theta, ll_x_phi in zip(lll_xi_circadian, lll_xi_nucleus):
    l_x_theta =  np.array(ll_x_theta)[:,0]
    l_y_phi = np.array(ll_x_phi)[:,0]


    abs_d_data_x = np.abs(np.diff(l_x_theta))
    mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
    masked_data_x = np.array([x if not m else np.nan for x,m in zip(l_x_theta, mask_x)  ])

    abs_d_data_y = np.abs(np.diff(l_y_phi))
    mask_y = np.hstack([ abs_d_data_y > abs_d_data_y.mean()+3*abs_d_data_y.std(), [False]])
    masked_data_y = np.array([x if not m else np.nan for x,m in zip(l_y_phi, mask_y)  ])

    x_t.append( (masked_data_x, masked_data_y )  )

x_t = np.array(x_t)



""""""""""""""""""""" CREATE FIG FOR ANIMATION """""""""""""""""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('on')
plt.xlabel("Cell-cycle phase")
plt.ylabel("Circadian phase")

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = [l for c in colors for l in ax.plot([], [],  '-', c=c)]
pts = [pt for c in colors for pt in ax.plot([], [],  'o', c=c)]

# prepare the axes limits
ax.set_xlim((0, 2*np.pi))
ax.set_ylim((0, 2*np.pi))


""""""""""""""""""""" ANIMATION FUNCTIONS """""""""""""""""
# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        pt.set_data([], [])

    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = i % x_t.shape[2]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y = xi[0,:i], xi[1,:i]
        if x.shape[0]<10:
            line.set_data(x, y)
        else:
            line.set_data(x[x.shape[0]-10:], y[y.shape[0]-10:])

        pt.set_data(x[-1:], y[-1:])

    fig.canvas.draw()
    return lines + pts


plt.pcolormesh(domain_theta, domain_phi, Fp.T, cmap='bwr', vmin = -0.3, vmax = 0.3)

""""""""""""""""""""" LAUNCH ANIMATION """""""""""""""""

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=30)

# Save as mp4. This requires mplayer or ffmpeg to be installed
anim.save('../Results/Animation_real_traces.mp4', fps=24, extra_args=['-vcodec', 'libx264'])

#plt.show()
