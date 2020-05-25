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
import seaborn as sn
sn.set_style("whitegrid")


""""""""""""""""""""" PARAMETERS DEFINITION """""""""""""""""""""
T = 24*5
phi = np.arange(0.0, 2*np.pi, 0.001)

theta_1 = 0
theta_2 = 0
l_theta_1=[theta_1]
l_theta_2=[theta_2]
l_val = [0]

T_theta_1 = 24
T_theta_2 = 10


w_1 = 2*np.pi/T_theta_1
w_2 = 2*np.pi/T_theta_2

K = 0.

resolution = 20
domain_theta = np.linspace(0,2*np.pi,resolution)
F = np.zeros((resolution, resolution))
for idx1, th1 in enumerate(domain_theta):
    for idx2, th2 in enumerate(domain_theta):
        F[idx1, idx2] = np.sin(th1-th2)


""""""""""""""""""""" PLOT DEFINITION """""""""""""""""""""

fig, ax_arr = plt.subplots(2, 2, sharex=False, figsize = (10,20))
plt.subplots_adjust(left=0.1, bottom=0.25)

l_1, = ax_arr[0,0].plot(phi, np.sin(phi), lw=2)
ax_arr[0,0].set_title("Oscillator 1")
l_2, = ax_arr[1,0].plot(phi, np.sin(phi), lw=2)
ax_arr[1,0].set_title("Oscillator 2")
l_3 = ax_arr[1,1].contourf(domain_theta, domain_theta, K*F,  vmin = -1, vmax = 1, cmap='bwr')

ax_arr[0,1].axis([0,T_theta_1,-np.pi,np.pi])
ax_arr[0,1].set_title("Dephasing")
#set lims
ax_arr[0,0].axis([0,2*np.pi,-1.5,1.5])
ax_arr[1,0].axis([0,2*np.pi,-1.5,1.5])
ax_arr[1,1].set_xlim((0, 2*np.pi))
ax_arr[1,1].set_ylim((0, 2*np.pi))
ax_arr[1,1].set_title("Phase-space (F is background)")
# set up lines and points
lines, =  ax_arr[1,1].plot([], [],  '-', c='brown', lw = 2)
pts, = ax_arr[1,1].plot([], [],  'o', c='brown')
point_1, = ax_arr[0,0].plot(0,np.sin(theta_1), marker="o", color="steelblue", ms=15)
point_2, = ax_arr[1,0].plot(0,np.sin(theta_2), marker="o", color="steelblue", ms=15)
lines_deph, = ax_arr[0,1].plot([], [], lw=2)

""""""""""""""""""""" SLIDER DEFINITION """""""""""""""""""""
ax_t = plt.axes([0.25, .03, 0.50, 0.02])
ax_K = plt.axes([0.25, 0.1, 0.50, 0.02])
s_t = Slider(ax_t, 't', 0, T, valinit=0)
s_K = Slider(ax_K, 'K', 0, 1, valinit=K)

""""""""""""""""""""" ANIMATION PARAMETERS """""""""""""""""""""
is_manual = False # True if user has taken control of the animation
interval = 100 # ms, time between animation frames
scale = interval / 1000

""""""""""""""""""""" ANIMATION FUNCTIONS """""""""""""""""""""

def update_slider(val):
    global is_manual
    global theta_1
    global theta_2
    global l_theta_1
    global l_theta_2
    global l_val

    is_manual=True
    theta_1+=w_1 * scale
    theta_2+=(w_2 + K*np.sin(theta_1-theta_2))*scale

    if len(l_theta_1)<80:
        l_theta_1.append(theta_1%(2*np.pi))
        l_theta_2.append(theta_2%(2*np.pi))
        l_val.append(val)
    else:
        l_theta_1= l_theta_1[1:]+[theta_1%(2*np.pi)]
        l_theta_2= l_theta_2[1:]+[theta_2%(2*np.pi)]
        l_val = l_val[1:]+[val]

    #mask domain_theta
    abs_d_data_x = np.abs(np.diff(l_theta_1))
    mask_x = np.hstack([ abs_d_data_x > abs_d_data_x.mean()+3*abs_d_data_x.std(), [False]])
    masked_data_x = np.array([x if not m else np.nan for x,m in zip(l_theta_1, mask_x)  ])

    abs_d_data_y = np.abs(np.diff(l_theta_2))
    mask_y = np.hstack([ abs_d_data_y > abs_d_data_y.mean()+3*abs_d_data_y.std(), [False]])
    masked_data_y = np.array([x if not m else np.nan for x,m in zip(l_theta_2, mask_y)  ])

    #mask dephasing
    l_deph = (np.array(l_theta_1)-np.array(l_theta_2))%(2*np.pi)
    l_deph = [x-2*np.pi if x>np.pi else x for x in l_deph]

    abs_d_data_deph = np.abs(np.diff(l_deph))
    mask_deph = np.hstack([ abs_d_data_deph > abs_d_data_deph.mean()+3*abs_d_data_deph.std(), [False]])
    masked_data_deph = np.array([x if not m else np.nan for x,m in zip(l_deph, mask_deph)  ])

    abs_d_data_t = np.abs(np.diff(  np.array(l_val)%T_theta_1     ))
    mask_t = np.hstack([ abs_d_data_t > abs_d_data_t.mean()+3*abs_d_data_t.std(), [False]])
    masked_data_t = np.array([x if not m else np.nan for x,m in zip(np.array(l_val)%T_theta_1, mask_t)  ])

    update(val, theta_1, theta_2, masked_data_x, masked_data_y, masked_data_deph, masked_data_t)

def update_strength(val):
    global K
    K = val
    #l_3.set_array( (K*F[:-1,:-1]).ravel()) #deleting last coordinate is needed to use ravel
    ax_arr[1,1].contourf(domain_theta, domain_theta, K*F,   vmin = -1, vmax = 1, cmap='bwr')

def update(val, theta_1, theta_2, l_theta_1, l_theta_2, l_deph, l_val):
    # update curve
    #l_1.set_ydata(np.sin(t))
    #l_2.set_ydata(np.sin(t))

    #point_1.set_data( (w_1*val)% (2*np.pi),np.sin(w_1*val))
    #point_2.set_data( (w_2*val)% (2*np.pi),np.sin(w_2*val))

    point_1.set_data( (theta_1)% (2*np.pi),np.sin(theta_1))
    point_2.set_data( (theta_2)% (2*np.pi),np.sin(theta_2))
    pts.set_data(theta_1% (2*np.pi),theta_2% (2*np.pi))
    lines.set_data(l_theta_1,l_theta_2)
    lines_deph.set_data(l_val,  l_deph  )

    # redraw canvas while idle
    fig.canvas.draw_idle()

def update_plot(num):
    global is_manual
    global K


    if is_manual:
        return l_1, l_2# don't change
    else:
        val = (s_t.val + scale) % s_t.valmax
        s_t.set_val(val)


        if val>35 and val<=60:
            K_prev = s_K.val
            K = min(K_prev + 0.01, s_K.valmax)
            if K_prev!=K:
                s_K.set_val(K)

        #print(val, K==s_K.valmax, flag, flag2)
        if val>60 and val<=85:
            K_prev = s_K.val
            K = max(K_prev - 0.1, s_K.valmax/2)
            if K_prev!=K:
                s_K.set_val(K)

        if val>85 and val<=110:
            K_prev = s_K.val
            K =  max(K_prev - 0.1, s_K.valmax/4)
            if K_prev!=K:
                s_K.set_val(K)

        if val>110:
            K_prev = s_K.val
            K =  max(K_prev - 0.1, 0)
            if K_prev!=K:
                s_K.set_val(K)

        is_manual = False # the above line called update_slider, so we need to reset this
        return l_1, l_2


def on_click(event):
    # Check where the click happened
    (xm,ym),(xM,yM) = s_t.label.clipbox.get_points()
    if xm < event.x < xM and ym < event.y < yM:
        # Event happened within the slider, ignore since it is handled in update_slider
        return
    else:
        # user clicked somewhere else on canvas = unpause
        global is_manual
        is_manual=False

""""""""""""""""""""" MAIN """""""""""""""""""""

# call update function on slider value change
s_t.on_changed(update_slider)
s_K.on_changed(update_strength)

fig.canvas.mpl_connect('button_press_event', on_click)

#ani = animation.FuncAnimation(fig, update_plot, interval=interval, blit = False)
ani = animation.FuncAnimation(fig, update_plot, frames=14, interval=interval, blit = False)
#ani.save('../Results/Animation/Animation_exhaustive_sync.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi=200)

plt.show()
