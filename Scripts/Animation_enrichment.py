"""
Created on Fri Oct 14 10:02:47 2019

@author: cdroin
"""

""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns
import scipy
from scipy.interpolate import interp1d
from scipy import signal
from itertools import groupby
from scipy import interpolate
sns.set()
sns.set_style("white")
current_palette = sns.color_palette()

""""""""""""""""""""" LOAD DATA """""""""""""""""""""
df = pd.read_csv('../Data_functions/statistics_enrichment_df.txt', sep="\t", error_bad_lines=False)

""""""""""""""""""""" FUNCTIONS FOR PLOTTING """""""""""""""""""""
def smooth(l_r ):
    l_r = signal.savgol_filter(l_r, 11, 3)
    return l_r

#functions to get pv and draw them
def get_value_from_terms(term):
    sodium_transport_KO = df.loc[(df.Var1 == term)&(df.model == 'CBKO')]
    sodium_transport_WT = df.loc[(df.Var1 == term)&(df.model == 'System_driven')]

    l_pv_KO = -np.log10(sodium_transport_KO['value'])
    l_angle_KO = sodium_transport_KO['Var2']/24*2*np.pi

    l_pv_WT = -np.log10(sodium_transport_WT['value'])
    l_angle_WT = sodium_transport_WT['Var2']/24*2*np.pi

    #smooth
    l_pv_KO  = smooth(l_pv_KO)
    l_pv_WT = smooth(l_pv_WT)
    return l_pv_KO, l_angle_KO, l_pv_WT, l_angle_WT

def draw_radar_plot(l_t_l_angle, l_t_l_pv, l_labels, l_color = current_palette, title = '', folder='', pathway_type = 'clock'):

    # Initialise the spider plot
    fig = plt.figure(figsize = (15,15))
    if pathway_type is not 'both':
        ax1 = fig.add_subplot(111,projection='polar')
    else:
        ax1 = fig.add_subplot(121,projection='polar')
        ax2 = fig.add_subplot(122,projection='polar')

    # Draw one axe per variable + add labels labels yet
    ticks = ['ZT' + str(x) for x in range(0,24,3)]
    ax1.set_xticklabels(ticks, size=22)
    ax1.tick_params(pad = 30)
    #plt.xticks(np.linspace(0,2*np.pi,8, endpoint = False), )#, color='black', size=8)

    ax1.set_ylim([0,6])
    ax1.set_yticks([2,4,6])
    ax1.set_yticklabels([r"$10^{-2}$",r"$10^{-4}$",r"$10^{-6}$"], color="grey", size=18)


    # Draw ylabels
    ax1.set_rlabel_position(50)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")
    ax1.grid(True)

    if pathway_type is 'both':
        ax2.set_xticklabels(ticks, size=22)
        ax2.tick_params(pad = 30)
        ax2.set_ylim([0,6])
        ax2.set_yticks([2,4,6])
        ax2.set_yticklabels([r"$10^{-2}$",r"$10^{-4}$",r"$10^{-6}$"], color="grey", size=18)
        ax2.set_rlabel_position(50)
        ax2.set_theta_direction(-1)
        ax2.set_theta_zero_location("N")
        ax2.grid(True)

    # Plot data
    for t_l_angle, t_l_pv, color, label in zip(l_t_l_angle, l_t_l_pv, l_color, l_labels):

        if pathway_type is 'system':
            ax1.plot(t_l_angle[0], t_l_pv[0], '.-', linewidth=1.5, markersize=3, label = label, color = color)
            ax1.fill(t_l_angle[0], t_l_pv[0], alpha=0.15, color= color, label='_nolegend_')
        elif pathway_type is 'clock':
            ax1.plot(t_l_angle[1], t_l_pv[1], '.-', linewidth=1.5, markersize=3, label = label, color = color)
            ax1.fill(t_l_angle[1], t_l_pv[1], alpha=0.15, color= color, label='_nolegend_')
        else:
            ax1.plot(t_l_angle[0], t_l_pv[0], '.-', linewidth=1.5, markersize=3, label = label, color = color)
            ax1.fill(t_l_angle[0], t_l_pv[0], alpha=0.15, color= color, label='_nolegend_')

            ax2.plot(t_l_angle[1], t_l_pv[1], '.-', linewidth=1.5, markersize=3, label = label, color = color)
            ax2.fill(t_l_angle[1], t_l_pv[1], alpha=0.15, color= color, label='_nolegend_')

            ax1.set_title('CBKO', va='bottom', size=15, y=1.1)
            ax2.set_title('System driven', va='bottom', size=15, y=1.1)


    if pathway_type is not 'both':
        ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    else:
        ax2.legend(loc='upper right', bbox_to_anchor=(1.5, 1.2))


    #plt.title(title, size=15)#, color=l_color[0], y=1.1)
    fig.suptitle(title, size=25)
    #plt.tight_layout()

    if title is not '':
        plt.savefig(folder+title+'.pdf')
    plt.show()

def plot_one_ZT(l_terms, title, folder, pathway_type = 'clock'):
    l_t_l_angle = []
    l_t_l_pv = []
    l_labels = []
    for term in l_terms:
        l_pv_KO, l_angle_KO, l_pv_WT, l_angle_WT = get_value_from_terms(term)

        l_pv_KO_crop = [i for i,x in zip(l_pv_KO, l_angle_KO) if x>=0 and x<=2*np.pi]
        l_angle_KO_crop = [x for i,x in zip(l_pv_KO, l_angle_KO) if x>=0 and x<=2*np.pi]

        l_pv_WT_crop = [i for i,x in zip(l_pv_WT, l_angle_WT) if x>=0 and x<=2*np.pi]
        l_angle_WT_crop = [x for i,x in zip(l_pv_WT, l_angle_WT) if x>=0 and x<=2*np.pi]

        #ensure closure
        l_pv_KO_crop[-1] = l_pv_KO_crop[0]
        l_pv_WT_crop[-1] = l_pv_WT_crop[0]

        #ensure no negative term
        l_pv_KO_crop = [x if x>=0 else 0.00001 for x in l_pv_KO_crop ]
        l_pv_WT_crop = [x if x>=0 else 0.00001 for x in l_pv_WT_crop ]

        l_t_l_angle.append((l_angle_WT_crop, l_angle_KO_crop))
        l_t_l_pv.append((l_pv_WT_crop, l_pv_KO_crop))
        l_labels.append(term.split('%')[0].lower().capitalize())
    draw_radar_plot(l_t_l_angle, l_t_l_pv, l_labels = l_labels, l_color = current_palette, title = title, folder = folder, pathway_type = pathway_type)

def interpolate_go(ll_angle, ll_radius, l_label, res):
    pal = sns.color_palette("hls", res*(len(ll_angle)-1)+1)
    ll_angle_new = []
    ll_radius_new = []
    l_label_new = []
    l_color_new = []


    tspan = np.linspace(0,1,len(ll_angle), endpoint = True)
    ll_angle = np.array(ll_angle)
    ll_radius = np.array(ll_radius)
    for j in range(ll_angle.shape[1]):
        f_angle = interpolate.interp1d(tspan, ll_angle[:,j])
        f_radius = interpolate.interp1d(tspan, ll_radius[:,j])
        new_tspan = np.linspace(0,1,(len(ll_angle)-1)*res+1, endpoint = True)

        ll_angle_new.append(f_angle(new_tspan))
        ll_radius_new.append(f_radius(new_tspan))


    ll_angle_new = np.array(ll_angle_new).T
    ll_radius_new = np.array(ll_radius_new).T



    for i in range(len(ll_angle)-1):
        for j in range(res):
            l_label_new.append(l_label[i])
            l_color_new.append(pal[i*res+j])

    l_label_new.append(l_label[-1])
    l_color_new.append(pal[-1])


    #repeat value when right on term to give a nice transition effects
    ll_angle_smooth = np.array([ll_angle_new[0,:] for j in range(2*res)])
    ll_radius_smooth = np.array([ll_radius_new[0,:] for j in range(2*res)])
    l_label_smooth = [l_label_new[0]for j in range(2*res)]
    l_color_smooth = [l_color_new[0]for j in range(2*res)]


    for i in range(0,len(ll_angle)-1):
        ll_angle_smooth = np.vstack((ll_angle_smooth, ll_angle_new[i*res:(i+1)*res]))
        new_block = np.array([ll_angle_new[(i+1)*res,:] for j in range(2*res)])
        ll_angle_smooth = np.vstack((ll_angle_smooth, new_block))

        ll_radius_smooth = np.vstack((ll_radius_smooth, ll_radius_new[i*res:(i+1)*res]))
        new_block = np.array([ll_radius_new[(i+1)*res,:] for j in range(2*res)])
        ll_radius_smooth = np.vstack((ll_radius_smooth, new_block))

        l_label_smooth = l_label_smooth + ['' for k in l_label_new[i*res:(i+1)*res]]
        l_label_smooth = l_label_smooth + [l_label_new[(i+1)*res] for j in range(2*res)]

        l_color_smooth = l_color_smooth + l_color_new[i*res:(i+1)*res]
        l_color_smooth = l_color_smooth + [l_color_new[(i+1)*res] for j in range(2*res)]


    return ll_angle_smooth, ll_radius_smooth, l_label_smooth, l_color_smooth
""""""""""""""""""""" ANIMATION """""""""""""""""""""
#SYSTEM DRIVEN

l_terms = ['RIBOSOME BIOGENESIS%GOBP%GO:0042254', #ZT15
          'CHAPERONE-MEDIATED PROTEIN FOLDING%GOBP%GO:0061077', #ZT18
          'MRNA SPLICING%REACTOME%R-HSA-72172.3', #ZT19
          'FATTY ACID BIOSYNTHETIC PROCESS%GOBP%GO:0006633', #ZT20
          'REGULATION OF CHOLESTEROL BIOSYNTHESIS BY SREBP (SREBF)%REACTOME DATABASE ID RELEASE 66%1655829',  #ZT21
          'HALLMARK_MTORC1_SIGNALING%MSIGDB_C2%HALLMARK_MTORC1_SIGNALING',  #ZT18
          'INSULIN RECEPTOR SIGNALING PATHWAY%GOBP%GO:0008286', #'RESPONSE TO INSULIN%GOBP%GO:0032868', #ZT12
          'LIPID METABOLIC PROCESS%GOBP%GO:0006629',  #ZT10
          ]

l_titles = [
          'Ribosome biogenesis (ZT17)', #ZT15
          'Protein folding (ZT18)', #ZT18
          'mRNA splicing (ZT19)', #ZT19
          'Fatty acid biosynthetic process (ZT20)', #ZT20
          'Cholesterol biosynthesis & SREBP (ZT21)', #ZT21
          'mTOR signaling (ZT22)', #ZT18
          'Insulin signaling (ZT10)', #ZT09
          'Lipid catabolism (ZT11)',  #ZT10
          ]

system_driven = True

ll_pv = []
ll_angle = []
l_label = []
res = 100
for k, term in enumerate(l_terms):
    l_pv_KO, l_angle_KO, l_pv_WT, l_angle_WT = get_value_from_terms(term)

    #ensure no negative term
    l_pv_KO = [x if x>=0 else 0.00001 for x in l_pv_KO ]
    l_pv_WT = [x if x>=0 else 0.00001 for x in l_pv_WT ]


    if system_driven:
        ll_pv.append([i for i,x in zip(l_pv_WT, l_angle_WT) if x>=0 and x<=2*np.pi])
        ll_angle.append([x for i,x in zip(l_pv_WT, l_angle_WT) if x>=0 and x<=2*np.pi])
    else:
        ll_pv.append([i for i,x in zip(l_pv_KO, l_angle_KO) if x>=0 and x<=2*np.pi])
        ll_angle.append([x for i,x in zip(l_pv_KO, l_angle_KO) if x>=0 and x<=2*np.pi])

    l_label.append(term)



ll_angle, ll_pv, l_label, l_color = interpolate_go(ll_angle, ll_pv, l_titles, res)


# Initialise the spider plot
fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(111,projection='polar')

# Draw one axe per variable + add labels labels yet
ticks = ['ZT' + str(x) for x in range(0,24,3)]
ax1.set_xticklabels(ticks, size=22)
#thetaticks = np.arange(0,360,45)
#ax1.set_thetagrids(thetaticks, frac=1.3)
ax1.tick_params(pad = 30)

ax1.set_ylim([0,6])
ax1.set_yticks([2,4,6])
ax1.set_yticklabels([r"$10^{-2}$",r"$10^{-4}$",r"$10^{-6}$"], color="grey", size=18)

#ax1.set_ylim([0,4])
#ax1.set_yticks([2,4])
#ax1.set_yticklabels([r"$10^{-2}$",r"$10^{-4}$"], color="grey", size=18)


# Draw ylabels
ax1.set_rlabel_position(50)
ax1.set_theta_direction(-1)
ax1.set_theta_zero_location("N")

# Plot data
l, = ax1.plot(ll_angle[0], ll_pv[0], '.-', linewidth=1.5, markersize=3, label = l_label[0], color = 'red')
f, = ax1.fill(ll_angle[0], ll_pv[0], alpha=0.15, color= 'red', label='_nolegend_')
#ax1.set_title('System Driven', va='bottom', size=18, y=1.1)

#ax1.grid(True)


#ax1.legend(loc='upper right', bbox_to_anchor=(1.5, 1.2))

#redefine color according to dominant angle
pal = sns.color_palette("hls", 360)


def update(i):

    angle_dom = ll_angle[i][np.argmax(ll_pv[i])]
    color = pal[int(round(angle_dom/(2*np.pi)*360))%(360)]
    l_color[i] = color

    l.set_data(ll_angle[i], ll_pv[i] )
    l.set_color(l_color[i])
    f.set_xy(np.vstack((ll_angle[i], ll_pv[i]) ).T)

    f.set_color(l_color[i])
    plt.title(l_label[i%(len(l_label))]+ '\n', size=25, color = l_color[i])
    #plt.tight_layout()
    return l,


ani = animation.FuncAnimation(fig, update, frames=len(l_color), interval=10, blit=True)

ani.save('../Results/Animation_enrichment.mp4', writer="ffmpeg")

#plt.show()
