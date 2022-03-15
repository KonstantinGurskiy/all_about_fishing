import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import pandas as pd
import seaborn as sns


plt.style.use('/run/media/softmatter/Новый том/Fishes/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')


def load_data_from_folder(folder):
    def load(name):
        data = []
        with open(f'{folder}/{name}') as f:
            for line in f:
                data.append([float(x.replace(',', '\n')) for x in line.split()])
        return np.array(data)
    return load



def readstate():




    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)

    x = (np.array(load_data(name)[:,0])-1000000)*0.0000001
    y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)


    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5001000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1001000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)


    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1020000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)

    vel = (y1+y2+y3)/3

    vel2 = vel**2

    E = vel2/2

    C = np.diff(E)

    x = x[:-1]

    return [x,C]

def getplot(xC):

    fig, ax = plt.subplots()

    variance = np.empty(xC[1][:-500].size)

    for i in range(xC[1][:-500].size):

        variance[i] = np.std(xC[1][i:(i+500)])


    ###ax.plot(xC[0][:-500], variance, color='C0', marker='', linestyle='-', label=r"C")
    ax.plot(xC[0], gaussian_filter(xC[1],sigma = 100), color='C0', marker='', linestyle='-', label=r"C")

    #ax2 = ax.twinx()
    #ax2.plot(xC[0], xC[1], color='C1', marker='*', linestyle='', label=r"C")
    ax.axvline(0.033,linestyle='-',color='black')
    ax.axvline(0.10,linestyle='-',color='black')
    ax.axvline(xC[0][np.argmax(variance)],linestyle='-',color='black')
    ax.axvline(0.305,linestyle='-',color='black')


    ax.legend(loc=1, labelcolor='markeredgecolor')


    ###ax.text(0.018, 0.0018,  r'I',  fontsize=10)
    ###ax.text(0.064, 0.0018, r'II', fontsize=10)
    ###ax.text(0.115, 0.0018, r'III', fontsize=10)
    ###ax.text(0.215, 0.0006, r'IV', fontsize=10)
    ###ax.text(0.343, 0.001, r'V', fontsize=10)


    ax.set_xlabel(r'$k_{\alpha}$, koefficient k of active force')
    ###ax.set_ylabel('$\sigma_C$, deviation of heat capacity', color='C0')
    ax.set_ylabel('$C$, heat capacity', color='C0')
    ax.set_xlim(left=0, right=max(xC[0]))

    plt.tight_layout()

    return fig








getplot(readstate()).savefig('/run/media/softmatter/Новый том/Fishes/n=1/heat_capacity.pdf')

#getplot(readstate()).show()

