import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import pandas as pd

plt.style.use('/run/media/softmatter/Новый том/Fishes/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')


def load_data_from_folder(folder):
    def load(name):
        data = []
        with open(f'{folder}/{name}') as f:
            for line in f:
                data.append([float(x.replace(',', '\n')) for x in line.split()])
        return np.array(data)
    return load


def noise():

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"

    name="data.npy"

    dt=np.load(folder+name)

    k = dt[:,0]
    P1 = dt[:,1]
    M1 = dt[:,2]

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"

    name="data.npy"

    dt=np.load(folder+name)

    P2 = dt[:,1]
    M2 = dt[:,2]

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"

    name="data.npy"

    dt=np.load(folder+name)

    P3 = dt[:,1]
    M3 = dt[:,2]


    P = (P1+P2+P3)/3
    M = (M1+M2+M3)/3

    sigma_P = np.empty(P.size-100)
    sigma_M = np.empty(M.size-100)

    for i in range(sigma_P.size):
        sum_P = 0
        sum_M = 0

        for ii in range(100):
            sum_P += P[i+ii]
            sum_M += M[i+ii]

        mean_P = sum_P/100
        mean_M = sum_M/100

        sum_P = 0
        sum_M = 0

        for ii in range(100):
            sum_P += ((P[i+ii]-mean_P)**2)
            sum_M += ((M[i+ii]-mean_M)**2)

        sigma_P[i] = (sum_P/100)**(0.5)
        sigma_M[i] = (sum_M/100)**(0.5)

    return np.array([k,sigma_P,sigma_M])


def plot(arr):

    fig, ax = plt.subplots()

    ax.plot(arr[0][50:-50], gaussian_filter(arr[1],sigma = 300), color='C0', marker='', linestyle='-', label=r"${\sigma_P}$")










    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    x = (np.array(load_data(name)[:,0])-1000000)*0.0000001

    y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)


    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5001000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1001000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    y2 = y2

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1020000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    y3 = y3



    y=(y1+y2+y3)/3



    y = gaussian_filter(y, sigma = 100)
    ax.plot(x, y/2, color='C3', marker='', linestyle='-', label=r"<V>(t)")






    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"

    name="data.npy"

    dt=np.load(folder+name)

    x = dt[:,0]
    y11 = dt[:,1]
    y21 = dt[:,2]

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"

    name="data.npy"

    dt=np.load(folder+name)


    y12 = dt[:,1]
    y22 = dt[:,2]

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"

    name="data.npy"

    dt=np.load(folder+name)


    y13 = dt[:,1]
    y23 = dt[:,2]


    y1=(y11+y12+y13)/3
    y2=(y21+y22+y23)/3



    ax.plot(x, gaussian_filter(y1, sigma = 100)/10, color='C2', marker='', linestyle='-', label=r"P")










    ax.set_xlabel(r'$k_{\alpha}$, koefficient k of active force')
    ax.set_ylabel(r'')


    ax.legend(loc=1, labelcolor='markeredgecolor')



    plt.tight_layout()

    return fig

plot(noise()).savefig('/run/media/softmatter/Новый том/Fishes/n=1/noise.pdf')








