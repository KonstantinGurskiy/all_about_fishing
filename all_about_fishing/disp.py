import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import curve_fit
import statistics

plt.style.use('/run/media/softmatter/Новый том/Fishes/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')

folder="/run/media/softmatter/Новый том/Fishes/"

name="AvD_AvVA_AvVR1"

def load_data_from_folder(folder):
    def load(name):
        data = []
        with open(f'{folder}/{name}') as f:
            for line in f:
                data.append([float(x.replace(' ', '\n')) for x in line.split()])
        return np.array(data)
    return load


load_data = load_data_from_folder(folder)


#def func(x, a, b):
    #return a*x+b


def y_disp():
    with open("/run/media/softmatter/Новый том/Fishes/1..20/fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_4k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0_py.lammpstrj") as f:
        data=[]
        dist=[]
        for i in range(39999):
            print(i)
            for ii in range(9):
                f.readline()
            for ii in range(128):
                data.append([float(x.replace(' ', '\n')) for x in f.readline().split()][2])
            dist.append(statistics.stdev(data))
            data=[]
    return dist

def x_disp():
    with open("/run/media/softmatter/Новый том/Fishes/1..20/fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_4k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0_py.lammpstrj") as f:
        data=[]
        dist=[]
        for i in range(39999):
            print(i)
            for ii in range(9):
                f.readline()
            for ii in range(128):
                data.append([float(x.replace(' ', '\n')) for x in f.readline().split()][1])
            dist.append(statistics.stdev(data))
            data=[]
    return dist


def disp():
    fig, ax = plt.subplots()
    x = np.array(load_data(name)[:,0])*0.000001
    y = np.array(load_data(name)[:,1])
    #print(x)
    ax.set_ylim(bottom=min(y), top=20)
    ax.set_xlim(left=min(x), right=max(x))

    #fig, ax = plt.subplots()

    ax.plot(x, y, color='C1', marker='', linestyle='dotted', label=r"<D>(t)")
    x = np.array(load_data(name)[:,0])*0.000001
    #y = np.array(load_data(name)[:,1])
    y=np.array(y_disp())


    #xdata=x[31000:-1]
    #ydata=y[31000:-1]

    #popt, pcov = curve_fit(func, xdata, ydata,maxfev=5000)

    #ax.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.7f, b=%5.7f' % tuple(popt))

    #xdata=x[2500:31000]
    #ydata=y[2500:31000]

    #popt, pcov = curve_fit(func, xdata, ydata,maxfev=5000)
    #ax.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.7f, b=%5.7f' % tuple(popt))

    #xdata=x[:2500]
    #ydata=y[:2500]

    #popt, pcov = curve_fit(func, xdata, ydata,maxfev=5000)
    #ax.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.7f, b=%5.7f' % tuple(popt))




    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.legend()
    #plt.show()

    #fig, ax = plt.subplots()

    ax.plot(x, y, color='C0', marker='', linestyle='dotted', label=r"$\sigma_y$")

    #print(sum(y.transpose()))




    ax.legend(loc=2, labelcolor='markeredgecolor')



    #print(sum(y.transpose()))





    y=np.array(x_disp())
    ax.plot(x, y, color='C2', marker='', linestyle='dotted', label=r"$\sigma_x$")
    ax.axvline(0.0025,linestyle='dashed',color='grey')
    ax.axvline(0.0315,linestyle='dashed',color='grey')
    ax.set_xlabel(r'$k_{\alpha}$, active force module')
    #ax.set_ylabel(r'$<D>$')


    ax.legend(loc=2, labelcolor='markeredgecolor')

    return fig



disp().savefig('dispersion.pdf')
