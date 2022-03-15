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
from pylab import*
from mpl_toolkits.mplot3d import Axes3D


plt.style.use('/run/media/softmatter/Новый том/Fishes/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')






def readstate(folder, name, N):

    f = open(folder + name+ '.lammpstrj', "r")
    cut = N - 1000 - 1
    N-=1
    mapsrc = np.zeros((200,200))
    for i in range(cut):
        for ii in range(9+128):
            f.readline()
    for i in range(N-cut):
        for ii in range(9):
            f.readline()
        for ii in range(128):
            d=[float(item) for item in f.readline().strip().split(' ')]
            mapsrc[int(d[2]+100)][int(d[1]+100)]+=1

    #mapsrc/=mapsrc.max()

    return mapsrc





def readstatepolar(folder, name,N):

    f = open(folder + name+ '.lammpstrj', "r")
    cut = N-1000-1
    N = N-1

    mapsrc = np.zeros((1000,1000))
    for i in range(cut):
        for ii in range(9+128):
            f.readline()
    for i in range(N-cut):
        for ii in range(9):
            f.readline()
        for ii in range(128):
            d = [float(item) for item in f.readline().strip().split(' ')]
            th = d[5]
            th=th%(2*np.pi)
            #print(th)
            #print(int(1000*th/2*np.pi))
            mapsrc[int(1000*th/(2*np.pi))][int(math.hypot(d[1],d[2])*10)] += 1

    #mapsrc/=mapsrc.max()

    th=np.linspace(0,2*np.pi,1000)
    r=np.linspace(0,100,1000)
    return [mapsrc,th,r]


def getpolarmap():

    folder = '/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/'
    name = 'fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0_py'

    N=1000



    st1=readstatepolar(folder,name,1000)
    st2=readstatepolar(folder,name,7000)
    st3=readstatepolar(folder,name,13000)
    st4=readstatepolar(folder,name,25000)
    st5=readstatepolar(folder,name,40000)


    theta=st1[1]
    rad=st1[2]


    #print(st1[0])
    #fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.imshow(st1[0], cmap='viridis', interpolation='nearest')
    #ax.set_rmax(100)




    #plt.show()

    fig = plt.figure()

    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
    #gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)





    rad, theta = np.meshgrid(rad, theta) #rectangular plot of polar data
    Y = rad
    X = theta
    fig = plt.figure()

    ax1 = fig.add_subplot(221, projection='polar')
    ax1.pcolormesh(X, Y, st1[0]**(-1))
    ax2 = fig.add_subplot(222, projection='polar')
    ax2.pcolormesh(X, Y, st2[0]**(-1))
    ax3 = fig.add_subplot(223, projection='polar')
    ax3.pcolormesh(X, Y, st4[0]**(-1))
    ax4 = fig.add_subplot(224, projection='polar')
    ax4.pcolormesh(X, Y, st5[0]**(-1))
    #plt.show()


    ax1.set_ylim(0, 40)
    ax2.set_ylim(0, 40)
    ax3.set_ylim(0, 40)
    ax4.set_ylim(0, 40)


    ax1.tick_params(labelsize=5)
    ax2.tick_params(labelsize=5)
    ax3.tick_params(labelsize=5)
    ax4.tick_params(labelsize=5)


    ax1.grid(color='grey', linestyle='-', linewidth=0.02)
    ax2.grid(color='grey', linestyle='-', linewidth=0.02)
    ax3.grid(color='grey', linestyle='-', linewidth=0.02)
    ax4.grid(color='grey', linestyle='-', linewidth=0.02)

    ax1.text(2.8, 70, r'I', fontsize=10)
    ax2.text(2.8, 70, r'II', fontsize=10)
    ax3.text(2.8, 70, r'IV', fontsize=10)
    ax4.text(2.8, 70, r'V', fontsize=10)


    return fig




def getmap():


    folder = '/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/'
    name = 'fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0_py'


    N=[1000, 7000, 13000, 25000, 40000]

    st1=readstate(folder,name,1000)
    st2=readstate(folder,name,7000)
    st3=readstate(folder,name,13000)
    st4=readstate(folder,name,25000)
    st5=readstate(folder,name,40000)

    fig = plt.figure()

    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)

    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col')

    ax1.imshow(st1**(-1), cmap='viridis', interpolation='nearest')
    ax2.imshow(st2**(-1), cmap='viridis', interpolation='nearest')
    ax3.imshow(st4**(-1), cmap='viridis', interpolation='nearest')
    ax4.imshow(st5**(-1), cmap='viridis', interpolation='nearest')
    #ax5.imshow(st5**(-1), cmap='hot', interpolation='nearest')


    ax1.set_xlim(60, 140)
    ax1.set_ylim(60, 140)
    ax2.set_xlim(60, 140)
    ax2.set_ylim(60, 140)
    ax3.set_xlim(60, 140)
    ax3.set_ylim(60, 140)
    ax4.set_xlim(60, 140)
    ax4.set_ylim(60, 140)




    return fig



getpolarmap().savefig('/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/allinone.pdf')
