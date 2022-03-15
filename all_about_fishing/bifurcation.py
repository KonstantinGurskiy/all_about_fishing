import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import pandas as pd
from tabulate import tabulate
import os
from scipy.spatial import distance

plt.style.use('/run/media/softmatter/Новый том1/Fish/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')

if(os.path.isfile("max_distance.txt")):
    os.remove("max_distance.txt")

if(os.path.isfile("statistics.txt")):
    os.remove("statistics.txt")

if(os.path.isfile("std.txt")):
    os.remove("std.txt")

def load_data_from_folder(folder):
    def load(name):
        data = []
        with open(f'{folder}/{name}') as f:
            for line in f:
                data.append([float(x.replace(',', '\n')) for x in line.split()])
        return np.array(data)
    return load

def read_dispersion():
    with open("std.txt", "r") as f:
        disp=np.empty((20,20))
        i=0
        for In in np.arange(0,1,0.05):
            ii=0
            for Ip in np.arange(0,10,0.5):
                disp[i][ii]=f.readline().split()[2]
                ii+=1
            i+=1
    return disp


def read_distance():
    with open("max_distance.txt", "r") as f:
        disp=np.empty((20,20))
        i=0
        for In in np.arange(0,1,0.05):
            ii=0
            for Ip in np.arange(0,10,0.5):
                disp[i][ii]=math.log(float(f.readline().split()[2]))
                ii+=1
            i+=1
    return disp


def dispersion(In, Ip):
    folder="/run/media/softmatter/Новый том1/Fish/dump"

    y = np.zeros((10,7))
    for Nlog in range(10):
        name='log_'+str(Nlog)+'_100_5_'+str(round(Ip, 2))+'_'+str(round(In, 3))+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'
        load_data = load_data_from_folder(folder)



        px = np.array(load_data(name)[:,4])
        py = np.array(load_data(name)[:,5])
        erx = np.array(load_data(name)[:,6])
        ery = np.array(load_data(name)[:,7])
        abs_M = np.absolute(np.array(load_data(name)[:,8]))
        size = np.array(load_data(name)[:,9])
        M = np.array(load_data(name)[:,8])

        y[Nlog] = np.array([str(round(Ip, 2)),str(round(In, 3)),np.mean((px**2+py**2)**(1/2)), np.mean((erx**2+ery**2)**(1/2)), np.mean(abs_M), np.mean(size), np.mean(M)])



    with open("statistics.txt", "a") as o:
        o.write(tabulate(y, headers=["Ip", "In", "p", "er", "abs_M", "size", "M"]))
        o.write('\n\n')

    std = np.std(y[:,2])

    with open("std.txt", "a") as o:
        o.write(str(Ip)+'\t'+str(In)+'\t'+str(std))
        o.write('\n')
    return std


def max_distance(In,Ip):


    folder="/run/media/softmatter/Новый том1/Fish/dump"

    max_dist = np.zeros((10))
    for Nlog in range(10):
        name='log_'+str(Nlog)+'_100_5_'+str(round(Ip, 2))+'_'+str(round(In, 3))+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'
        load_data = load_data_from_folder(folder)



        mean_x = np.array(load_data(name)[:,2])
        mean_y = np.array(load_data(name)[:,3])


        max_dist[Nlog] = np.amax(distance.cdist([mean_x,mean_y], [mean_x,mean_y], 'euclidean'))

    max_distance = math.log(max_dist.mean())

    ###print(max_distance)


    with open("max_distance.txt", "a") as o:
        o.write(str(Ip)+'\t'+str(In)+'\t'+str(max_distance))
        o.write('\n')
    return max_distance


disp=np.empty((20,20))
i=0
for In in np.arange(0,1,0.05):
    ii=0
    for Ip in np.arange(0,10,0.5):
        disp[i][ii]=max_distance(In, Ip)
        ii+=1
    i+=1

###disp = read_distance()

#print(disp)
fig, ax = plt.subplots()



ax.pcolor(disp.transpose(), cmap='viridis')


ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20), labels=np.around(np.arange(0,10,0.5),2),fontsize=6)
ax.set_xticklabels(labels=np.around(np.arange(0,1,0.05),2), rotation=45,fontsize=6)
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         #rotation_mode="anchor")


i=0
for In in np.arange(0,1,0.05):
    ii=0
    for Ip in np.arange(0,10,0.5):
        text = ax.text(i+0.5, ii+0.5, round(disp[i][ii],2),
                       ha="center", va="center", color="w",fontsize=2)
        ii+=1
    i+=1



ax.set_ylabel(r'$I_p$')
ax.set_xlabel(r'$I_n$')
#plt.yticks(rotation=90)


fig.tight_layout()


fig.savefig('max_distance.pdf')


