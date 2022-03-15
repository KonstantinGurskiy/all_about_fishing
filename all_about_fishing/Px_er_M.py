import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
#import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from scipy import spatial
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LightSource, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.datasets import make_classification
from numpy import unique
from numpy import where
from matplotlib import pyplot
import os
from statsmodels.tsa.stattools import acf

import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score







def display_colorbar(x, y, z, name, cluster1, cluster2, cluster3):


    ###cmap = plt.cm.copper
    ###ls = LightSource(315, 45)
    ###rgb = ls.shade(z, cmap)

    ###fig, ax = plt.subplots()
    ###ax = plt.subplot(111)

    ###im = ax.pcolormesh(x, y, z, cmap=cmap)
    ###fig.colorbar(im)


    #(unique, counts) = np.unique(nred, axis = 0, return_counts=True)
    #counts = np.array(counts)
    #unique = np.array(unique)
    #z = np.zeros((20,20))


    #for i in range(len(unique[:,0])):
        #z[int(unique[i,1])][int(unique[i,0])] = counts[i]


    #print(y)

    cmap = plt.cm.viridis
    ls = LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    fig, ax = plt.subplots()
    ax = plt.subplot(111)
    im = ax.pcolormesh(x, y, z, cmap=cmap)
    fig.colorbar(im)



    ###ax.scatter(np.take(In, unique[:,0].astype(int))-0.015, np.take(Ip, unique[:,1].astype(int))+0.15, c = 'C0',marker='s', s = 3)

    ###for i in range(len(unique[:,0])):
        ###ax.text(In[int(unique[:,0][i])]-0.015, Ip[int(unique[:,1][i])]+0.15,str(count_blue[i]),fontsize=10,color='white')


    #(unique, counts) = np.unique(ngr, axis = 0, return_counts=True)
    #count_green = np.array(counts)

    #nIn=[]
    #for i in ngr[:,0]:
        #nIn.append(In[int(i)]-0.015)

    #nIp=[]
    #for i in ngr[:,1]:
        #nIp.append(Ip[int(i)]-0.15)


    ####ax.scatter(nIn,nIp, c = 'C2', s = 3)


    #(unique, counts) = np.unique(nor, axis = 0, return_counts=True)
    #count_orange = np.array(counts)


    #nIn=[]
    #for i in nor[:,0]:
        #nIn.append(In[int(i)]+0.015)

    #nIp=[]
    #for i in nor[:,1]:
        #nIp.append(Ip[int(i)]+0.15)


    ####ax.scatter(nIn,nIp, c = 'C1', s = 3)


    #(unique, counts) = np.unique(nred, axis = 0, return_counts=True)
    #count_red = np.array(counts)


    #nIn=[]
    #for i in nred[:,0]:
        #nIn.append(In[int(i)]+0.015)

    #nIp=[]
    #for i in nred[:,1]:
        #nIp.append(Ip[int(i)]-0.15)


    ####ax.scatter(nIn,nIp, c = 'C3', s = 3)


    #(unique, counts) = np.unique(nprpl, axis = 0, return_counts=True)
    #count_purple = np.array(counts)


    #nIn=[]
    #for i in nprpl[:,0]:
        #nIn.append(In[int(i)])

    #nIp=[]
    #for i in nprpl[:,1]:
        #nIp.append(Ip[int(i)])

    ###ax.scatter(nIn,nIp, c = 'C4', s = 3)
   #
    #fig.colorbar(im)


    ax.set_xlabel('$F_{k}$')
    ax.set_ylabel('$n$')

    ax.scatter(F_k_osob[cluster1[:,1]], num_osob[cluster1[:,0]], color='C0', s=7)
    ax.scatter(F_k_osob[cluster2[:,1]], num_osob[cluster2[:,0]]-0.2, color='C1',s=7)
    ax.scatter(F_k_osob[cluster3[:,1]], num_osob[cluster3[:,0]]+0.2, color='C2',s=7)
    ax.set_title(name, size='x-large')
    plt.savefig('log(MdivP)'+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
    plt.show()


def plot_3d(x, y, z, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.

    X, Y = np.meshgrid(x, y)
    #R = np.sqrt(X**2 + Y**2)
    Z = z

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(name + '3D.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)

def plot_xy(x, y, name):

    ff=plt.figure(figsize=(10.0,10.0 * 0.6), frameon=False)
    plt.plot(x, y, '.', color = 'black')
    ax.set_ylim([y_box * per_obr_y, y_box ])

    plt.savefig(name + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
    plt.close(ff)




def read_state(f):
    f.readline()                                                                        # ITEM: TIMESTEP
    f.readline()                                                                        # 0
    f.readline()                                                                        # ITEM: NUMBER OF ATOMS
    N=int(f.readline())
    f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
    xsize = [float(item) for item in f.readline().split(' ')]
    ysize = [float(item) for item in f.readline().split(' ')]
    zsize = [float(item) for item in f.readline().split(' ')]
    f.readline()                                                                        # ITEM: ATOMS id ...
    d=[[float(item) for item in f.readline().strip().split(' ')[1:6]] for i in range(N)]
    return [N,xsize,ysize,d]

def read_er_m_p(f, N_relax, N_run, thermo):
    for i in range(int(N_relax / thermo)):
        f.readline()

    d=[[float(item) for item in f.readline().strip().split('\t')[2::]] for i in range(int(N_run/thermo))]
    return np.array(d)

def get_disp(np_array):
    mean = sum(np_array) / len(np_array)
    D = sum(np.square(np_array - mean) ) / len(np_array)
    return D



log_data = np.arange(0,10, 1)
N_fish = [100]
size_box = [5]
Ip = np.arange(0, 10, 0.5)
In = np.arange(0, 1, 0.05)
d_t = np.array([0.1])
N_relax = np.array([20000])
N_run = np.array([100000])
thermo = np.array([100])
num_osob = np.array([0])
F_k_osob = np.array([0.6])
v_osob = np.array([0.99])
k_f_osob = np.array([0.0])




log_data = np.arange(0,5, 1)
Ip = np.array([3.0])
In = np.arange(0.1, 0.6, 0.1)
In = [0.5]
num_osob = np.array([1, 2, 3, 4, 5, 10, 15, 20])
F_k_osob = np.arange(0.0, 2, 0.1)

path = '/run/media/softmatter/new/Fish/dump/'


#log_0_100_5_3.0_0.0_0.1_20000_100000_100_1_0.0_1.0_0.0.txt





# 2              3              4              5                  6           7             8          9
# 0              1              2               3                  4          5             6
#<mean_x<<"\t"<<mean_y<<"\t"<<mean_px<<"\t"<<mean_py<<"\t"<<mean_erx<<"\t"<<mean_ery<<"\t"<<M<<"\t"<<size<<"\n"

name_log=np.empty((len(log_data),len(In),len(num_osob),len(F_k_osob)), dtype=object)

for i in range(len(log_data)):
    for i_n in range(len(In)):
        for n_osoba in range(len(num_osob)):
            for f_k_os in range(len(F_k_osob)):
                name = path +'log_'+ str(i) + '_100_5_3.0_'+ str(round(In[i_n], 2)) + '_0.1_20000_100000_100_'+ str(num_osob[n_osoba]) + '_'+ str(round(F_k_osob[f_k_os], 1)) + '_1.0_0.0.txt'
                name_log[i][i_n][n_osoba][f_k_os]=name

def read_one_state(i_n, n_osoba, f_k_os):
    P=[]
    M=[]
    Size=[]
    for i in range(len(log_data)+1):
        f = open(path +'log_'+ str(i) + '_100_5_3.0_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_'+ str(n_osoba) + '_'+ str(round(f_k_os, 1)) + '_0.99_0.0.txt' , "r")
        #print(f)
        st = read_er_m_p(f, N_relax, N_run, thermo)
        moment = st[:, 6]
        size = st[:, 7]
        px = st[:, 2]
        py = st[:, 3]
        p = (px*px+py*py)**(-0.5)
        Size.append(size)
        P.append(p)
        M.append(moment)
    return [P,M,Size]


def state():
    p_mean_cont = np.zeros((len(Ip),len(In)))
    moment_mean_cont = np.zeros((len(Ip),len(In)))
    er_mean_cont = np.zeros((len(Ip),len(In)))
    size_mean_cont = np.zeros((len(Ip),len(In)))


    d_p_cont = np.zeros((len(Ip),len(In)))
    d_moment_cont = np.zeros((len(Ip),len(In)))
    d_er_cont = np.zeros((len(Ip),len(In)))
    d_size_cont = np.zeros((len(Ip),len(In)))



    p_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]
    er_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]

    #p_mean_list_y = [[] for i in range(len(In))]
    #er_mean_list_y = [[] for i in range(len(In))]

    moment_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]
    P_list_PM = []
    M_list_PM = []
    P_M_list = [[], [], []]


    ####prove_file = open('name_prove.txt', 'w')
    for lgd in range(len(log_data)):
        for i in range(len(In)): #
            for j in range(len(Ip)): # ip
                for ii in range(len(num_osob)):
                    for jj in range(len(F_k_osob)):
                        #k = lgd*len(In)*len(Ip)+i*len(Ip)+j
                        f = open(name_log[lgd][i][ii][jj], "r")
                        #print(name_log[i][j][ii][jj])
                        #prove_file.write(name_log[k] + '; Ip = ' +str(Ip[j] ) + '; In = ' + str(round(In[i], 2)) + 'loq ='+ str(log_data[lgd]) + ' k = '+ str(k) + '\n')
                        #print(name_log[k], '; Ip = ',str(Ip[j] ), '; In = ', str(round(In[i], 2)), 'loq =', log_data[lgd], ' k = ', str(k))
                        st = read_er_m_p(f, N_relax, N_run, thermo)
                        x_cm = st[:, 0]
                        y_cm = st[:, 1]
                        px = st[:, 2]
                        py = st[:, 3]
                        p_xy = np.sqrt(np.square(px) + np.square(py))
                        erx = st[:, 4]
                        ery = st[:, 5]
                        er_xy = np.sqrt(np.square(erx) + np.square(ery))
                        moment = st[:, 6]
                        size = st[:, 7]

                        p_mean_cont[j, i] += (sum(p_xy)) / len(px)
                        er_mean_cont[j, i] += (sum( er_xy)) / len(px)
                        moment_mean_cont[j, i] += (sum(abs(moment)) / len(moment))
                        size_mean_cont[j, i] += (sum(size)/ len(size))

                        d_p_cont[j, i] += get_disp(p_xy)
                        d_er_cont[j, i] += get_disp( er_xy)
                        d_moment_cont[j, i] += get_disp(moment)
                        d_size_cont[j, i] += get_disp(size)

                        P_list_PM.append(sum(p_xy) / len(px))
                        M_list_PM.append(sum(abs(moment)) / len(moment))
                        #P_M_list[0].append(sum(p_xy) / len(px))
                        #P_M_list[1].append(abs(sum((moment)) / len(moment)))
                        #P_M_list[2].append(np.mean(np.array(size)))
                        P_M_list[0].append(np.max(p_xy[200:]))
                        P_M_list[1].append(abs(np.max((moment[200:]))))
                        P_M_list[2].append(np.max(np.array(size[200:])))
    return(P_M_list)






#st = read_one_state(0.3, 20, 1.0)
#P = np.mean(np.array(st[0]),axis=0)
#M = np.mean(np.array(st[1]),axis=0)
#size = np.mean(np.array(st[2]),axis=0)

#plt.figure()
#plt.plot(M[100:])
#plt.show()

#name_log=[]


###path = '/run/media/softmatter/new/Fish/dump/'
###log_data = np.arange(0,10,1)

###for i in log_data:
    ###for i_n in In:
        ###for i_p in Ip:
            ###name = path +'log_'+ str(i) + '_100_5_'+ str(i_p) + '_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'
            ###name_log.append(name)






###path = '/run/media/softmatter/new/Fish/Fish/dump0/'

###log_data = np.arange(0,10, 1)



###for i in log_data:
    ###for i_n in In:
        ###for i_p in Ip:
            ###name = path +'log_'+ str(i) + '_100_5_'+ str(i_p) + '_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'
            ###name_log.append(name)


###log_data = np.arange(60,80, 1)
###path = '/run/media/softmatter/new/Fish/Fish/dump/'

###for i in log_data:
    ###for i_n in In:
        ###for i_p in Ip:
            ###name = path +'log_'+ str(i) + '_100_5_'+ str(i_p) + '_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_0_0.6_1.0_0.0.txt'
            ###name_log.append(name)


###log_data = np.arange(5,60, 1)
###path = '/run/media/softmatter/new/Fish/Fish/dump2/'
###for i in log_data:
    ###for i_n in In:
        ###for i_p in Ip:
            ###name = path +'log_'+ str(i) + '_100_5_'+ str(i_p) + '_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'
            ###name_log.append(name)

###path = '/run/media/softmatter/new/Fish/Fish/dump3/'
###log_data = np.arange(80,100, 1)

###for i in log_data:
    ###for i_n in In:
        ###for i_p in Ip:
            ###name = path +'log_'+ str(i) + '_100_5_'+ str(i_p) + '_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_0_0.6_1.0_0.0.txt'
            ###name_log.append(name)



length = int(len(name_log)/(len(Ip)*len(In)))

st = state()

#print('/run/media/artur/Новый том/FISH_MD_data/Fish/dump/log_0_100_5_Ip_In_0.1_20000_100000_100_0_0.6_0.99_0.0.txt')
p_mean_cont = np.zeros((len(In),len(Ip)))
moment_mean_cont = np.zeros((len(In),len(Ip)))
er_mean_cont = np.zeros((len(In),len(Ip)))

d_p_cont = np.zeros((len(In),len(Ip)))
d_moment_cont = np.zeros((len(In),len(Ip)))
d_er_cont = np.zeros((len(In),len(Ip)))

p_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]
er_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]

#p_mean_list_y = [[] for i in range(len(In))]
#er_mean_list_y = [[] for i in range(len(In))]

moment_mean_list = [[ [] for i in range(len(In))] for i in range(len(Ip))]
###P_list_PM = []
###M_list_PM = []
###P_M_list = [[], []]
####prove_file = open('name_prove.txt', 'w')
###for lgd in range((length)):
    ###for i in range(len(In)): #
        ###for j in range(len(Ip)): # ip
            ###k = (len(In) *len(Ip)) * lgd + len(Ip)*i + j
            ###f = open(name_log[k], "r")
            ####prove_file.write(name_log[k] + '; Ip = ' +str(Ip[j] ) + '; In = ' + str(round(In[i], 2)) + 'loq ='+ str(log_data[lgd]) + ' k = '+ str(k) + '\n')
            ####print(name_log[k], '; Ip = ',str(Ip[j] ), '; In = ', str(round(In[i], 2)), 'loq =', log_data[lgd], ' k = ', str(k))
            ###st = read_er_m_p(f, N_relax, N_run, thermo)
            ###x_cm = st[:, 0]
            ###y_cm = st[:, 1]
            ###px = st[:, 2]
            ###py = st[:, 3]
            ###p_xy = np.sqrt(np.square(px) + np.square(py))
            ###erx = st[:, 4]
            ###ery = st[:, 5]
            ###er_xy = np.sqrt(np.square(erx) + np.square(ery))
            ###moment = st[:, 6]
            ###size = st[:, 7]

            ###p_mean_cont[j, i] += (sum(p_xy)) / len(px)
            ###er_mean_cont[j, i] += (sum( er_xy)) / len(px)
            ###moment_mean_cont[j, i] += (sum(abs(moment)) / len(moment))

            ###d_p_cont[j, i] += get_disp(p_xy)
            ###d_er_cont[j, i] += get_disp( er_xy)
            ###d_moment_cont[j, i] += get_disp(moment)

            ###P_list_PM.append(sum(p_xy) / len(px))
            ###M_list_PM.append(sum(abs(moment)) / len(moment))
            ###P_M_list[0].append(sum(p_xy) / len(px))
            ###P_M_list[1].append(sum(abs(moment)) / len(moment))

#prove_file.close()

p_mean_cont /= (length)
er_mean_cont /= (length)
moment_mean_cont /= (length)






#print(p_mean_cont)
x = In
y = Ip
###z = p_mean_cont
###display_colorbar(x, y, z, 'P')
###plot_3d(x, y, z, 'P')


###z = er_mean_cont
###display_colorbar(x, y, z, 'er')
###plot_3d(x, y, z, 'er')

###z = moment_mean_cont
###display_colorbar(x, y, z, 'M')
###plot_3d(x, y, z, 'M')


###z = d_p_cont / len(log_data)
###display_colorbar(x, y, z, 'D_P')
###plot_3d(x, y, z, 'D_P')

###z = d_er_cont
###display_colorbar(x, y, z, 'D_er')
###plot_3d(x, y, z, 'D_er')

###z = d_moment_cont / len(log_data)
###display_colorbar(x, y, z, 'D_M')
###plot_3d(x, y, z, 'D_M')

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

#plt.plot(P_list_PM, np.log(M_list_PM),  '.')
#plt.ylim([-5, 12])

#plt.show()

#plt.savefig('P_M_all_graf' + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#print(P_M_list)

P_M_list = np.array(st)
#print(P_M_list)
P=[]
M=[]
Size=[]

P = P_M_list[0]
M = P_M_list[1]
Size = P_M_list[2]
#print(P)
#print(P_M_list[1])
#for i in range(int(len(P_M_list[0])/5)):
    #P = np.append(P, np.mean(np.array(P_M_list[0][i:-1:(len(P_M_list[0]/len(log_data)))])))
    #M = np.append(M, np.mean(np.array(P_M_list[1][i:-1:(len(P_M_list[0]/len(log_data)))])))
    #Size = np.append(Size, np.mean(np.array(P_M_list[2][i:-1:(len(P_M_list[0]/len(log_data)))])))

#P/=np.max(P)
#M/=np.max(M)
#Size/=np.max(Size)
#M=np.log(M)
x=F_k_osob
y=num_osob
#print(len(Size))
#print(num_osob)
#z=[]
#for i in range(int(len(P_M_list[0])/5)):
    #z.append(np.mean(Size[i:-1:160]))
##z=np.reshape(Size, (len(num_osob),len(F_k_osob),len(log_data)),order='C')
##print(z)
#z=np.log(np.reshape(z, (len(num_osob),len(F_k_osob)),order='C'))
#z=np.mean(z,axis=2)
#print(z.shape)


#print(y)
#print(Size)
fig = plt.figure()
ax = fig.add_subplot()
ax = plt.axes(projection ="3d")
#print(Size)
#ax.scatter((Size),(P),(M))
#plt.ylim([-10, -8])
#plt.xlim([-0.005, 0])
cluster1=[]
cluster2=[]
cluster3=[]
cluster4=[]
clusterMSP1=[]
clusterMSP2=[]
clusterMSP3=[]
clusterMSP4=[]
k=0

for lgd in range(len(log_data)):
    for i in range(len(In)): #
        for j in range(len(Ip)): # ip
            for ii in range(len(num_osob)):
                for jj in range(len(F_k_osob)):
                    #k = lgd*len(In)*len(Ip)+i*len(Ip)+j
                    #if(Size[k]>0.2 and P[k]<0.8):
                        #cluster1.append([ii,jj])
                        #clusterMSP1.append([Size[k],P[k],M[k]])
                    #else:
                        #if(P[k]>0.8 and Size[k]<0.2):
                            #cluster2.append([ii,jj])
                            #clusterMSP2.append([Size[k],P[k],M[k]])
                        #else:
                            #if(M[k]>0.5):
                                #cluster3.append([ii,jj])
                                #clusterMSP3.append([Size[k],P[k],M[k]])
                            #else:
                                #cluster4.append([ii,jj])
                                #clusterMSP4.append([Size[k],P[k],M[k]])

                    if(np.log(Size[k]/P[k])>6.5 and M[k]<10000 and F_k_osob[jj]!=0):
                        cluster1.append([ii,jj])
                        clusterMSP1.append([Size[k],P[k],M[k]])
                    else:
                        if(M[k]<10000 and F_k_osob[jj]!=0):
                            cluster2.append([ii,jj])
                            clusterMSP2.append([Size[k],P[k],M[k]])
                    if(F_k_osob[jj]==0):
                        cluster3.append([ii,jj])
                        clusterMSP3.append([Size[k],P[k],M[k]])

                    #ax.text(Size[k], P[k], M[k], str(name_log[lgd][i][ii][jj]), fontsize=1)
                    k+=1




clusterMSP1=np.array(clusterMSP1)
clusterMSP2=np.array(clusterMSP2)
clusterMSP3=np.array(clusterMSP3)
#clusterMSP4=np.array(clusterMSP4)



ax.scatter(clusterMSP1[:,0], clusterMSP1[:,1], clusterMSP1[:,2], color='C0', s=10)
ax.scatter(clusterMSP2[:,0], clusterMSP2[:,1], clusterMSP2[:,2], color='C1',s=10)
ax.scatter(clusterMSP3[:,0], clusterMSP3[:,1], clusterMSP3[:,2], color='C2',s=10)
#ax.scatter(clusterMSP4[:,0], clusterMSP4[:,1], clusterMSP4[:,2], color='C3',s=10)




ax.set_xlabel(r'Size')
ax.set_ylabel(r'P')
ax.set_zlabel(r'M')
#plt.show()
plt.savefig('3d_In='+str(In[0]) + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)

cluster1=np.array(cluster1)
cluster2=np.array(cluster2)
cluster3=np.array(cluster3)
cluster4=np.array(cluster4)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(F_k_osob[cluster1[:,1]]-0.01, num_osob[cluster1[:,0]]-0.1, color='C0', s=10)
ax.scatter(F_k_osob[cluster2[:,1]]+0.01, num_osob[cluster2[:,0]]-0.1, color='C1',s=10)
ax.scatter(F_k_osob[cluster3[:,1]]-0.01, num_osob[cluster3[:,0]]+0.1, color='C2',s=10)
#ax.scatter(F_k_osob[cluster4[:,1]]+0.01, num_osob[cluster4[:,0]]+0.1, color='C3',s=10)
yint = []
locs, labels = plt.yticks()
for each in locs:
    yint.append(int(each))
plt.yticks(yint)
plt.ylim([0,22])
ax.legend(['Разрыв', 'Вытягивание', 'Миллинг'])
plt.savefig('In='+str(In[0]) + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)

plt.show()





zforheatmap=np.mean(np.reshape((M/P),(len(log_data),len(num_osob),len(F_k_osob))), axis=0)


#print(zforheatmap.shape)





####X = P_M_list.transpose()
####print(X)
#blue = []
#green = []
#orange = []
#red = []
#purple = []

#for i in range(len(M)):
    #if((M[i]>-6.5*P[i]+6.5) and (P[i]<0.35) and ((M[i]<-7.5*P[i]+8.125) or (M[i]<-40*P[i]+12))):
        #green.append(i)
    #else:
        #if(((M[i]<-3*P[i]+4.3)or(M[i]<-6*P[i]+5)) and (P[i]<0.53) and (M[i]>2)):
            #orange.append(i)
        #else:
            #if((P[i]>0.8) and (M[i]<-2*P[i]+3) and (M[i]>-2)):
                #red.append(i)
                #print(name_log[i])
            #else:
                #blue.append(i)
    #if((M[i]>5) and (M[i]>5.5) and (P[i]<0.2)):
        #print(name_log[i])



###Mcpy = np.take(M, purple)
###Pcpy = np.take(P, purple)
###pyplot.scatter(Pcpy,Mcpy,c='C4')

#Mcpy = np.take(M,blue)
#Pcpy = np.take(P,blue)
#pyplot.scatter(Pcpy,Mcpy,c='C0')

#Mcpy = np.take(M,green)
#Pcpy = np.take(P,green)
#pyplot.scatter(Pcpy,Mcpy,c='C2')

#Mcpy = np.take(M,orange)
#Pcpy = np.take(P,orange)
#pyplot.scatter(Pcpy,Mcpy,c='C1')

#Mcpy = np.take(M,red)
#Pcpy = np.take(P,red)
#pyplot.scatter(Pcpy,Mcpy,c='C3')

#### define the model
####model = DBSCAN(eps=0.01, min_samples=7)
###### fit model and predict clusters
####yhat = model.fit_predict(np.column_stack((np.array(P),np.array(M))))
###### retrieve unique clusters
####clusters = unique(yhat)
##### create scatter plot for samples from each cluster
####for cluster in clusters:
	##### get row indexes for samples with this cluster
	####row_ix = where(yhat == cluster)
	##### create scatter of these samples
	####pyplot.scatter(P[row_ix], M[row_ix])
#### show the plot
#plt.xlabel(r'P')
#plt.ylabel(r'M')
#plt.ylim([-5, 12])
#pyplot.show()


####blue = np.array(blue)
####green = np.array(green)
####orange = np.array(orange)
####red = np.array(red)
####purple = np.array(purple)


####new_blue = np.empty((len(blue),2))

####for i in range(len(blue)):
    ####new_blue[i]=np.array([round(int(math.floor((blue[i]%(len(In)*len(Ip)))/len(Ip))),2),int((blue[i]%len(Ip)))])




####new_green = np.empty((len(green),2))

####for i in range(len(green)):
    ####new_green[i]=np.array([round(int(math.floor((green[i]%(len(In)*len(Ip)))/len(Ip))),2),int((green[i]%len(Ip)))])



####new_orange = np.empty((len(orange),2))

####for i in range(len(orange)):
    ####new_orange[i]=np.array([round(int(math.floor((orange[i]%(len(In)*len(Ip)))/len(Ip))),2),int((orange[i]%len(Ip)))])



####new_red = np.empty((len(red),2))

####for i in range(len(red)):
    ####new_red[i]=np.array([round(int(math.floor((red[i]%(len(In)*len(Ip)))/len(Ip))),2),int((red[i]%len(Ip)))])



####new_purple = np.empty((len(purple),2))

####for i in range(len(purple)):
    ####new_purple[i]=np.array([round(int(math.floor((purple[i]%(len(In)*len(Ip)))/len(Ip))),2),int((purple[i]%len(Ip)))])

#ph_d.transpose()
#print(ph_d)
#print(new_ph_d)




####z = d_er_cont
y=num_osob
x=F_k_osob
#print(y)
z=zforheatmap
#display_colorbar(x, y, np.log(z), '$log(M/P)$', cluster1, cluster2, cluster3)


# Identify Noise
#n_noise = list(dbscan_cluster.labels_).count(-1)
#print('Estimated no. of noise points: %d' % n_noise)

## Calculating v_measure
#print('v_measure =', v_measure_score(y, labels))



