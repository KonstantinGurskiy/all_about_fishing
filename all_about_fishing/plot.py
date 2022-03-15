import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import pandas as pd
from matplotlib.legend_handler import HandlerPatch

plt.style.use('/run/media/softmatter/new/Fish/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')



def load_data_from_folder(folder):
    def load(name):
        data = []
        with open(f'{folder}/{name}') as f:
            f.readline()
            for line in f:
                data.append([x.replace(' ', '\n') for x in line.split()])
        return np.array(data)
    return load



def plot(i_n):



    In = np.arange(0.1, 0.6, 0.1)
    num_osob = np.array([1, 2, 3, 4, 5, 10, 15, 20])
    F_k_osob = np.arange(0.0, 2, 0.1)



    folder="/run/media/softmatter/new/Fish/"
    name="name_prove.txt"
    load_data = load_data_from_folder(folder)

    y = np.array(load_data(name)[:,2:-1], dtype= 'int')


    fig, ax = plt.subplots()

    phase1=[]
    phase2=[]
    phase3=[]
    phase4=[]
    c0_dot =  plt.Circle((0.0, 0.0), 0.1, facecolor="C0")
    c1_dot =  plt.Circle((0.5, 0.5), 0.1, facecolor="C1")
    c2_dot =  plt.Circle((0.5, 0.5), 0.1, facecolor="C2")
    c3_dot =  plt.Circle((0.5, 0.5), 0.1, facecolor="C3")
    k=0
    for i in range(5):
        for ii in range(len(In)):
            for iii in range(len(num_osob)):
                for iiii in range(len(F_k_osob)):
                    if(y[k][0]==0 and y[k][1]==0):
                        phase1.append([In[ii], num_osob[iii], F_k_osob[iiii]])
                    else:
                        if(y[k][0]==0 and y[k][1]==1):
                            phase2.append([In[ii], num_osob[iii], F_k_osob[iiii]])
                        else:
                            if(y[k][0]==1 and y[k][1]==0):
                                phase3.append([In[ii], num_osob[iii], F_k_osob[iiii]])
                            else:
                                if(y[k][0]==-1 and y[k][1]==1):
                                    phase4.append([round(In[ii],2), num_osob[iii], F_k_osob[iiii]])
                    k+=1



    phase1=np.array(phase1)
    phase2=np.array(phase2)
    phase3=np.array(phase3)
    phase4=np.array(phase4)





    for i in range(len(phase1)):
        if(round(phase1[i][0],1)==i_n):
            ax.scatter(phase1[i,2]+0.01,phase1[i,1]+0.1, color = 'C0', s = 5)
    for i in range(len(phase2)):
        if(round(phase2[i][0],1)==i_n):
            ax.scatter(phase2[i,2]-0.01,phase2[i,1]+0.1, color = 'C1', s = 5)
    for i in range(len(phase3)):
        if(round(phase3[i][0],1)==i_n):
            ax.scatter(phase3[i,2]+0.01,phase3[i,1]-0.1, color = 'C2', s = 5)
    for i in range(len(phase4)):
        if(round(phase4[i][0],1)==i_n):
            ax.scatter(phase4[i,2]-0.01,phase4[i,1]-0.1, color = 'C3', s = 5)


    ax.set_xlabel('$F_{k}$, коэффициент силы')
    ax.set_ylabel('$n$, число специальных рыб')


    ax.legend(handles = [c0_dot, c1_dot, c2_dot, c3_dot],labels=['Частицы не вылетают, фаза не меняется','Частицы не вылетают, фаза меняется','Частицы вылетают резко, фаза не меняется','Частицы вылетают плавно, фаза меняется'], loc=[0,1.0], labelcolor = ['C0','C1','C2','C3'], fontsize=5)

    plt.title('$I_{n} = $' + str(round(i_n,1)), loc='right')

    plt.tight_layout()
    return fig


i_n=0.5
plot(i_n).savefig('/run/media/softmatter/new/Fish/In_'+str(round(i_n,1))+'.pdf')

