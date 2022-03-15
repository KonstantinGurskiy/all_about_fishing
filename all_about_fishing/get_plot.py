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



def usual_spec(load_data,name):
    y1 = np.array(load_data(name)[1])
    #print(x1)
    x1 = np.arange(0,1,0.001)
    #print(x1)
    folder="/run/media/softmatter/Новый том/Fishes/"
    name="usual_distribution"
    load_data = load_data_from_folder(folder)

    y2 = np.array(load_data(name)[1])


    fig, ax = plt.subplots()

    ax.plot(x1, y1/sum(y1), color='C0', marker='', linestyle='dotted', label=r"$P_{usual}$")
    ax.plot(x1, y2/sum(y2), color='C1', marker='', linestyle='dotted', label=r"$P_{spec}$")

    ax.set_ylim(bottom=0, top=0.01)
    ax.set_xlim(left=0, right=1)

    ax.set_xlabel(r'Z, относительное положение частиц на оси Z')
    ax.set_ylabel(r'$P(Z)$, веротяность попадания частицы ')


    ax.legend(loc=2, labelcolor='markeredgecolor')

    #style_arrow = mpatches.ArrowStyle('-|>', head_length=.2, head_width=.08)

    #ax.annotate('Text',
                #xy=(x, data[7]), xycoords='data',
                #xytext=(0, 20), textcoords='offset points',
                #arrowprops=dict(arrowstyle=style_arrow, lw=0.8, fc="k", ec="k"),
                #bbox=dict(pad=-0.5, facecolor="white", edgecolor="white"),
                #ha='center', va='top')

    #plt.scatter(x,data[7])

    plt.tight_layout()
    return fig

def theta():
    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="theta_distribution"+str(1)
    load_data = load_data_from_folder(folder)
    y1 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"
    name="theta_distribution"+str(1)
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"
    name="theta_distribution"+str(1)
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name))
    x = np.arange(0,math.pi,math.pi/100)
    fig, ax = plt.subplots()
    y=(y1+y2+y3)/3

    ax.plot(x, y.transpose(), color='C0', marker='', linestyle='-', label=r"$P_{theta1}$")











    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="theta_distribution"+str(10)
    load_data = load_data_from_folder(folder)
    y1 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"
    name="theta_distribution"+str(10)
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"
    name="theta_distribution"+str(10)
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name))
    x = np.arange(0,math.pi,math.pi/100)

    y=(y1+y2+y3)/3

    ax.plot(x, y.transpose(), color='C1', marker='', linestyle='-', label=r"$P_{theta10}$")















    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="theta_distribution"+str(20)
    load_data = load_data_from_folder(folder)
    y1 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"
    name="theta_distribution"+str(20)
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"
    name="theta_distribution"+str(20)
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name))
    x = np.arange(0,math.pi,math.pi/100)

    y=(y1+y2+y3)/3

    ax.plot(x, y.transpose(), color='C2', marker='', linestyle='-', label=r"$P_{theta20}$")















    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="theta_distribution"+str(30)
    load_data = load_data_from_folder(folder)
    y1 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"
    name="theta_distribution"+str(30)
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"
    name="theta_distribution"+str(30)
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name))
    x = np.arange(0,math.pi,math.pi/100)

    y=(y1+y2+y3)/3

    ax.plot(x, y.transpose(), color='C3', marker='', linestyle='-', label=r"$P_{theta30}$")












    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="theta_distribution"+str(40)
    load_data = load_data_from_folder(folder)
    y1 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2/"
    name="theta_distribution"+str(40)
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name))

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/"
    name="theta_distribution"+str(40)
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name))
    x = np.arange(0,math.pi,math.pi/100)

    y=(y1+y2+y3)/3

    ax.plot(x, y.transpose(), color='C4', marker='', linestyle='-', label=r"$P_{theta40}$")










    #print(sum(y.transpose()))

    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=math.pi)

    ax.set_xlabel(r'$\theta$, particle orientation angle')
    ax.set_ylabel(r'$P(\theta)$, probability of occurence')


    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig


def dis():
    folder="/run/media/softmatter/Новый том/Fishes/2..20;2/"
    name="AvD_AvVA_AvVR20"
    load_data = load_data_from_folder(folder)
    x = np.array(load_data(name)[:,0])*0.00001
    y1 = np.array(load_data(name)[:,1])
    #print(x)
    ###name="AvD_AvVA_AvVR2"
    ###load_data = load_data_from_folder(folder)
    ###y2 = np.array(load_data(name)[:,1])
    ###name="AvD_AvVA_AvVR3"
    ###load_data = load_data_from_folder(folder)
    ###y3 = np.array(load_data(name)[:,1])
    ###name="AvD_AvVA_AvVR4"
    ###load_data = load_data_from_folder(folder)
    ###y4 = np.array(load_data(name)[:,1])
    ###name="AvD_AvVA_AvVR5"
    ###load_data = load_data_from_folder(folder)
    ###y5 = np.array(load_data(name)[:,1])
    fig, ax = plt.subplots()

    ax.plot(x, y1, color='C0', marker='', linestyle='dotted')
    ###ax.plot(x, y2, color='C1', marker='', linestyle='dotted')
    ###ax.plot(x, y3, color='C2', marker='', linestyle='dotted')
    ###ax.plot(x, y4, color='C4', marker='', linestyle='dotted')
    ###ax.plot(x, y5, color='C5', marker='', linestyle='dotted')
    ax.plot(x, (y1), color='C3', marker='', linestyle='dotted', label=r"<D>(t)")
    ###ax.axvline(0.025,linestyle='dashed',color='grey')
    ###ax.axvline(0.315,linestyle='dashed',color='grey')
    #print(sum(y.transpose()))

    ax.set_ylim(bottom=0, top=max(y1))
    ax.set_xlim(left=0, right=max(x))

    ax.set_xlabel(r'$k_{\alpha}$')
    ax.set_ylabel(r'<D>, average distance between particles')


    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return plt

def vel():

    fig, ax = plt.subplots()






    folder="/run/media/softmatter/Новый том/Fish_school/Rez/"
    name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_50000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-08cut_10000000extra_fishes_0beta_0.000000square_0.txt"
    load_data = load_data_from_folder(folder)
    x = (np.array(load_data(name)[:,0])-10000000)*0.00000001

    y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)


    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5001000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1001000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y2 = y2

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1020000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y3 = y3



    ##y=(y1+y2+y3)/3

    y=y1

    y = gaussian_filter(y, sigma = 100)
    ax.plot(x, y, color='C0', marker='', linestyle='-', label=r"<V>(t)")



    ax2 = ax.twinx()

    folder="/run/media/softmatter/Новый том/Fish_school/Rez/"
    name="AvD_AvVA_AvVR1"
    load_data = load_data_from_folder(folder)
    x = np.array(load_data(name)[:,0])*0.000001
    y1 = np.array(load_data(name)[:,1])


    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.array(load_data(name)[:,1])

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.array(load_data(name)[:,1])

    yd=y1

    ##yd = (y1+y2+y3)/3
    yd = gaussian_filter(yd, sigma = 100)
    ax2.plot(x, yd, color='C3', marker='', linestyle='-', label=r"<D>(t)")
    ax.plot(x, yd, color='C3', marker='', linestyle='-', label=r"<D>(t)")


    ###ax.axvline(0.128,linestyle='dashed',color='black')
    ###ax.axvline(0.15,linestyle='dashed',color='black')
    ###ax.axvline(0.16,linestyle='dashed',color='black')
    ###ax.axvline(0.237,linestyle='dashed',color='black')
    ###ax.axvline(0.272,linestyle='dashed',color='black')

    #####ax.text(0.018, 0.18,  r'I',  fontsize=10)
    #####ax.text(0.064, 0.18, r'II', fontsize=10)
    #####ax.text(0.115, 0.18, r'III', fontsize=10)
    #####ax.text(0.215, 0.12, r'IV', fontsize=10)
    #####ax.text(0.343, 0.065, r'V', fontsize=10)



    ##ax2.spines["right"].set_color('C3')
    ##ax2.tick_params(axis='y', colors='C3')
    ##ax2.set_ylabel('<D>, distance', color='C3')

    ##ax2.spines["left"].set_color('C0')
    ##ax.tick_params(axis='y', colors='C0')


    ##ax.legend(loc=2, labelcolor='markeredgecolor')










    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1000000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##x = (np.array(load_data(name)[:,0])-1000000)*0.0000001
    ##x = x
    ##y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y1 = y1

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5001000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1001000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y2 = y2

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1020000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y3 = y3



    ##y=(y1+y2+y3)/3

    ##y = gaussian_filter(y, sigma = 100)
    ##ax.plot(x, y, color='C0', marker='', linestyle='-', label=r"<V>(t)")



    ##ax2 = ax.twinx()

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##x = np.array(load_data(name)[:,0])*0.00001
    ##x = x
    ##y1 = np.array(load_data(name)[:,1])
    ##y1 = y1

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.array(load_data(name)[:,1])

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.array(load_data(name)[:,1])



    ##yd = (y1+y2+y3)/3
    ##yd = gaussian_filter(yd, sigma = 100)
    ##ax2.plot(x, yd, color='C3', marker='', linestyle='-', label=r"<D>(t)")
    ##ax.plot(x, yd, color='C3', marker='', linestyle='-', label=r"<D>(t)")

    ##ax.axvline(0.04,linestyle='dashed',color='black')
    ##ax.axvline(0.10,linestyle='dashed',color='black')
    ##ax.axvline(0.15,linestyle='dashed',color='black')
    ##ax.axvline(0.30,linestyle='dashed',color='black')



    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez1"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_1000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-07cut_200000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##x = (np.array(load_data(name)[:,0])-200000)*0.0000005
    ##x = x
    ##y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y1 = y1

    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_1001000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-07cut_200000extra_fishes_0beta_0.000000square_0.txt"
    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez2"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y2 = y2

    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_1002000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-07cut_200000extra_fishes_0beta_0.000000square_0.txt"
    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez3"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y3 = y3




    ##y=(y1+y2+y3)/3

    ##y = gaussian_filter(y, sigma = 50)
    ##ax.plot(x, y, color='C1', marker='', linestyle='-', label=r"<$V_{short}$>(t)")



    ##ax2 = ax.twinx()

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez1"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##x = np.array(load_data(name)[:,0])*0.00005

    ##x = x
    ##y1 = np.array(load_data(name)[:,1])
    ##y1 = y1

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez2"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.array(load_data(name)[:,1])

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/short/Rez3"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.array(load_data(name)[:,1])



    ##yd = (y1+y2+y3)/3
    ##yd = gaussian_filter(yd, sigma = 50)
    ##ax2.plot(x, yd, color='C4', marker='', linestyle='-', label=r"<$D_{short}$>(t)")
    ##ax.plot(x, yd, color='C4', marker='', linestyle='-', label=r"<$D_{short}$>(t)")

    ##ax.axvline(0.03,linestyle='dashed',color='black')
    ##ax.axvline(0.10,linestyle='dashed',color='black')
    ##ax.axvline(0.16,linestyle='dashed',color='black')
    ##ax.axvline(0.32,linestyle='dashed',color='black')







    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez1"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_10000000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-08cut_2000000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##x = (np.array(load_data(name)[:,0])-2000000)*0.00000005
    ##x = x
    ##y1 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y1 = y1

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez2"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_10010000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-08cut_2010000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y2 = y2

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez3"
    ##name="log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_10020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_5e-08cut_2020000extra_fishes_0beta_0.000000square_0.txt"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.sqrt(np.array(load_data(name)[:,7])**2+np.array(load_data(name)[:,6])**2)
    ##y3 = y3


    ##y=(y1+y2+y3)/3

    ##y = gaussian_filter(y, sigma = 500)
    ##ax.plot(x, y, color='C2', marker='', linestyle='-', label=r"<$V_{long}$>(t)")



    ##ax2 = ax.twinx()

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez1"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##x = np.array(load_data(name)[:,0])*0.000005
    ##x = x
    ##y1 = np.array(load_data(name)[:,1])
    ##y1 = y1

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez2"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y2 = np.array(load_data(name)[:,1])

    ##folder="/run/media/softmatter/Новый том/Fishes/n=1/long/Rez3"
    ##name="AvD_AvVA_AvVR1"
    ##load_data = load_data_from_folder(folder)
    ##y3 = np.array(load_data(name)[:,1])



    ##yd = (y1+y2+y3)/3
    ##yd = gaussian_filter(yd, sigma = 500)
    ##ax2.plot(x, yd, color='C5', marker='', linestyle='-', label=r"<$D_{long}$>(t)")
    ##ax.plot(x, yd, color='C5', marker='', linestyle='-', label=r"<$D_{long}$>(t)")

    ##ax.axvline(0.04,linestyle='dashed',color='black')
    ##ax.axvline(0.10,linestyle='dashed',color='black')
    ##ax.axvline(0.16,linestyle='dashed',color='black')
    ##ax.axvline(0.30,linestyle='dashed',color='black')






    ax2.spines["right"].set_color('C3')
    ax2.tick_params(axis='y', colors='C3')
    ax2.set_ylabel('<D>, distance', color='C3')



    ax2.spines["left"].set_color('C0')
    ax.tick_params(axis='y', colors='C0')


    ax.legend(loc=2, labelcolor='markeredgecolor')


    ax.set_ylim(bottom=min(y), top=max(y))
    ax.set_xlim(left=min(x), right=max(x))

    ax.set_xlabel(r'$k_{\alpha}$')
    ax.set_ylabel(r'<V>, velocity', color = 'C0')



    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig

def thetadist():
    x = np.array(load_data(name)[:,0])
    y = np.array(load_data(name)[:,1])

    fig, ax = plt.subplots()

    ax.plot(x, y, color='C0', marker='', linestyle='dotted', label=r"$\sigma_{\theta}$")

    #print(sum(y.transpose()))

    ax.set_ylim(bottom=min(y), top=max(y))
    ax.set_xlim(left=min(x), right=max(x))

    ax.set_xlabel(r't, timestep')
    ax.set_ylabel(r'$\sigma_{\theta}, \theta$ distribution')


    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig



def phase_diagram():
    folder="/run/media/softmatter/Новый том/Fishes/2..20;2"

    name="p_d"
    load_data = load_data_from_folder(folder)
    x = np.array(load_data(name)[:,0])/2
    y1 = np.array(load_data(name)[:,1])
    y2 = np.array(load_data(name)[:,2])

    fig, ax = plt.subplots()

    ax.plot(x, y1, color='C0', marker='', linestyle='dotted', label=r"low phase crossing point256")
    ax.plot(x, y2, color='C1', marker='', linestyle='dotted', label=r"high phase crossing point256")
    #ax.fill_between(x,y1,y2,color='C1')
    #ax.fill_between(x,0,y1,color='C0')
    #ax.fill_between(x,y1,y2,color='C2')
    #print(sum(y.transpose()))

    ax.set_ylim(bottom=0, top=0.4)
    ax.set_xlim(left=0, right=max(x))
    folder="/run/media/softmatter/Новый том/Fishes/"

    name="p_d"

    load_data = load_data_from_folder(folder)
    x = np.array(load_data(name)[:,0])
    y1 = np.array(load_data(name)[:,1])
    y2 = np.array(load_data(name)[:,2])
    ax.plot(x, y1, color='C3', marker='', linestyle='dotted', label=r"low phase crossing point128")
    ax.plot(x, y2, color='C4', marker='', linestyle='dotted', label=r"high phase crossing point128")
    ax.set_xlabel(r'n, number of special particles')
    ax.set_ylabel(r'$k_{\alpha}$, koefficient k of active force')


    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig




def P_M():
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

    #load_data = load_data_from_folder(folder)
    #x = np.array(load_data(name)[:,0])
    #y1 = np.array(load_data(name)[:,1])
    #y2 = np.array(load_data(name)[:,2])

    fig, ax = plt.subplots()

    ax.plot(x, gaussian_filter(y1, sigma = 100), color='C2', marker='', linestyle='-', label=r"P")
    ax.plot(x, gaussian_filter(y2*5000, sigma = 200), color='C1', marker='', linestyle='-', label=r"M")
    #ax.fill_between(x,y1,y2,color='C1')
    #ax.fill_between(x,0,y1,color='C0')
    #ax.fill_between(x,y1,y2,color='C2')
    #print(sum(y.transpose()))

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



    y = gaussian_filter(y*6.5, sigma = 100)
    ax.plot(x, y, color='C0', marker='', linestyle='-', label=r"<V>(t)")

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/"
    name="AvD_AvVA_AvVR1"
    load_data = load_data_from_folder(folder)
    x = np.array(load_data(name)[:,0])*0.00001
    y1 = np.array(load_data(name)[:,1])


    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez2"
    name="AvD_AvVA_AvVR1"
    load_data = load_data_from_folder(folder)
    y2 = np.array(load_data(name)[:,1])

    folder="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3"
    name="AvD_AvVA_AvVR1"
    load_data = load_data_from_folder(folder)
    y3 = np.array(load_data(name)[:,1])



    yd = (y1+y2+y3)/3

    yd = gaussian_filter(yd/10, sigma = 100)
    ax.plot(x, yd, color='C3', marker='', linestyle='-', label=r"<D>(t)")



    ax.set_ylim(bottom=0, top=2)
    ax.set_xlim(left=0, right=max(x))


    ax.set_xlabel(r'$k_{\alpha}$, koefficient k of active force')
    ax.set_ylabel(r'')


    ax.legend(loc=1, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig



def thetamean():


    folder="/run/media/softmatter/Новый том/Fishes/n=1/Rezrotating-/"
    name="theta_mean"
    load_data = load_data_from_folder(folder)


    y1 = np.array(load_data(name)[:,0])
    y2 = np.array(load_data(name)[:,1])
    y3 = np.array(load_data(name)[:,2])



    x1 = np.arange(0,40000,1)[:-1]



    fig, ax = plt.subplots()

    ax.plot(x1, gaussian_filter(y1,30), color='C0', marker='', linestyle='-', label=r"$\theta_{usual}$")
    ax.plot(x1, gaussian_filter((y2),30)-(2*np.pi), color='C2', marker='', linestyle='-', label=r"$\theta_{spec}$")
    ax.plot(x1, y3, color='C1', marker='', linestyle='-', label=r"$\theta_{activity}$")
    #ax.plot(x1, gaussian_filter((y4-y4[1])/2,50), color='C3', marker='', linestyle='dotted')

    #ax.set_ylim(bottom=-2*np.pi, top=0)
    #ax.set_xlim(left=0, right=4)

    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel(r't')


    ax.legend(loc=2, labelcolor='markeredgecolor')



    plt.tight_layout()
    return fig


#theta().show()
#usual_spec(load_data,name).show()
#dis().show()
#vel().show()

#for step in range(1,41):
    #theta(step).savefig('/run/media/softmatter/Новый том/Fishes/n=1/normal/theta_distribution'+str(step)+'.pdf')
#theta().savefig('/run/media/softmatter/Новый том/Fishes/n=1/normal/theta_distribution.pdf')
#usual_spec(load_data,name).savefig('axes_distribution.pdf')
#dis().savefig('Av_D.pdf')
###phase_diagram().savefig('phase_diagram_compare.pdf')
#thetadist().savefig('theta_distribution1.pdf')
###vel().savefig('/run/media/softmatter/Новый том/Fishes/n=1/Av_VA_rotating.pdf')
###vel().savefig('/run/media/softmatter/Новый том/Fishes/n=1/looong.pdf')
thetamean().savefig('/run/media/softmatter/Новый том/Fishes/n=1/Rezrotating-/thetamean.pdf')
