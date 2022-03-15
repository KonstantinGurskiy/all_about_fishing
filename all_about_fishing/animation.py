
import math
import scipy
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import interpolate
import matplotlib
import sys
import os
import subprocess

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.style.use('/run/media/softmatter/Новый том1/Fish/Figure_style_with_matplotlib_mplstyle-main/figStyle.mplstyle')

def get_row(fb, ind_start, ind_stop, ind_row):
    row = []
    for ind in range(ind_start, ind_stop):
        row.append(float(fb[ind].replace('\n', '').split()[ind_row]))
    
    return row

def load_lammpstrj(file_name):
    
    with open(file_name) as file:
        fb = file.readlines()
    num_parts = int(fb[3])
    num_frames = len(fb)//(num_parts + 9)
    parts = []
    for i in range(num_frames):
        print(i)
        val = [get_row(fb, i*(num_parts+9) + 9, (i + 1)*(num_parts+9), j) for j in range(4)]
        parts.append(val)
        
    return np.array(parts) # трехмерный массив [кадр, столбец, строка]

def load_log(file_name):
    with open(file_name) as file:
        file.readline()
        fb = file.readlines()
    x_mean = get_row(fb, 0, len(fb) - 1, 2)
    y_mean = get_row(fb, 0, len(fb) - 1, 3)
    
    return (x_mean, y_mean)

data_fish = load_lammpstrj('/run/media/softmatter/Новый том1/Fish/dump/1_100_5_5.0_0.05_0.1_20000_100000_100_0_0.6_0.99_0.0_py.lammpstrj')
x_mean, y_mean = load_log("/run/media/softmatter/Новый том1/Fish/dump/log_1_100_5_5.0_0.05_0.1_20000_100000_100_0_0.6_0.99_0.0.txt")
x_mean = np.array(x_mean)
y_mean = np.array(y_mean)
print(data_fish.shape)
print(x_mean.shape, y_mean.shape)

print(x_mean)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    ax1.clear()
    ax1.plot(x_mean, y_mean, marker="", linestyle='--', color='C7')
    ax1.plot(data_fish[i, 1, :6] + x_mean[i], data_fish[i, 2, :6] + y_mean[i], marker=".", linestyle='', color='C3', markersize=15)
    ax1.plot(data_fish[i, 1, 6:] + x_mean[i], data_fish[i, 2, 6:] + y_mean[i], marker=".", linestyle='', color='C0', markersize=5)


ani = animation.FuncAnimation(fig, animate, frames=range(0, len(x_mean), 10), interval=0)
plt.show()
