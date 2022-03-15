import numpy as numpy
import math
import os
import statistics


def read_state(f,N,i,cut):
    f.readline()                                                                        # ITEM: TIMESTEP
    f.readline()                                                                        # 0
    f.readline()                                                                        # ITEM: NUMBER OF ATOMS
    A=int(f.readline())
    f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
    f.readline() 
    f.readline() 
    f.readline() 
    f.readline()                                                                        # ITEM: ATOMS id ... 
    arr_x_y=numpy.zeros((A,2))
    arr_dist=numpy.zeros(A*(A-1))
    arr_vx_vy=numpy.zeros((A,2))
    arr_vel_abs=numpy.zeros(A)
    arr_vel_rel=numpy.zeros(A*(A-1))
    for ii in range(A):
        if(i>=cut):
            d=[float(item) for item in f.readline().strip().split(' ')] 
            #print(d)
            arr_x_y[ii]=[d[1],d[2]]
            arr_vx_vy[ii]=[d[3],d[4]]
            #f.readline()
        else:
            d=f.readline()
            #f.readline()
    if(i>=cut):
        for j in range(A):
            arr_vel_abs[j]=math.hypot(arr_vx_vy[j][0],arr_vx_vy[j][1])
            for jj in range(A-1):
                arr_dist[j+jj*A]=math.hypot(arr_x_y[jj+1][0]-arr_x_y[j][0],arr_x_y[jj+1][1]-arr_x_y[j][1])
                arr_vel_rel[j+jj*A]=math.hypot(arr_vx_vy[jj+1][0]-arr_vx_vy[j][0],arr_vx_vy[jj+1][1]-arr_vx_vy[j][1])
    current_values=str(i)+'\t'+str(numpy.mean(arr_dist))+'\t'+str(numpy.mean(arr_vel_abs))+'\t'+str(numpy.mean(arr_vel_rel))+'\n'
    print(i)
    return current_values

for iter in range(1,2):

    filename="/run/media/softmatter/Новый том/Fish_school/Rez/AvD_AvVA_AvVR" + str(iter)

    if(os.path.isfile(filename)):
        os.remove(filename)

    file_name_ = 'fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_50000000n_dump_100spec_ryb_'+str(iter)+'k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-08cut_10000000extra_fishes_0beta_0.000000square_0_py'

    format_file = '.lammpstrj'

    print('/run/media/softmatter/Новый том/Fish_school/Rez/' + file_name_ + format_file)

    f=open('/run/media/softmatter/Новый том/Fish_school/Rez/' + file_name_ + format_file, "r")


    N=400000-1
    cut=0
    for i in range(N):
        st = read_state(f,N,i,cut)
        if(i>=cut):
            fout = open(filename, "a")
            fout.write(st)
            fout.close()


        
        
