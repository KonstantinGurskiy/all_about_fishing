import numpy as numpy
import math
import os
import statistics
def read_state(f,nspec):
    N=40000-1
    cut=40000-1000
    arraxesdist=numpy.zeros(1000)
    arrdist=numpy.zeros(1000)
    for i in range(N):
        f.readline()                                                                        # ITEM: TIMESTEP
        f.readline()                                                                        # 0
        f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        A=int(f.readline())
        f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        f.readline() 
        f.readline() 
        f.readline() 
        f.readline()                                                                        # ITEM: ATOMS id ... 
        for k in range(A):
            if(i>=cut and k<nspec):
                d=[float(item) for item in f.readline().strip().split(' ')[0:6]] 
                for q in range(1000):
                    if(d[1]>=(q-500)/10 and d[2]<(q+1-500)/10):
                        arraxesdist[q]+=1
                        #print(arraxesdist)
            else:
                if(k>nspec and i>=cut):
                    d=[float(item) for item in f.readline().strip().split(' ')[0:6]] 
                    for q in range(1000):
                        if(d[1]>=(q-500)/10 and d[2]<(q+1-500)/10):
                            arrdist[q]+=1
                else:
                    d=f.readline()
    #print(arraxesdist)
    return [arraxesdist,arrdist]


if(os.path.isfile("axes_distribution")):
    os.remove("axes_distribution")
if(os.path.isfile("usual_distribution")):
    os.remove("usual_distribution")


for nspec in [6]:
    #for k_alpha in [0.0001,0.0005,0.001,0.003,0.007,0.01,0.03,0.07,0.1]:
    for k_alpha in [0]:
        #file_name_ = 'log_fishesn_128scale_20.000000If_0.010000Ip_' + format(Ip, '.6f') + 'In_'+ format(In, '.6f')+'dt_0.010000rc_hydro_scale_10.000000nmax_1000000n_dump_100spec_ryb_10vd_1.000000dej_10.000000'

        file_name_ = 'fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_' + format(nspec,'d') + 'k_alpha_'+ format(k_alpha, '.6f') +'alpha_0.000000v_alpha_0.000000v_k_1e-08cut_1000000extra_fishes_0beta_0.000000square_0_py'

        format_file = '.lammpstrj'

        print('/run/media/softmatter/Новый том/Fishes/' + file_name_ + format_file)

        f=open('/run/media/softmatter/Новый том/Fishes/' + file_name_ + format_file, "r")

        

        [st,dist] = read_state(f,nspec)
        print(st)
        print(dist)
        
        f = open("axes_distribution", "a")
        fout = open("usual_distribution", "a")
        f.write(str(nspec)+'\t'+str(k_alpha)+'\t'+str(statistics.stdev(st))+'\n'+str(' '.join(map(str, st)))+'\n')
        fout.write(str(nspec)+'\t'+str(k_alpha)+'\t'+str(statistics.stdev(dist))+'\n'+str(' '.join(map(str, dist)))+'\n')
        f.close()
        fout.close()
        
        
 
