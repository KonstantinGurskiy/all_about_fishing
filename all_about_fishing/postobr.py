import numpy as numpy
import math
import os
import csv

def read_state(f):
    N=40000-10
    cut=0
    ts=numpy.empty(N-cut)
    Px=numpy.empty(N-cut)
    Py=numpy.empty(N-cut)
    erx=numpy.empty(N-cut)
    ery=numpy.empty(N-cut)
    Vx=numpy.empty(N-cut)
    Vy=numpy.empty(N-cut)
    m=numpy.empty(N-cut)
    for i in range(N):
        d=[float(item) for item in f.readline().strip().split('\t')]
        if(i>=cut):
            ts[i-cut]=(d[0]-1000000)*0.0000001
            Px[i-cut]=d[4]
            Py[i-cut]=d[5]
            erx[i-cut]=d[6]
            ery[i-cut]=d[7]
            Vx[i-cut]=d[8]
            Vy[i-cut]=d[9]
            m[i-cut]=d[10]
    #P = math.sqrt(numpy.mean(Px*Px)+numpy.mean(Py*Py))
    #M = numpy.mean(numpy.absolute(m))/(math.sqrt(numpy.mean(erx*erx)+numpy.mean(ery*ery))*math.sqrt(numpy.mean(Vx*Vx)+numpy.mean(Vy*Vy)))
    #P = math.sqrt(numpy.mean(Px*Px+Py*Py))
    #M = numpy.mean(numpy.absolute(m))/(math.sqrt(numpy.mean(erx*erx+ery*ery))*math.sqrt(numpy.mean(Vx*Vx+Vy*Vy)))
            P = numpy.sqrt(Px*Px+Py*Py)
    #print(numpy.absolute(m)/(numpy.sqrt(erx*erx+ery*ery)*numpy.sqrt(Vx*Vx+Vy*Vy)))
            M = numpy.absolute(m)/(numpy.sqrt(erx*erx+ery*ery)*numpy.sqrt(Vx*Vx+Vy*Vy))
    return numpy.array([ts,P/P.max(),M/M.max()]).transpose()


if(os.path.isfile("/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/data.npy")):
    os.remove("/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/data.npy")



file_name_ = 'log_fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5020000n_dump_100spec_ryb_1k_alpha_0.000000alpha_0.000000v_alpha_0.000000v_k_1e-07cut_1020000extra_fishes_0beta_0.000000square_0'

format_file = '.txt'




f=open('/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/' + file_name_ + format_file, "r")


st = read_state(f)
        
#print(st)
with open("/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez3/data.npy", 'wb') as f:
    numpy.save(f,st)

f.close()
        
        
