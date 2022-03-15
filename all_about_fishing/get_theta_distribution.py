import numpy as numpy
import math
import os
import statistics

###filename="/run/media/softmatter/Новый том/Fishes/n=1/normal/Rez1/theta_distribution1"

###def read_state(f):
    ###N=2500-1
    ###cut=1500-1
    ###arrthetadist=numpy.zeros(100)
    ####alpha=1
    ###for i in range(N):
        ####alpha+=math.pi/500000
        ###f.readline()                                                                        # ITEM: TIMESTEP
        ###f.readline()                                                                        # 0
        ###f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        ###A=int(f.readline())
        ###f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        ###f.readline()
        ###f.readline()
        ###f.readline()
        ###f.readline()                                                                        # ITEM: ATOMS id ...
        ###for k in range(A):
            ###if(i>=cut):
                ###d=[float(item) for item in f.readline().strip().split(' ')[0:6]]
                ####d[5]+=alpha
                ###while(d[5]<0):
                    ###d[5]+=2*math.pi
                ###while(d[5]>2*math.pi):
                    ###d[5]-=2*math.pi
                ###for q in range(100):
                    ###if(d[5]>=q*2*math.pi/100 and d[5]<(q+1)*2*math.pi/100):
                        ###arrthetadist[q]+=1
                        ####print(arrthetadist)
            ###else:
                ###d=f.readline()
    ####print(arrthetadist)
    ###return arrthetadist

###def dist(f,filename):
    ###N=40000 - 1
    ###cut= 1 - 1
    ###arrthetadist=numpy.zeros(360)

    ###fout = open(filename, "a")
    ####alpha=1
    ###for i in range(N):
        ####alpha+=math.pi/500000
        ###print(i)
        ###f.readline()                                                                        # ITEM: TIMESTEP
        ###f.readline()                                                                        # 0
        ###f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        ###A=int(f.readline())
        ###f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        ###f.readline()
        ###f.readline()
        ###f.readline()
        ###f.readline()
        ###for k in range(A-1):
            ###if(i>=cut):
                ###d=[float(item) for item in f.readline().strip().split(' ')[0:6]]
                ####d[5]+=alpha
                ###while(d[5]<0):
                    ###d[5]+=2*math.pi
                ###while(d[5]>2*math.pi):
                    ###d[5]-=2*math.pi
                ###for q in range(360):
                    ###if(d[5]>=q*2*math.pi/360 and d[5]<(q+1)*2*math.pi/360):
                        ###arrthetadist[q]+=1
                        ####print(arrthetadist)
            ###else:
                ###d=f.readline()
    ####print(arrthetadist)
        #####fout.write(str(i)+'\t'+str(statistics.stdev(arrthetadist))+'\n')

    ###fout.write(str((arrthetadist/arrthetadist.max()).tolist())[1:-1])
    ###return 0



def meantheta(f,filename):
    N=40000 - 1
    cut= 1 - 1


    fout = open(filename, "a")
    #alpha=1
    for i in range(N):
        #alpha+=math.pi/500000
        print(i)
        arrthetamean=0
        f.readline()                                                                        # ITEM: TIMESTEP
        f.readline()                                                                        # 0
        f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        A=int(f.readline())
        f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        if(i>=cut):
            spec=[float(item) for item in f.readline().strip().split(' ')][5]#first spec fish
            #spec%=(2*numpy.pi)
        for k in range(A-1):
            if(i>=cut):
                d=[float(item) for item in f.readline().strip().split(' ')]
                #d[5]+=alpha
                #d[5]%=(2*numpy.pi)
                arrthetamean+=d[5]
            else:
                d=f.readline()
        fout.write(str(arrthetamean/127)+' '+str(spec)+' '+str(-(((i+10000)*numpy.pi/5000))%(2*numpy.pi))+'\n')
        #fout.write(str(arrthetamean/127)+' '+str(spec)+' '+str(-(((i+10000)*numpy.pi/5000)))+'\n')


    return 0




filename="/run/media/softmatter/Новый том/Fishes/n=1/Rezrotating-/theta_mean1"

if(os.path.isfile(filename)):
    os.remove(filename)

file_name_ = 'fishesn_128scale_20.000000If_0.010000Ip_0.500000In_0.050000dt_0.010000rc_hydro_scale_5.000000nmax_5000000n_dump_100spec_ryb_1k_alpha_0.300000alpha_0.000000v_alpha_-500000.000000v_k_0cut_1000000extra_fishes_0beta_0.000000square_0_py'

format_file = '.lammpstrj'


f=open('/run/media/softmatter/Новый том/Fishes/n=1/Rezrotating-/' + file_name_ + format_file, "r")

st=meantheta(f,filename)

f.close()
