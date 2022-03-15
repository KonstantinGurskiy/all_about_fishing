#arrIp=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]
#arrIn=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]

#arrIp=[0]
#arrIn=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]

arrn1=[11,12,13,14,16,17,18,19]
arrn=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
arrk=[0.0001,0.0005,0.001,0.003,0.007,0.01,0.03,0.07,0.1]
n1=[128,256]
n=[384,512]
#arrk=[0.0005,0.001,0.003,0.007,0.01,0.03,0.07,0.1]
#arrn=[20]
#arrk=[1]

for k in arrk:
    for j in arrn:
        for z in n:
            f=open('/run/media/softmatter/Новый том/Fishes/in.txt','r')
            lines=f.readlines()
            lines[0]=str(z)+'\n'
            lines[12]=str(int(z/128)*j)+'\n'
            lines[13]=str(k)+'\n'
            f.close()
            save_changes = open('/run/media/softmatter/Новый том/Fishes/in.txt', 'w')
            save_changes.writelines(lines)
            save_changes.close()
            import os
            os.system('./a.out')

for k in arrk:
    for j in arrn1:
        for z in n1:
            f=open('/run/media/softmatter/Новый том/Fishes/in.txt','r')
            lines=f.readlines()
            lines[0]=str(z)+'\n'
            lines[12]=str(int(z/128)*j)+'\n'
            lines[13]=str(k)+'\n'
            f.close()
            save_changes = open('/run/media/softmatter/Новый том/Fishes/in.txt', 'w')
            save_changes.writelines(lines)
            save_changes.close()
            import os
            os.system('./a.out')
