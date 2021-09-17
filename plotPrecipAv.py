import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator


import numpy as np

zkuL=pickle.load(open('averageR_Aug.pklz','rb'))

plt.figure(figsize=(8,8))
plt.suptitle('August 2018')

for i in range(4):
    plt.subplot(2,2,i+1)
    for k in range(2):
        plt.plot(zkuL[i*2+k][:],np.arange(34)+50)
        plt.ylim(84,50)
    plt.title('%3i<=binZeroDeg<%3i'%(154+i*4,154+i*4+4),fontsize=12)
    if i==0 or i==2:
        plt.ylabel('Range bin')
    if i==2 or i==3:
        plt.xlabel('Precipitation rate(mm/h)')

plt.savefig('precipAvg_Aug.png')



zkuL=pickle.load(open('averageR_Jan.pklz','rb'))

plt.figure(figsize=(8,8))
plt.suptitle('January 2018')

for i in range(4):
    plt.subplot(2,2,i+1)
    for k in range(2):
        plt.plot(zkuL[i*2+k][:],np.arange(34)+50)
        plt.ylim(84,50)
    plt.title('%3i<=binZeroDeg<%3i'%(154+i*4,154+i*4+4),fontsize=12)
    if i==0 or i==2:
        plt.ylabel('Range bin')
    if i==2 or i==3:
        plt.xlabel('Precipitation rate(mm/h)')

plt.savefig('precipAvg_Jan.png')

