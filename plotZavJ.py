import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator

d=pickle.load(open('bcStatsDPR.pklz','rb'))
d2=pickle.load(open('bcStats.pklz','rb'))
sfcPrecip=d['sfcPrecip']
sfcPrecipCMB=d2['sfcPrecip']
clutJ=d['clutJ']
plt.figure()
sfcPrecipL=[]
sfcPrecipCMBL=[]
import numpy as np

for i in range(12):
    sfcPrecipL.append(np.mean(sfcPrecip[i]))

for i in range(6):
    sfcPrecipCMBL.append(np.mean(sfcPrecipCMB[i]))

#plt.plot(169-np.arange(6)*2,sfcPrecipCMBL)
#plt.plot(170-np.arange(12),sfcPrecipL,'--',color='blue')
#plt.ylabel('Precipitation rate (mm/h)')
#plt.xlabel('Range bin')
#plt.xlim(170,159)
#plt.xlim(170,159)
#plt.legend(['CMB','DPR'])
#plt.tight_layout()
#plt.savefig('sfcPrecipRange.png')


[zkuL,zkaL]=pickle.load(open('averageZ_Jan_NH.pklz','rb'))

plt.figure(figsize=(8,8))
plt.suptitle('Ku-band')

for i in range(4):
    plt.subplot(2,2,i+1)
    for k in range(4):
        plt.plot(zkuL[i*4+k][:],np.arange(76)+100)
        plt.ylim(168,100)
        plt.xlim(0,30)
    plt.title('%3i<=binZeroDeg<%3i'%(154+i*4,154+i*4+4),fontsize=12)
    if i==0 or i==2:
        plt.ylabel('Range bin')
    if i==2 or i==3:
        plt.xlabel('dBZ')

plt.savefig('kuBandZavg_Jan_NH.png')

plt.figure(figsize=(8,8))
plt.suptitle('Ka-band')
for i in range(4):
    plt.subplot(2,2,i+1)
    for k in range(4):
        plt.plot(zkaL[i*4+k][:],np.arange(76)+100)
        plt.ylim(168,100)
        plt.xlim(0,25)
    plt.title('%3i<=binZeroDeg<%3i'%(154+i*4,154+i*4+4),fontsize=12)
    if i==0 or i==2:
        plt.ylabel('Range bin')
    if i==2 or i==3:
        plt.xlabel('dBZ')


plt.savefig('kaBandZavg_Jan_NH.png')

stop
clutFAD=np.zeros((22,49),float)
for i in range(49):
    for b in clutJ[i]:
        if b-151>=0 and b-151<=21:
            clutFAD[b-151,i]+=1

fig=plt.figure()
plt.pcolormesh(np.arange(49),151+np.arange(22),clutFAD,cmap='jet',\
               norm=matplotlib.colors.LogNorm())
plt.title('Lowest clutter-free bin distribution',fontsize=14)
ax=fig.gca();
ax.yaxis.set_major_locator(MaxNLocator(integer=True));
plt.ylim(171,151)
plt.ylabel('Range bin')
plt.xlabel('Ray')
cbar=plt.colorbar()
cbar.ax.set_title('Counts',fontsize=13)
plt.savefig('clutFreeBinDist.png')
