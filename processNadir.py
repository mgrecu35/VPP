import numpy as np

from  netCDF4 import Dataset
import glob

fs=glob.glob("johnkV2/nadir/2*HDF5")

from numba import jit
@jit(nopython=False)
def getAvgProf(bzd,pType,stormTop,sfcType,pRateCMB,sfcPrecip,bcL,c1,mProf,\
               nc,s1,s2):
    a=np.nonzero(bcL>=nc)
    a1=np.nonzero(sfcType[a]!=0)
   
    for i in a[0][a1]:
        ir1=int(np.random.random()*7)
        pRateCMB[i,0:stormTop[i]-1]=0
        c=np.nonzero(pRateCMB[i,stormTop[i]:]<-99)
        if len(c[0])>0:
            pRateCMB[i,stormTop[i]+c[0]]=sfcPrecip[i]

        if pRateCMB[i,stormTop[i]:].min()<-99 and pType[i]<=2:
            print(pRateCMB[i,stormTop[i]:])
            print(bcL[i],pType[i],i)
            stop
        ir=min(76+ir1,nc)
        s1+=pRateCMB[i,ir]
        s2+=pRateCMB[i,nc]
        #print(ir,ir1,nc)
    #stop
    for ibzd in range(65,88):
        b=np.nonzero(bzd[a][a1]==ibzd)
        c=np.nonzero(pType[a][a1][b]==1)
        c1[ibzd-65,0]+=len(c[0])     
        if len(c[0])>0:
            mProf[ibzd-65,:,0]+=pRateCMB[a[0][a1][b][c],0:(nc+1)].sum(axis=0)
        c=np.nonzero(pType[a][a1][b]==2)
        d=np.nonzero(stormTop[a][a1][b][c]<ibzd-4)
        if len(d[0])>0:
            mProf[ibzd-65,:,1]+=pRateCMB[a[0][a1][b][c][d],0:(nc+1)].sum(axis=0)
            c1[ibzd-65,1]+=len(d[0]) 
        d=np.nonzero(stormTop[a][a1][b][c]>=ibzd-4)
        if len(d[0])>0:
            mProf[ibzd-65,:,2]+=pRateCMB[a[0][a1][b][c][d],0:(nc+1)].sum(axis=0)
            c1[ibzd-65,2]+=len(d[0])
    return s1,s2
nc=80

c1=np.zeros((23,3),int)
mProf=np.zeros((23,(nc+1),3),float)
fs=sorted(fs)
s1=0.0
s2=0.0
for f in fs:
    fh=Dataset(f)
    bcL=fh['bcf'][:]
    pType=fh['pType'][:]
    bzd=fh['bzd'][:]
    pRateCMB=fh['precip1D'][:]
    sfcPrecip=fh['sfcPrecip'][:]
    #pRateCMB[pRateCMB<0]=0
    stormTop=fh['btop'][:]
    sfcType=fh['sfcType'][:]
    s1,s2=getAvgProf(bzd,pType,stormTop,sfcType,pRateCMB,sfcPrecip,\
               bcL,c1,mProf,nc,s1,s2)
    print(f,s1,s2)

    #stop

for i in range(23):
    mProf[i,:,0]/=mProf[i,-1,0]
    mProf[i,:,1]/=mProf[i,-1,1]
    mProf[i,:,2]/=mProf[i,-1,2]

import matplotlib.pyplot as plt
st=['stratiform','convective','shallow']
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

plt.figure(figsize=(8,12))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.pcolormesh(65+np.arange(23),np.arange((nc+2)),np.array(mProf[:,:,i]).T,cmap='gist_earth',vmax=1.1);
    plt.ylim(nc+1,40)   
    plt.xlim(65,87)
    if i==2:
        plt.xlabel('Zero degree bin')
    plt.ylabel('Range bin')
    plt.colorbar()
    plt.title("Land %s"%st[i])
    #plt.colorbar()
plt.tight_layout()
plt.savefig('correctionTableDPR_Land_Dec2018.png')
