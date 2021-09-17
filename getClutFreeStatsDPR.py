from os import walk
#from readGPROFSub import *
mypath='/itedata/ITE057/201509'
mypath='/gpmdata/'
mypath='/itedata/ITE749/'
f1 = []
f2 = []
f3 = []
f4 = []
import glob
#print glob.glob("/home/adam/*.txt")
import datetime
s=datetime.datetime(2018,8,1)
#import h5py as h5
import numpy as np

from  netCDF4 import Dataset
ifig=0
levs=levels=[0.1,0.2,\
                 0.4,0.8,1.6,3.2,6.4,12.8,25.6,50.16,100.24,200.48]

dbL=[]
#from pyresample import bilinear, geometry, kd_tree
clutJ=[78, 78, 78, 78, 79, 79, 80, 80, 80, 80, 81, 81, 81, 82, 81, 82, 82,
       82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 83, 82, 82, 82, 82, 82, 81,
       81, 81, 81, 81, 80, 80, 80, 80, 79, 79, 79, 78, 78, 77, 77]
import matplotlib.pyplot as plt
import xarray as xr
sfcPrecipL=[[] for j in range(16)]
bzL=[[] for j in range(16)]
clutJL=[[] for i in range(49)]
import time
pRateL=[]
zmL=[]
bzdL=[]
bcL=[]
pTypeL=[]
for iday in range(1):
    c=s+datetime.timedelta(days=iday)
    dir3=mypath+'%4.4i/%2.2i/%2.2i/radar/2A.GPM.DPR.V9*'%(c.year,c.month,c.day)
        #l2=glob.glob(dir2)
    l3=glob.glob(dir3)
    l3=sorted(l3)
    t1=time.time()
    print(iday)
    for l00 in l3:
        cAlg=Dataset(l00,'r')
        lat1=cAlg['FS']['Latitude'][:,25]
        lon1=cAlg['FS']['Longitude'][:,25]
        a1=np.nonzero((lat1[:]+45)*(lat1[:]+65)<0)
        h0=cAlg['FS/VER/heightZeroDeg'][a1[0],:]
        bz=cAlg['FS/VER/binZeroDeg'][a1[0],:]
        pType=(cAlg['FS/CSF/typePrecip'][a1[0],:]/1e7).astype(int)
        sType=(cAlg['FS/PRE/landSurfaceType'][a1[0],:])
        cFree=cAlg['FS/PRE/binClutterFreeBottom'][a1[0],:]
        zM=cAlg['FS/PRE/zFactorMeasured'][a1[0],:,100:,:]
        bbPeak=cAlg['FS/CSF/binBBPeak'][a1[0],:]
        bbTop=cAlg['FS/CSF/binBBTop'][a1[0],:]
        pathAtten=cAlg['FS/SRT/pathAtten'][a1[0],:,0]
        relFactor=cAlg['FS/SRT/reliabFactor'][a1[0],:]
        #
        l01=l00.replace("2B.GPM.DPRGMI.CORRA2018","2A.GPM.Ku.V8-20180723")
        a2=np.nonzero((3-pType)*pType>0)
        precipRate=cAlg['FS/SLV/precipRate'][a1[0],:,100:]
        a3=np.nonzero(sType[a2]==0)
        for i,j in zip(a2[0][a3],a2[1][a3]):
            clutJL[j].append(cFree[i,j])
            if cFree[i,j]>=168 and abs(j-24)<12:
                pRateL.append(precipRate[i,j,:76])
                zmL.append(zM[i,j,:76,:])
                bzdL.append(bz[i,j])
                bcL.append(cFree[i,j])
                pTypeL.append([pType[i,j],bbPeak[i,j],bbTop[i,j],\
                                   pathAtten[i,j],\
                                   relFactor[i,j]])
        #continue
        for i,j in zip(a2[0][a3],a2[1][a3]):
            if abs(j-24)<3:
                for k in range(16) :
                    if cFree[i,j]>=170-k and bz[i,j]<=170:
                        if precipRate[i,j,70-k]==precipRate[i,j,70-k]\
                                and  precipRate[i,j-12,:70-k].max()>0.01:
                            sfcPrecipL[k].append(precipRate[i,j,70-k])
                        #bzL[k].append(bz[i,j])
    print(time.time()-t1)
    #stop
import pickle
import pickle
d={"sfcPrecip":sfcPrecipL, "clutJ":clutJL}
pickle.dump(d,open("bcStatsDPR_Aug2018.pklz","wb"))
d2={"pRate":pRateL,"bzdL":bzdL,"bcL":bcL, "zm":zmL,"TBBPR":pTypeL}
pickle.dump(d2,open("precipProfilesDPR_Aug2018.pklz","wb"))

