import numpy as np

from  netCDF4 import Dataset

clutJ=[78, 78, 78, 78, 79, 79, 80, 80, 80, 80, 81, 81, 81, 82, 81, 82, 82,
       82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 83, 82, 82, 82, 82, 82, 81,
       81, 81, 81, 81, 80, 80, 80, 80, 79, 79, 79, 78, 78, 77, 77]
import matplotlib.pyplot as plt
import xarray as xr
sfcPrecipL=[[] for j in range(7)]
bzL=[[] for j in range(7)]
clutJL=[[] for i in range(49)]
import time
pRateL=[]
bzdL=[]
bcL=[]
for iday in range(31):
    c=s+datetime.timedelta(days=iday)
    dir3=mypath+'%4.4i/%2.2i/%2.2i/radar/2B.*GPM*COR*'%(c.year,c.month,c.day)
        #l2=glob.glob(dir2)
    l3=glob.glob(dir3)
    l3=sorted(l3)
    t1=time.time()
    print(iday)
    for l00 in l3:
        cAlg=Dataset(l00,'r')
        lat1=cAlg['NS']['Latitude'][:,25]
        lon1=cAlg['NS']['Longitude'][:,25]
        a1=np.nonzero((lat1[:]+45)*(lat1[:]+65)<0)
        h0=cAlg['NS/Input/zeroDegAltitude'][a1[0],:]
        bz=cAlg['NS/Input/zeroDegBin'][a1[0],:]
        pType=(cAlg['NS/Input/precipitationType'][a1[0],:]/1e7).astype(int)
        sType=(cAlg['NS/Input/surfaceType'][a1[0],:])
        cFree=cAlg['NS/Input/lowestClutterFreeBin'][a1[0],:]
        #precip=
        l01=l00.replace("2B.GPM.DPRGMI.CORRA2018","2A.GPM.Ku.V8-20180723")
        a2=np.nonzero((3-pType)*pType>0)
        precipRate=cAlg['MS/precipTotRate'][a1[0],:,50:]
        a3=np.nonzero(sType[a2]==0)
        for i,j in zip(a2[0][a3],a2[1][a3]):
            clutJL[j].append(cFree[i,j])
            if cFree[i,j]>=83 and abs(j-24)<12:
                pRateL.append(precipRate[i,j-12,:34])
                bzdL.append(bz[i,j])
                bcL.append(cFree[i,j])
        #continue
        for i,j in zip(a2[0][a3],a2[1][a3]):
            if abs(j-24)<3:
                for k in range(7) :
                    if cFree[i,j]>=83-k and bz[i,j]<=83:
                        if precipRate[i,j-12,33-k]==precipRate[i,j-12,33-k]\
                                and  precipRate[i,j-12,:34-k].max()>0.01:
                            sfcPrecipL[k].append(precipRate[i,j-12,33-k])
                            bzL[k].append(bz[i,j])
    print(time.time()-t1)
    #stop
import pickle
import pickle
d={"sfcPrecip":sfcPrecipL,"bzL":bzL}
pickle.dump(d,open("bcStats.pklz","wb"))
d2={"pRate":pRateL,"bzdL":bzdL,"bcL":bcL}
pickle.dump(d2,open("precipProfiles.pklz","wb"))



