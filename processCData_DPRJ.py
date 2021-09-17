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

import pickle
#d={"sfcPrecip":sfcPrecipL,"bzL":bzL}
d=pickle.load(open("bcStatsDPR_Jan2018_NH.pklz","rb"))
#d2={"pRate":pRateL,"bzdL":bzdL,"bcL":bcL}
d2=pickle.load(open("precipProfilesDPR_Jan2018_NH.pklz","rb"))

pRate=d2["pRate"]
zm=d2["zm"]
bzdL=d2["bzdL"]
pRate=np.array(pRate)
bzdL=np.array(bzdL)

pRatemL=[]
countsL=[]
zm=np.array(zm)
zkuL=[]
zkaL=[]
zm[zm<0]=0
for i in range(154,171):
    a=np.nonzero(bzdL==i)
    pRatem=pRate[a[0],:].mean(axis=0)
    zkum=10*np.log10(10.**(0.1*zm[a[0],:,0]).mean(axis=0))
    zkam=10*np.log10(10.**(0.1*zm[a[0],:,1]).mean(axis=0))
    pRatemL.append(pRatem)
    zkuL.append(zkum)
    zkaL.append(zkam)
    countsL.append(len(a[0]))

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator

countsL=np.array(countsL)
fig=plt.figure();
plt.bar(range(154,171),countsL/countsL.sum()*100);
plt.xlabel('Zero Degree Bin');
plt.ylabel('Percentage of profiles (%)');
ax=fig.gca();
ax.xaxis.set_major_locator(MaxNLocator(integer=True));
plt.tight_layout();
plt.savefig('bzdHist_Jan_NH.png');


pickle.dump([zkuL,zkaL],open('averageZ_Jan_NH.pklz','wb'))

