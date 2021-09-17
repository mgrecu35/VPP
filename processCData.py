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
#d=pickle.load(open("bcStats.pklz","rb"))
#d2={"pRate":pRateL,"bzdL":bzdL,"bcL":bcL}
d2=pickle.load(open("precipProfiles_Jan.pklz","rb"))

pRate=d2["pRate"]
bzdL=d2["bzdL"]
pRate=np.array(pRate)
bzdL=np.array(bzdL)

pRatemL=[]
countsL=[]
for i in range(76,84):
    a=np.nonzero(bzdL==i)
    pRatem=pRate[a[0],:].mean(axis=0)
    pRatemL.append(pRatem)
    countsL.append(len(a[0]))

pickle.dump(pRatemL,open('averageR_Jan.pklz','wb'))
