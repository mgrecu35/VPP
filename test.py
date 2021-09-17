import numpy as np

from numba import jit
from netCDF4 import Dataset
fh=Dataset("trainingDset.nc")
pRate=fh["pRate"][:,:]
bzdL=fh["bzdL"][:]
bcL=fh["bcL"][:]
nx=pRate.shape[0]
tData1=np.zeros((nx,8),float)
@jit(nopython=True)
def test_random(pRate,bzdL,bcL,icv,tData1):
    s=0
    nx=pRate.shape[0]
    ic=0
    for i in range(nx):
        ir=int(13*np.random.rand())
        if pRate[i,68-ir]>0.0:
            tData1[ic,0:6]=pRate[i,68-ir-5:68-ir+1]
            tData1[ic,6]=168-ir
            tData1[ic,7]=bzdL[i]
            ic+=1
    icv[0]=ic
    
import time
t1=time.time()
icv=np.zeros((1),int)
test_random(pRate,bzdL,bcL,icv,tData1)
print(icv)
print(time.time()-t1)
