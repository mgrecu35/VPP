import numpy as np

from  netCDF4 import Dataset
import glob

fs=glob.glob("/media/grecu/ExtraDrive1/subSet/2*201806*HDF5")
fs=glob.glob("johnkV2/nadir/2*201806*HDF5")

from numba import jit
@jit(nopython=False)
def getAvgProf(bzd,pType,stormTop,sfcType,pRateCMB,sfcPrecip,bcL,c1,mProf,\
               nc,s1,s2):
    a=np.nonzero(bcL>=nc)
    a1=np.nonzero(sfcType[a]==0)
   
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
nc=82

@jit(nopython=True)
def gridbsfc(bsfc,lon,lat,countG,bsfcG,dx,dy,lonmin,latmin,nx):
    for i in range(nx):
        i0=int((lon[i]-lonmin)/dx)
        j0=int((lat[i]-latmin)/dy)
        if i0>=0 and j0>=0 and i0<nx and j0<ny:
            countG[i0,j0]+=1
            bsfcG[i0,j0]+=bsfc[i]
                
dx=2.5
dy=2.5
ny=int(130/dy)
nx=int(360/dx)
lonmin=-180
latmin=-65
countG=np.zeros((nx,ny),float)
bsfcG=np.zeros((nx,ny),float)
c1=np.zeros((23,3),int)
mProf=np.zeros((23,(nc+1),3),float)
fs=sorted(fs)
s1=0.0
s2=0.0

localZenith=np.array([18.207573  , 17.449835  , 16.68716   , 15.929145  ,
                      15.173092  , 14.418981  , 13.661457  , 12.902104  ,
                      12.141968  , 11.388494  , 10.627809  ,  9.871619  ,
                      9.117242  ,  8.360936  ,  7.604285  ,  6.849935  ,
                      6.095216  ,  5.340118  ,  4.5851707 ,  3.8293185 ,
                      3.0731256 ,  2.321466  ,  1.5661428 ,  0.8165892 ,
                      0.11821083,  0.70589805,  1.4615734 ,  2.210915  ,
                      2.9672987 ,  3.722397  ,  4.4760995 ,  5.2310266 ,
                      5.9850426 ,  6.737083  ,  7.496731  ,  8.250167  ,
                      9.009109  ,  9.76346   , 10.521223  , 11.273891  ,
                      12.028936  , 12.789576  , 13.5483675 , 14.300527  ,
                      15.0620775 , 15.818633  , 16.57288   , 17.33766   ,
                      18.098036  ])
iCount=0
clutElevL=[]
chist=np.array([0.32257848, 0.52421935, 0.65297341, 0.74334077, 0.80896207,
                0.8571751 , 0.89060766, 0.91332445, 0.92933464, 0.94047089,
                0.94821858, 0.95356588, 0.9580969 , 0.96262175, 0.9671272 ,
                0.97182571, 0.97716156, 0.98398188, 0.99110105, 0.99593795,
                0.99838682, 0.99942877, 0.99984133, 1.        ])

clutDist=np.array([0.00298139, 0.08212662, 0.47179976, 0.75722084,\
                   0.9367083 , 0.9731587 , 0.98817819, 0.99493455,\
                   0.99784073, 0.99911876, \
                   0.99967064, 0.99992479, 1.        ])
sfcDist=np.array([0.58646311, 0.72222528, 0.80667289, 0.86553844, \
                  0.90333541,
                  0.92778961, 0.94419531, 0.95588936, 0.96403488, \
                  0.96990343,
                  0.97370876, 0.97620051, 0.97858642, 0.98117985, 0.98366693,
                  0.98625621, 0.98923303, 0.99278207, 0.99648364, 0.99899407,
                  1.        ])

@jit(nopython=True)
def bisect(chist,r):
    n1=0
    n2=chist.shape[0]-1
    if r<chist[0]:
        return 0
    if r>chist[n2-1]:
        return n2
    nmid=int((n1+n2)/2)
    #print(nmid)
    it=0
    while not (r>=chist[nmid-1] and r<chist[nmid]) and it<7:
        it+=1
        #print(chist[nmid-1],r,chist[nmid],nmid,n1,n2)
        if r>chist[nmid-1]:
            n1=nmid
        else:
            n2=nmid
        nmid=int((n1+n2)/2)
    return nmid


s1L=[]
s2L=[]
dataL=[]
sfcElevL=[]
bzdL=[]
relClutL=[]
bzdL=[]
zmLs=[]
bcfL=[]
pTypeL=[]
btopL=[]
for f in fs[:]:
    if 'subset' in f:
        continue
    fh=Dataset(f)
    bcL=fh['bcf'][:]
    pType=fh['pType'][:]
    bzd=fh['bzd'][:]
    bcf=fh['bcf'][:]
    stormTop=fh['btop'][:]
    sfcType=fh['sfcType'][:]
    bsfc=fh['bsfc'][:]
    lon=fh['lon'][:]
    lat=fh['lat'][:]
    ray=fh['ray'][:]
    zm=fh['precip1D'][:]
    zm[zm<0]=0
    for i,bt in enumerate(stormTop):
        zm[i,:bt]=0
    nx1,=lon.shape
    print(f)
    
    a=np.nonzero(sfcType!=0)
    b=np.nonzero(bcf[a]>83)
    zm[:,85:]=np.nan
    zmLs.extend(zm[a][b])
    bzdL.extend(bzd[a][b])
    bcfL.extend(bcf[a][b])
    pTypeL.extend(pType[a][b])
    btopL.extend(stormTop[a][b])
        
    #print(bzd[a][b].mean())
    #bzdL.extend(bzd[a])

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

bzdL=np.array(bzdL)
bcfL=np.array(bcfL)
zmLs=np.array(zmLs)
pTypeL=np.array(pTypeL)
datL=[]
zmL1=[]
cL1=[]

for i in range(68,88):
    a=np.nonzero(bzdL==i)
    b=np.nonzero(bcfL[a]>=82)
    c=np.nonzero(pTypeL[a][b]==1)
    zmL1.append(10*(0.1*zmLs[a][b][c]).mean(axis=0))
    cL1.append(len(b[0]))
   

zmL1=np.array(zmL1)
fig=plt.figure(figsize=(8,4))
plt.subplot(111)
f=plt.pcolormesh(range(68,88),range(0,88),\
                 zmL1.T,cmap='jet',vmin=0,vmax=4)
plt.title('Minimal clutter')
plt.ylim(88,50)

plt.figure(figsize=(10,8))
plt.suptitle("Stratiform Land")
plt.subplot(121)
for i in range(0,20):
    plt.plot(zmL1[i,:],range(0,88),'-')
    plt.ylim(84,60)
    plt.xlim(0,4)
plt.xlabel('mm/h')
plt.title('Conditional mean profiles')
plt.ylabel('Range bin')

plt.subplot(122)
for i in range(0,20):
    plt.plot(zmL1[i,:]/zmL1[i,68+i],-(68+i)+np.arange(0,88),'-')

plt.title('Normalized conditional means')

plt.ylim(15,-15)
plt.xlim(0,2)
plt.ylabel('Relative range bin')
plt.xlabel('')
plt.tight_layout()
plt.savefig("avg_st_Profiles_Land.png")

import pickle
pickle.dump(zmL1,open("zmL1_Land_st.pklz","wb"))
stop
#plt.figure()
#plt.plot(prof1/countg,29-np.arange(30))

#stop    
countg=np.zeros((33),float)
prof1=np.zeros((33),float)
n=zmLs.shape[0]
for i in range(n):
    if pTypeL[i]==2 and bzdL[i]>=68 and bcfL[i]>=82 and bzdL[i]<75 and btopL[i]<bzdL[i]-4:
        if zmLs[i,bzdL[i]]>0:
            i1=max(0,bzdL[i]-20)
            i2=min(bzdL[i]+13,bcfL[i])
            #zm1+=zmLs[i][0:80].data
            #c+=1
            for k in range(i1,i2):
                i0=k-bzdL[i]+20
                countg[i0]+=1
                prof1[i0]+=zmLs[i,k]
        #stop


plt.figure()
plt.plot(prof1/countg,33-np.arange(33))
