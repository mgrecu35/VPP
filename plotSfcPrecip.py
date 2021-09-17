import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from matplotlib.ticker import MaxNLocator


d2=pickle.load(open('bcStats_Aug.pklz','rb'))
sfcPrecipCMB_a=d2['sfcPrecip']
sfcPrecipCMBJ_a=d2['sfcPrecipJ']
counts_a=d2['countP']
clutJa=d2['clutJ']
d2=pickle.load(open('bcStats_Jan.pklz','rb'))
sfcPrecipCMB_j=d2['sfcPrecip']
sfcPrecipCMBJ_j=d2['sfcPrecipJ']
clutJj=d2['clutJ']
counts_j=d2['countP']

plt.figure()

sfcPrecipCMBL_a=[]
sfcPrecipCMBL_j=[]
import numpy as np
sfcPrecipr_j=[]
sfcPrecipr_a=[]
for i in range(49):
    sfcPrecipCMBL_a.append(np.mean(sfcPrecipCMBJ_a[i]))
    sfcPrecipCMBL_j.append(np.mean(sfcPrecipCMBJ_j[i]))

for i in range(7):
    sfcPrecipr_a.append(np.mean(sfcPrecipCMB_a[i]))
    sfcPrecipr_j.append(np.mean(sfcPrecipCMB_j[i]))

plt.plot(np.arange(49) ,sfcPrecipCMBL_j)
plt.plot(np.arange(49) ,sfcPrecipCMBL_a)
plt.ylabel('Precipitation rate (mm/h)')
plt.xlabel('Ray')
plt.title('Conditional average precipitation')
plt.xlim(0,48)
#lt.xlim(170,159)
#lt.xlim(170,159)
plt.legend(['January','August'])
plt.tight_layout()
plt.savefig('sfcPrecip_Ray.png')

plt.figure()
plt.plot(np.arange(49),counts_j[:,1]/counts_j[:,0]*100)
plt.plot(np.arange(49),counts_a[:,1]/counts_a[:,0]*100)
plt.legend(['January','August'])
plt.xlabel('Ray')
plt.ylabel('Precipitation fraction (%)')
plt.savefig('percentage_Ray.png')
plt.xlim(0,48)
plt.figure()
plt.plot(84*2+1-np.arange(7)*2 ,sfcPrecipr_j)
plt.plot(84*2+1-np.arange(7)*2 ,sfcPrecipr_a)
plt.ylabel('Precipitation rate (mm/h)')
plt.xlabel('Ray')
plt.xlim(169,159)
#lt.xlim(170,159)
plt.legend(['January','August'])
plt.tight_layout()
plt.savefig('sfcPrecip_Bin.png')


