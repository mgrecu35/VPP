import os
f=open('files','r').readlines()
for f1 in f:
    cmd='git rm -f %s'%f1[:-1]
    print(cmd)
    os.system(cmd)
