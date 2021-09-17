from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
import numpy as np
fh=Dataset("june2018_training_set.nc")
Xy=fh["trainDataSet"][:]
#Xy[:,-1]=np.log(Xy[:,-1]/Xy[:,0])
Xy=Xy[:,:]

sc=StandardScaler()
sc.fit(Xy)
Xy_s=sc.transform(Xy)

nt=Xy.shape[0]


r=np.random.random(nt)
a=np.nonzero(r>0.5)
b=np.nonzero(r<0.5)
x_train=Xy_s[a[0],:-1]
x_test=Xy_s[b[0],:-1]

y_train=Xy_s[a[0],-1]
y_test=Xy_s[b[0],-1]

import tensorflow as tf
import tensorflow.keras as keras
K = keras.backend

def correct_model(n1,n2):
    input1 = keras.layers.Input(shape=[n1])
    z = keras.layers.Dense(40, activation="relu")(input1)
    z = keras.layers.Dropout(0.1)(z)
    z = keras.layers.Dense(40, activation="relu")(z)
    z = keras.layers.Dropout(0.1)(z)
    output = keras.layers.Dense(n2)(z)
    model = keras.models.Model(
        inputs=[input1], outputs=[output])
    return model

itrain=0
if itrain==1:
    cluttCorrect=correct_model(7,1)
    
    cluttCorrect.compile(optimizer=tf.keras.optimizers.Adam(),  \
                         loss='mse',\
                         metrics=[tf.keras.metrics.MeanSquaredError()])
    
    
    history = cluttCorrect.fit(x_train, y_train, batch_size=32,epochs=50,
                               validation_data=(x_test, y_test))
    
    cluttCorrect.save("cluttCorrect.h5")
else:
    cluttCorrectmodel = tf.keras.models.load_model('cluttCorrect.h5')


fh=Dataset("dec2018_training_set.nc")
Xy_Dec=fh["trainDataSet"][:]
Xy_Dec_s=sc.transform(Xy_Dec)
xL=[]
for i in range(65,88):
    a=np.nonzero(np.abs(Xy[:,2]-i)<1e-3)
    b=np.nonzero(np.abs(Xy[:,1][a]-1)<1e-3)
    r1=Xy[a[0][b],0].mean()
    r2=Xy[a[0][b],-1].mean()
    a=np.nonzero(np.abs(Xy_Dec[:,2]-i)<1e-3)
    b=np.nonzero(np.abs(Xy_Dec[:,1][a]-1)<1e-3)
    r1b=Xy_Dec[a[0][b],0].mean()
    r2b=Xy_Dec[a[0][b],-1].mean()
    print(r1/r2,r1b/r2b)
    xL.append([i,r1/r2,r1b/r2b])

xL=np.array(xL)
