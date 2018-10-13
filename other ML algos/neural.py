import pandas as pa
import numpy as np
import matplotlib.pyplot as plt  

datset=pa.read_csv("Churn_Modelling.csv")

#%%
x= datset.ix[:,3:13].values
y= datset.ix[:,-1].values
#%%
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencode_x1=LabelEncoder()
x[:,1]=labelencode_x1.fit_transform(x[:,1])

labelencode_x2=LabelEncoder()
x[:,2]=labelencode_x2.fit_transform(x[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]               #prevention from dummy variable trap
#%%

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20 , random_state=0)
#%%
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)
#%% formation of the ANN model

import keras
from keras.models import Sequential
from keras.layers import Dense

#%%initialise ANN(sequential)
classifier=Sequential()
classifier.add(Dense(output_dim=6,input_dim=11,activation="relu",init="uniform"))

#%%adding the second hidden layer
classifier.add(Dense(output_dim=6,activation="relu",init="uniform"))

#%%adding the output layer

classifier.add(Dense(output_dim=1,activation="sigmoid",init="uniform"))

#%%compililng the ANN
classifier.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

#%%fitting ANN to the training set
classifier.fit(xtrain,ytrain,batch_size=10,nb_epoch=100)
#%%
ypred=classifier.predict(xtest)
#%%
ypred=(ypred>0.5)
#%%
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)













