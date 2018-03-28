import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianDropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import random
import time

from __future__ import division, print_function

import recon_utils as rcu


x_train, y_train, temp_train, temp = rcu.prep_data()

print(y_train.shape,x_train.shape)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#real data
j=2

plt.plot(y_test[j],'-',color='blue',label='original power data')
plt.title('groundtruth power points')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(np.array(range(temp)),x_test[j][0:temp],'-*',color='green',label='1 fs resolution power data')#,label='3fs resolution x_train data')
#plt.plot(y_test[j],'-',color='green',label='original power data')
plt.title('power points')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(x_train[j][temp:len(x_test[j])],'-*',label='spectral power data')
plt.title('spectral power points')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.plot(x_train[j],label='x_test')
plt.plot(y_train[j],label='y_test')
plt.legend(loc='upper right')
plt.show()


value=[[0.7,0.5,0.7,0.5]]
neuron=[[2000,2000,2000,2000]]#,[500,800]]
rate=[0.01]


drop=value
error={}
for n in neuron:
    for j in drop:
        for l in rate:

            def baseline_model():
                # create model
                model = Sequential()
                #model.add(GaussianDropout(0.05,input_shape=(25,)))
                #model.add(Dense(850, input_dim=900, kernel_initializer='normal', activation='linear'))
                model.add(Dense(n[0], input_dim=61, kernel_initializer='uniform', activation='relu'))
                model.add(Dropout(j[0]))
                model.add(Dense(n[1], activation='relu'))
                model.add(Dropout(j[1]))
                model.add(Dense(n[2], activation='relu'))
                model.add(Dropout(j[2]))
                model.add(Dense(n[3], activation='relu'))
                model.add(Dropout(j[3]))
                #model.add(Dropout(j[2]))
                #model.add(Dense(n[2], activation='relu'))
                #model.add(Dropout(j[2]))
                #model.add(Dense(100, activation='relu'))
                #model.add(Dropout(n[2]))
                model.add(Dense(85, activation='softmax'))
                        #sgd = SGD(lr=l, decay=1e-8, momentum=0.9, nesterov=True)
                        
                Ada=keras.optimizers.Adagrad(lr=l, epsilon=1e-12, decay=0.000) #0.002 for relu #5,1e-10,0 originally
                #RMS=keras.optimizers.RMSprop(lr=2, rho=0.9, epsilon=1e-10, decay=0.15)
                #model.compile(loss='mean_squared_error', optimizer=Ada,metrics=['mse'])
                model.compile(loss='categorical_crossentropy', optimizer=Ada,metrics=['mse'])
                return model


            # fix random seed for reproducibility
            seed = 7
            np.random.seed(seed)
            # evaluate model with standardized dataset
            estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=75, verbose=1)


            t_0=time.time()
            history=estimator.fit(x_train,y_train)
            t=time.time()-t_0
            print("t=%fs"%t)



            for m in ['train','test']:
                if m=='train':
                    y_pred_train=estimator.predict(x_train)
                if m=='test':
                    y_pred_test=estimator.predict(x_test)



                if m=='train':
                    temp_train=np.sqrt(metrics.mean_squared_error(y_pred_train,y_train))/y_train.mean()
                    temp_mse=metrics.mean_squared_error(y_pred_train,y_train)
                if m=='test':
                    temp_test=np.sqrt(metrics.mean_squared_error(y_pred_test,y_test))/y_test.mean()
            error[(tuple(n),tuple(j),l)]={'train':temp_train,'mse':temp_mse,'test':temp_test,}
            print(tuple(n),tuple(j),l,temp_train,temp_test,temp_mse)
for u,v in error.iteritems():
    print(u,v)
        #plt.plot(range(len(y_train[15])),y_pred[3],'.-',color='red',label='predict')

        #if m=='train':
        #    plt.plot(range(len(y_train[15])),y_train[3],'.-',color='blue',label='training set')
        #if m=='test':
        #    plt.plot(range(len(y_train[15])),y_test[3],'.-',color='blue',label='testing set')

    #plt.legend(loc='upper right')

mse_train=np.mean((y_pred_train-y_train)**2,1)
mse_test=np.mean((y_pred_test-y_test)**2,1)
ex=np.where(mse_test==np.min(mse_test))[0][0]
ex=np.where(mse_test==np.max(mse_test))[0][0]
plt.plot(y_pred_test[ex,:]); plt.plot(y_test[ex,:]); plt.show()

# calculate log of error
y_err_train=np.log10(mse_train)
y_err_test=np.log10(mse_test)

# standard normalization
mse_mu=np.mean(y_err_train)
mse_sig=np.std(y_err_train)
y_err_train=(y_err_train-mse_mu)/mse_sig
y_err_test=(y_err_test-mse_mu)/mse_sig

# add predictions of first model to features of error model
x_err_train=np.concatenate((x_train,y_pred_train),axis=1)
x_err_test=np.concatenate((x_test,y_pred_test),axis=1)

# fit model
model = rcu.err_model(num_feat=x_err_train.shape[1])
#estimator = KerasRegressor(build_fn=rcu.err_model, epochs=500, batch_size=75, verbose=1)
#history=estimator.fit(x_err_train,y_err_train)
model.fit(x_err_train, y_err_train, nb_epoch=1000, batch_size=75, verbose=1)
err_test_pred = model.predict(x_err_test)
err_train_pred = model.predict(x_err_train)

plt.plot(err_train_pred,y_err_train,'.'); plt.show()
plt.plot(err_test_pred,y_err_test,'.'); plt.show()

#save model
value=[[0.5,0.5,0.5,0.5]]
neuron=[[2000,2000,2000,2000]]
rate=[0.01]

from keras.models import load_model
#(estimator.model).save('0322_1_neurons:'+str(neuron[0])+'_dropout:'+str(value[0])+'_rate:'+str(rate[0])+'_model.h5')  # creates a HDF5 file
del estimator.model  # deletes the existing model
estimator.model = load_model('neurons:'+str(neuron[0])+'_dropout:'+str(value[0])+'_rate:'+str(rate[0])+'_model.h5')
#estimator.model = load_model('neurons:'+str(neuron[0])+'_dropout:'+str(value[0])+'_rate:'+str(rate[0])+'_model.h5')


#history.history.keys()
#plt.plot(history.history['mean_squared_error'])
#plt.xlabel("epochs")
#plt.ylabel("mse")
