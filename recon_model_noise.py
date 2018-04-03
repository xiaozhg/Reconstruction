import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import keras
from keras.models import Sequential, load_model
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
import os

from __future__ import division, print_function

import recon_utils as rcu

# parameters
train_recon_model=False  # if True, train new reconstruction model.  If False, import old model
data_path='/Users/dratner/Desktop/Data/Reconstruction/Data'
recon_model=os.path.join(data_path,'0322_model.h5')

# Load and split data
print('Loading data...')
x_train, y_train, temp_train, temp = rcu.prep_data(data_path)
print(y_train.shape,x_train.shape)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


#-----------------------------------
# plot data

#real data
show_features=False
if show_features:
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


#value=[[0.7,0.5,0.7,0.5]]
#neuron=[[2000,2000,2000,2000]]#,[500,800]]
#rate=[0.01]




drop=[0.7,0.5,0.7,0.5]
neuron=[2000,2000,2000,2000]#,[500,800]]
lr=0.01

if train_recon_model:
    print('Training reconstruction model...')
    model = rcu.baseline_model(neuron,drop,lr)
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset

    t_0=time.time()
    history=model.fit(x_train,y_train)
    t=time.time()-t_0
    print("t=%fs"%t)

#(estimator.model).save('0322_1_neurons:'+str(neuron[0])+'_dropout:'+str(value[0])+'_rate:'+str(rate[0])+'_model.h5')  # creates a HDF5 file
else:
    print('Loading reconstruction model...')

    model = load_model(recon_model)


for m in ['train','test']:
    if m=='train':
        y_pred_train=model.predict(x_train)
    if m=='test':
        y_pred_test=model.predict(x_test)



    if m=='train':
        temp_train=np.sqrt(metrics.mean_squared_error(y_pred_train,y_train))/y_train.mean()
        temp_mse=metrics.mean_squared_error(y_pred_train,y_train)
    if m=='test':
        temp_test=np.sqrt(metrics.mean_squared_error(y_pred_test,y_test))/y_test.mean()
error={}
error[(tuple(neuron),tuple(drop),lr)]={'train':temp_train,'mse':temp_mse,'test':temp_test,}
print(tuple(neuron),tuple(drop),lr,temp_train,temp_test,temp_mse)
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
best_ex=np.where(mse_test==np.min(mse_test))[0][0]
worst_ex=np.where(mse_test==np.max(mse_test))[0][0]
med_ex=np.where(mse_test==np.percentile(mse_test,70,interpolation='nearest'))[0][0]
plt.plot(y_pred_test[worst_ex,:]); plt.plot(y_test[worst_ex,:]);
plt.xlabel('time (arb. units)'); plt.ylabel('power (arb. units)'); plt.title('Worst case'); plt.show()
plt.plot(y_pred_test[med_ex,:]); plt.plot(y_test[med_ex,:]);
plt.xlabel('time (arb. units)'); plt.ylabel('power (arb. units)'); plt.title('Median case'); plt.show()
plt.plot(y_pred_test[best_ex,:]); plt.plot(y_test[best_ex,:]);
plt.xlabel('time (arb. units)'); plt.ylabel('power (arb. units)'); plt.title('Best case'); plt.show()

# calculate log of error
y_err_train=np.log10(mse_train)
y_err_test=np.log10(mse_test)

# standard normalization
mse_mu=np.mean(y_err_train)
mse_sig=np.std(y_err_train)
y_err_norm_train=(y_err_train-mse_mu)/mse_sig
y_err_norm_test=(y_err_test-mse_mu)/mse_sig

# add predictions of first model to features of error model
x_err_train=np.concatenate((x_train,y_pred_train),axis=1)
x_err_test=np.concatenate((x_test,y_pred_test),axis=1)


#-------------------------------------
# fit model
print('Training error model...')

# error model parameters
nb_epoch=2000
epoch_per_chunk=20
batch_size=1000
lr=1e-3

model = rcu.err_model(num_feat=x_err_train.shape[1])

# Optimizers and metrics
Ada=keras.optimizers.Adagrad(lr=lr, epsilon=1e-12, decay=0.000) #0.002 for relu #5,1e-10,0 originally
Adam=keras.optimizers.Adam(lr=lr)
#RMS=keras.optimizers.RMSprop(lr=2, rho=0.9, epsilon=1e-10, decay=0.15)
model.compile(loss='mean_squared_error', optimizer=Adam,metrics=['mse'])
#model.compile(loss='mean_absolute_error', optimizer=Ada,metrics=['mse'])

#estimator = KerasRegressor(build_fn=rcu.err_model, epochs=500, batch_size=75, verbose=1)
#history=estimator.fit(x_err_train,y_err_train)
n_chunk=np.int(nb_epoch/epoch_per_chunk)
t0=time.time()
test_err=np.zeros(n_chunk)
train_err=np.zeros(n_chunk)
epoch_count=np.zeros(n_chunk)
for e in range(n_chunk):
    model.fit(x_err_train, y_err_norm_train, nb_epoch=epoch_per_chunk, batch_size=batch_size, verbose=0)
    err_norm_test_pred = model.predict(x_err_test)
    err_norm_train_pred = model.predict(x_err_train)
#    test_err[e]=metrics.mean_absolute_error(err_norm_test_pred,y_err_norm_test)
#    train_err[e]=metrics.mean_absolute_error(err_norm_train_pred,y_err_norm_train)
    test_err[e]=metrics.mean_squared_error(err_norm_test_pred,y_err_norm_test)
    train_err[e]=metrics.mean_squared_error(err_norm_train_pred,y_err_norm_train)
    epoch_count[e]=(e+1)*epoch_per_chunk
    t_e=time.time()-t0
    print('%d/%d epochs in %d seconds, train error: %0.3f, test error: %0.3f' % (epoch_count[e],nb_epoch,t_e,train_err[e],test_err[e]))


#-----------------------------
# post processing
#-----------------------------


err_norm_test_pred = model.predict(x_err_test)
err_norm_train_pred = model.predict(x_err_train)

plt.plot(err_norm_train_pred,y_err_norm_train,'.'); plt.show()
plt.plot(err_norm_test_pred,y_err_norm_test,'.'); plt.show()

# de-normalize predictions
err_test_pred = err_norm_test_pred*mse_sig + mse_mu
err_train_pred = err_norm_train_pred*mse_sig + mse_mu
plt.plot(err_train_pred,y_err_train,'.');
plt.xlabel('log10 pred MSE'); plt.ylabel('log10 true MSE'); plt.title('Train set'); plt.show()
plt.plot(err_test_pred,y_err_test,'.');
plt.xlabel('log10 pred MSE'); plt.ylabel('log10 true MSE'); plt.title('Test set'); plt.show()
#plt.plot(10**err_train_pred,10**y_err_train,'.');
#plt.xlabel('pred MSE'); plt.ylabel('true MSE'); plt.title('Train set'); plt.show()
plt.plot(10**err_test_pred,10**y_err_test,'.');
plt.xlabel('pred MSE'); plt.ylabel('true MSE'); plt.title('Test set'); plt.show()


# plot new worst-case
good_shots=np.where(err_test_pred<np.median(err_test_pred))[0]
#good_shots=np.where(err_test_pred<-5)[0]
worst_ex=np.where(mse_test==np.max(mse_test[good_shots]))[0][0]
plt.plot(y_pred_test[worst_ex,:]); plt.plot(y_test[worst_ex,:]);
plt.xlabel('time (arb. units)'); plt.ylabel('power (arb. units)'); plt.title('Worst case'); plt.show()


# training
plt.plot(epoch_count,train_err); plt.plot(epoch_count,test_err);
plt.xlabel('epochs'); plt.ylabel('loss (MAE)'); plt.legend(['train','test']);  plt.show()

