import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianDropout

def prep_data(data_path):

    num_out=100 #we want about num output features, input power features are half of that
    feature_num=170 #total # of features
    Division=feature_num/num_out
    Tfeature_num=340 #total # of power+pspec
    pspec_num=30 #we want about pspec_num pspec points
    Division_pspec=(Tfeature_num-feature_num)/pspec_num
    power_ratio=2
    nts=0.1


    #real data
    data_num=20000 #total # of data
    #truth=np.array(pd.read_csv('20180321groundtruth.csv'))
    features=np.array(pd.read_csv(os.path.join(data_path,'20180322features.csv')))
    conv=np.array(pd.read_csv(os.path.join(data_path,'20180322newpower_conv_sigt.csv')))
    new_pspec=np.array(pd.read_csv(os.path.join(data_path,'20180322newpspec_conv_0_08.csv')))
    x_train=[[] for x in range(data_num)]
    y_train=[[] for x in range(data_num)]
    temp_train=[[] for x in range(data_num)]
    np.random.seed(seed=5)
    noise=np.random.randn(data_num,Tfeature_num)
    K_list=[]
    for i in range(data_num):
        for k in range(feature_num):
        #power y_train
            if k%(Division*2)==0:
                y_train[i].append(features[i][k])
        norm=sum(y_train[i])
        for j in range(len(y_train[i])):
            y_train[i][j]=y_train[i][j]/(norm+0.0)
        #power x_train
        for k in range(feature_num):
             if k%(Division*2)==0:
                temp_train[i].append(conv[i][k]*(1+nts*noise[i][k]))
                if (k/(Division*2))%power_ratio==0:
                    x_train[i].append(conv[i][k]*(1+nts*noise[i][k]))
                    if i==0:
                        K_list.append(k/(Division*2))
        
        norm=sum(temp_train[i])    #need to change to temp_train
        #norm_x=sum(x_train[i])
        for j in range(len(x_train[i])):
            x_train[i][j]=x_train[i][j]/(norm+0.0)
        for j in range(len(temp_train[i])):
            temp_train[i][j]=temp_train[i][j]/(norm+0.0)
        
        #pspec x_train
        temp=len(x_train[i])
        
        #for k in range(len(new_pspec[0])):
         #   if k>400 and k<425:
          #      x_train[i].append(new_pspec[i][k])
        
        for k in range(feature_num,Tfeature_num):
            if k-feature_num>76 and k-feature_num<95:
                x_train[i].append(features[i][k]*(1+nts*noise[i][k]))
        a_norm=sum(x_train[i][temp:len(x_train[i])])
        for j in range(temp,len(x_train[i])):
            x_train[i][j]=x_train[i][j]/(a_norm+0.0)
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    temp_train=np.array(temp_train)

    return x_train, y_train, temp_train, temp

def baseline_model(neuron,drop,lr):
    # create model
    model = Sequential()
    #model.add(GaussianDropout(0.05,input_shape=(25,)))
    #model.add(Dense(850, input_dim=900, kernel_initializer='normal', activation='linear'))
    model.add(Dense(neuron[0], input_dim=61, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(drop[0]))
    model.add(Dense(neuron[1], activation='relu'))
    model.add(Dropout(drop[1]))
    model.add(Dense(neuron[2], activation='relu'))
    model.add(Dropout(drop[2]))
    model.add(Dense(neuron[3], activation='relu'))
    model.add(Dropout(drop[3]))
    #model.add(Dropout(j[2]))
    #model.add(Dense(n[2], activation='relu'))
    #model.add(Dropout(j[2]))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(n[2]))
    model.add(Dense(85, activation='softmax'))
            #sgd = SGD(lr=l, decay=1e-8, momentum=0.9, nesterov=True)
            
    Ada=keras.optimizers.Adagrad(lr=lr, epsilon=1e-12, decay=0.000) #0.002 for relu #5,1e-10,0 originally
    #RMS=keras.optimizers.RMSprop(lr=2, rho=0.9, epsilon=1e-10, decay=0.15)
    #model.compile(loss='mean_squared_error', optimizer=Ada,metrics=['mse'])
    model.compile(loss='categorical_crossentropy', optimizer=Ada,metrics=['mse'])
    return model


def err_model(num_feat,num_neur=256/2,f_drop=0.5,f_drop2=0.5):
    # create model
    model = Sequential()
    #model.add(GaussianDropout(0.05,input_shape=(25,)))
    #model.add(Dense(850, input_dim=900, kernel_initializer='normal', activation='linear'))
    model.add(Dense(num_neur, input_dim=num_feat, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(f_drop))
    model.add(Dense(num_neur/4, activation='relu'))
    model.add(Dropout(f_drop2))
    model.add(Dense(num_neur/16, activation='relu'))
    model.add(Dropout(f_drop2))
    #model.add(Dropout(j[2]))
    #model.add(Dense(n[2], activation='relu'))
    #model.add(Dropout(j[2]))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(n[2]))
    model.add(Dense(1, activation='linear'))
            #sgd = SGD(lr=l, decay=1e-8, momentum=0.9, nesterov=True)
            
    return model




