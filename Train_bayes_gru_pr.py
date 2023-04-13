#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
import os
from os.path import join, abspath, dirname, realpath
import sys
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
from dateutil import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import tensorflow as tf
from dateutil.relativedelta import relativedelta

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
import argparse
import csv

from functions import *

import multiprocessing as mp
from multiprocessing import Process

from functools import partial
from skopt import gp_minimize
from skopt import space
import random



#given a model and a dataset it makes predictions and give back the AUC PR
def prd(df_p,model):

    predictions=[]
    true_values=[]
    orders=np.unique(df_p['order'])
    
    #processes sequences one by one depending on their order
    for o in orders:
    
        df_batch=df_p[df_p['order']==o]
        df_batch=df_batch.sort_values(by='date')
        df_batch.reset_index(drop=True, inplace=True)
        
        
        X=df_batch.loc[:, ['temp_avg','sun_perc','precip_tot','pres_avg',
                             'cloud_avg','wind_avg','hum_avg',]].to_numpy()
        #resize in respect to keras input
        X=np.resize(X,(X.shape[0],1,X.shape[1]))
        
        pred=model.predict(X,batch_size=1)
        model.reset_states()
       
        pred=np.concatenate(pred).ravel()
        
        #does not take into consideration the predictions made for imputed data
        for number,j in enumerate(df_batch['imputation']):
             if not j:
                    predictions.append(pred[number])
                    true_values.append(df_batch['migraine_start'][number])
         
    predictions=np.asarray(predictions)
    true_values=np.asarray(true_values)
    

        
    precision, recall, thresholds = precision_recall_curve(true_values, predictions)
    auc_precision_recall=auc(recall, precision)
    
    auc_roc= round(roc_auc_score(true_values,predictions),3)
    return(auc_precision_recall)
    

#fucntion that trains and tests the model for the crossvalidation
def fit_lstm(df,df_test,fold,params,epochs=1):
   
    batch_size=1

    # Since most days are not migraine days, the two classes are not balanced 
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.asarray([0,1]),
                                                  y=df.loc[:,'migraine_start'])

    weights = {i : class_weights[i] for i in range(2)}
    
    orders=np.unique(df['order'])

    model=modelcreation_bayes(params)
    
    #one loop for every epoch
    for i in np.arange(epochs):
        #sequences are processed one by one
        for o in orders:
            df_batch=df[df['order']==o]
            df_batch=df_batch.sort_values(by='date')
            df_batch.reset_index(drop=True, inplace=True)

            X=df_batch.loc[:, ['temp_avg','sun_perc','precip_tot','pres_avg',
                             'cloud_avg','wind_avg','hum_avg',]].to_numpy()


            X=np.resize(X,(X.shape[0],1,X.shape[1]))


            Y=df_batch.loc[:,['migraine_start']]
            
            #for every sequence fit function is called for the same model
            model.fit(X,Y,epochs=1,batch_size=batch_size,shuffle=False, class_weight=weights, verbose=0)
            #reseting states after every group of data so that different characteristics of every group do not influence next group
            #and weather conditions of a previous group do not influence the predictions of the current group 
            model.reset_states()
            
            
    train_scores[f'Train fold: {fold}'] = prd(df,model=model)# * df.shape[1]/(df.shape[1]+df_test.shape[1])
    
    test_scores[f'Test fold: {fold}']=prd(df_test,model=model)
    
    times[f'Test fold: {fold}'] = df_test.shape[0]
    



#given the data and the parameters it creates a model and then performce cross-validation
def cross_val(data, params,epochs=1):
    s=pd.date_range(start=min(data['date']),end=max(data['date']))
    folds=np.array_split(s, 5)
    jobs=[]
    
    #cross validation in parallel for every fold
    for foldnumber,k in enumerate(folds):
        print(f'>>>>fold number:{foldnumber}')
        first_day=min(k)
        
        train, test = data_split(data,test_first_day=first_day,
                                 test_size=max(k)-min(k)+relativedelta(days=1))

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        #every fold is one job and all jobs run at the same time
        p=mp.Process(target= fit_lstm, args=[train,test,foldnumber,
                                             params,epochs])
        p.start()
        jobs.append(p)

    for t in jobs:
        t.join()
    time.perf_counter()


    #weighted average depending on the test size
    for i in test_scores.keys():
        corrected[i]=test_scores[i]*times[i]/sum(times.values())
    

    
    print(f'AUC calculated with weighted average cv: {sum(corrected.values())}')

    #returns the weighted average of the performance of the model after cv
    return(sum(corrected.values()))
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int,
                        help='[int] epochs for the trainig')
    args = parser.parse_args()

    return args



def save_results(diction,name):
    with open(name+'delete.csv', 'w') as f:
        for key in diction.keys():
            f.write("%s,%s\n"%(key,diction[key]))




#creates the model depending on the parameters we have chosen 

def modelcreation_bayes(params,cell_type='GRU'):
    #batch_size, units, layers, dropout=params
    units, layers, dropout=params
    new_model = tf.keras.Sequential()
    #creates layers one by one
    for i in range(layers):
        if cell_type=='LSTM':
            new_model.add(tf.keras.layers.LSTM(units, activation='tanh', return_sequences=True,
                                       stateful=True, batch_input_shape=(1,1,7)))
        elif cell_type=='GRU':
            new_model.add(tf.keras.layers.GRU(units, activation='tanh', return_sequences=True,
                                       stateful=True, batch_input_shape=(1,1,7)))
            
        if dropout!=0:
            new_model.add(tf.keras.layers.Dropout(dropout*0.1))

    new_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    opt=tf.keras.optimizers.Adam()
    new_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    #returns an empty model with the desired structure
    return(new_model)


#this function is optimised for the Bayesian optimization
#since we try to find it's maximum value, it is equal to -results of cv
def optimize(params,train,epochs):
    print("params_opt=",params)
    return(-cross_val(data=train,params=params, epochs=epochs))



def main(args):
    
    train =pd.read_csv('train_plus.csv', sep=',', index_col=False, dtype='unicode')#, error_bad_lines=False
    train['date']=pd.to_datetime(train['date'])
    train['migraine_start']=train['migraine_start'].astype(int)
    train['order']=train['order'].astype(float).astype(int)
    train['imputation']=train['imputation'].astype(float).astype(int)
    #train.iloc[:,3:10]=train.iloc[:,3:10].astype(float)
    train.loc[:, ['temp_avg','sun_perc','precip_tot',
                  'pres_avg','cloud_avg','wind_avg','hum_avg',]] = train.loc[:, ['temp_avg','sun_perc','precip_tot','pres_avg',
                  'cloud_avg','wind_avg','hum_avg',]].astype(float)
    
    train=train.sort_values(['order', 'date'])
    train.reset_index(drop=True, inplace=True)
    
    weather_full=pd.read_csv("weather_full.csv")
    weather_full['date']=pd.to_datetime(weather_full['date'])
    weather_full.loc[:, ['temp_avg','sun_perc','precip_tot',
                  'pres_avg','cloud_avg','wind_avg','hum_avg',]] = weather_full.loc[:, ['temp_avg','sun_perc','precip_tot','pres_avg',
                  'cloud_avg','wind_avg','hum_avg',]].astype(float)
    
    
    #Scaling of the data
    scaler = StandardScaler()
    df_clean_plus=train
    #scaling only the weather variables
    scaler.fit(train.iloc[:,3:-2])
    df_scaled = pd.DataFrame(scaler.transform(train.iloc[:,3:-2]))
    df_clean_plus.iloc[:,3:-2]=df_scaled
    del df_clean_plus


    start = time.time()


    #Bayesian optimization
    optimization_function = partial(optimize,
                                   train=train,
                                   epochs=args.epochs)

    #parameter space for bayesian opt
    param_space= [space.Integer(10,100, name="units"),
                  space.Integer(1,4, name="layers"),
                  space.Integer(0,4, name="dropout")]

    #the combination of parameters that gives the max value for the optimization function is passed in results
    result = gp_minimize(optimization_function, dimensions=param_space, n_calls=50,n_random_starts=1,verbose=True)


    #save_results(train_scores,name="train")
    #save_results(test_scores,name="test")
    print(result)
    end = time.time()
    print("Time:"+str(int(end - start)))
    
    


if __name__=='__main__':
    train_scores=mp.Manager().dict()
    test_scores=mp.Manager().dict()
    corrected=mp.Manager().dict()
    times=mp.Manager().dict()
    
    
    #comment out for py file
    args=parse_args()
    #turn this to a comment for py
    
    #class args:
    #    epochs=10

    random.seed(1995)
    main(args)


