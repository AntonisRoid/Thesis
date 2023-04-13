#!/usr/bin/env python
# coding: utf-8

import time
start = time.time()
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
#import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import tensorflow as tf
from dateutil.relativedelta import relativedelta


# used for patients who have data both before and after data splits (dates)
def split_overlaps(first_period,second_period,same):
    
    data_missing=0
    #same=np.intersect1d(p_test,p_train)
    
    for p in same:
        first_data=sum(first_period.loc[first_period['patient_id']==p]['migraine_start'])*2 + first_period.loc[first_period['patient_id']==p].shape[0]
        
        second_data=sum(second_period.loc[second_period['patient_id']==p]['migraine_start'])*2+second_period.loc[second_period['patient_id']==p].shape[0]
        
        #if data of the first period are more then discard second period
        if first_data>second_data:
            data_missing=data_missing+second_period[second_period.patient_id==p].shape[0]
            second_period=second_period.drop(second_period[second_period.patient_id==p].index)
        
        #if data of the second period are more then discard first period
        elif first_data<second_data:
            data_missing=data_missing+first_period[first_period.patient_id==p].shape[0]
            first_period=first_period.drop(first_period[first_period.patient_id==p].index)
        
        #if two periods are equal, keep days that were more dense    
        elif second_data==first_data:
            if ((max(second_period.loc[second_period['patient_id']==p]['date'])-min(second_period.loc[second_period['patient_id']==p]['date'])).days< 
                (max(first_period.loc[first_period['patient_id']==p]['date'])-min(first_period.loc[first_period['patient_id']==p]['date'])).days):
                data_missing=data_missing+first_period[first_period.patient_id==p].shape[0]
                first_period=first_period.drop(first_period[first_period.patient_id==p].index)
            else:
                data_missing=data_missing+second_period[second_period.patient_id==p].shape[0]
                second_period=second_period.drop(second_period[second_period.patient_id==p].index)
    
    return(first_period, second_period,data_missing)



#given the first desired date of the test set and that the test set is 1 year long,
#this function splits the data in train and test set 

def data_split(x,test_first_day, test_size=relativedelta(years=1)):
    train=x.loc[((x['date']<test_first_day) | (x['date']>=test_first_day+test_size))]
    test=x.loc[((x['date']>=test_first_day) & (x['date']<test_first_day+test_size))]
    
    p_test=np.unique(train['patient_id'])
    p_train=np.unique(test['patient_id'])
    same=np.intersect1d(p_test,p_train)
    
    #patients with data in both train and test set are dealt with the split_overlaps function
    train,test,data_missing=split_overlaps(train,test,same=same)            
    return(train, test)


#patient's sequences are given an order number so that we adress them individually
def order_split(data):
    
    kept_patients=[]
    pat_id=np.unique(data['patient_id'])
    df_clean=pd.DataFrame(columns=list(data.columns))
    order=0
    
    #patients' data with gaps of missing data that are bigger than one month, are split in multiple sequences 
    for j,patient in enumerate(pat_id):

        pat_data=data[data['patient_id']==patient].sort_values(by='date')
        pat_data.reset_index(drop=True, inplace=True)
        p=pat_data['date']
        day_before_gap=[]
        #iteration to the days of each patients to find gaps inbetween
        for i in np.arange(1,p.shape[0]):
            gap = relativedelta(p[i],p[i-1])
            if gap.months>0 or gap.years>0:
                day_before_gap.append(i)

        if day_before_gap:

            slices=np.split(np.arange(0,p.shape[0]), day_before_gap)

            for group_of_days in slices:
                addition=pat_data.iloc[group_of_days,:]
                addition['order']=order

                df_clean = df_clean.append(addition, ignore_index = True)
                order=order+1
        else:
            pat_data['order']=order
            df_clean = df_clean.append(pat_data)
            order=order+1
    df_clean['order']=df_clean['order'].astype(int)
    return(df_clean)


#calculates the missing days between two dates
def m_days(x):
    first_day=min(x['date'])
    last_day=max(x['date'])
    
    days=pd.date_range(start=first_day,end=last_day) 

    x=x.set_index('date')

    m=days.difference(x.index)
    
    return(m)


#creates the imputed data. Missing days are classified as nonmigraine days and weather variables are taken from weather_full.csv
def fill_missing(df_clean,weather_full):
    
    missing= pd.DataFrame(columns=weather_full.columns.tolist())
    pat_order=np.unique(df_clean['order'])

    for patient in pat_order:
        x=df_clean[df_clean['order']==patient]

        missing_days=m_days(x)

        for day in missing_days:
            new_data=weather_full.loc[weather_full['date']==day,:]
            new_data.insert(loc=0,
              column='patient_id',
              value=np.unique(x['patient_id'])[0]) #possible bug. could be set as np.unique to be safe but this would take more time
            new_data.insert(loc=2,
              column='migraine_start',
              value=0)
            new_data.insert(loc=10,
              column='order',
              value=patient)
            new_data.insert(loc=11,
              column='imputation',
              value=1)
            
            missing=missing.append(new_data)

    missing['migraine_start']=missing['migraine_start'].astype(int)
    
    return(missing)
    