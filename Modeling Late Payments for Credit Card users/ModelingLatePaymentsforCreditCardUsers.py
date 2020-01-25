#!/usr/bin/env python
# coding: utf-8

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import csv
import numpy as np
import pandas as pd
import seaborn as sns 
import math
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 

# is the entry a date
def isDate(dl):
    try:
        # check if int
        val = int(dl)
        # check if 8 digits
        if(math.log10(val)>7 and math.log10(val)<8 and val//1e7):
            return 1
        return 0
    except ValueError:
        return 0


def getDateCols(df):
    cols = []
    for col in df.columns:
        counter = 0
        while counter<5:
            if not isDate(df[col][counter]):
                break
            counter+=1
        if counter == 5 :
            cols.append(col)
    return cols

#gets a one hot vector of weekdays for the input vector
def splitDates(df):
    # get the columns which contain dates
    dateCols = getDateCols(df)
    # extract the columns contain dates
    dates = df[dateCols]
    # extract year, month, day
    dates = pd.concat((dates//1e4, (dates%1e4)//1e2, dates%1e2), axis = 1)
    # rename duplicate column names 
    dates.columns = ['{}-ver{}'.format(dates.columns[c], c) for c in range(dates.columns.shape[0])]
    #print(dates)
    df = df.drop(dateCols, axis = 1)
    df = pd.concat((df, dates), axis = 1)
    #print(df)
    return df #days

def getNonNumCols(df):
    nonNumericCols = []
    for col in df.columns:
        if(not pd.to_numeric(df[col], errors='coerce').notnull().all()):
            nonNumericCols.append(col)
    return nonNumericCols

class MLP(nn.Module):
    def __init__(self, N):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(N, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x



def main():
    for i in range(3):
        current_num = i+1
        training = pd.read_csv("target{}_training_data.csv".format(current_num))
        training_labels = pd.read_csv("target{}_training_label.csv".format(current_num))
        test = pd.read_csv("target{}_test_data.csv".format(current_num))
        testIDs = test["ID"]

        # get rid of features that are predominantly null 
        training = training.loc[:,training.isna().sum()/training.shape[1]<0.5]
        test = test.T.reindex(training.columns).T
        
        # get rid of IDs since they provide no useful information
        training.drop(["ID"], axis = 1, inplace=True)
        test.drop(["ID"], axis = 1, inplace=True)
        training_labels.drop(["ID"], axis = 1, inplace=True)


        # split dates (of format YYYYMMDD to seperate features)
        training = splitDates(training)
        test = splitDates(test)

        # if a column has less than 30 unique values consider it a category
        cats = training.loc[:, training.nunique() < 30].columns

        # split categories to one hot vectors
        training = pd.concat((training.drop(cats, axis = 1), pd.get_dummies(training[cats].astype(str))),axis=1)
        test = pd.concat((test.drop(cats, axis = 1, errors = 'ignore'), pd.get_dummies(test[cats.intersection(test.columns)].astype(str))),axis=1)

        # fill null values
        test = test.fillna(training.mean())
        training = training.fillna(training.mean())

        # process remaining non numeric values in columns
        nonNumCols = getNonNumCols(training)

        # drop them
        training = training.drop(nonNumCols, axis = 1)
        test = test.drop(nonNumCols, axis = 1)

        # normalize the numeric values
        test=(test-training.min())/(training.max()-training.min())
        training=(training-training.min())/(training.max()-training.min())
        test = test.T.reindex(training.columns).T.fillna(0)

        # Training params
        epoch = 10
        batchsize = 128

        # create a model with the number of features matching the input
        model = MLP(training.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        # Using binary cross entropy as the loss function
        loss_func = nn.BCELoss()

        # split to training and validation
        x_train, x_valid, y_train, y_valid = train_test_split(training,training_labels, test_size=0.2, random_state=42, stratify=training_labels)


        for trial in range(epoch):
            # Train the model
            numBatches = math.ceil(x_train.shape[0]/batchsize)
            for el in range(numBatches):
                optimizer.zero_grad()
                batch = x_train[el*batchsize:((el+1)*batchsize)]
                ts = torch.FloatTensor(batch.values)
                real_ys = torch.FloatTensor(np.array(y_train[el*batchsize:el*batchsize+batch.shape[0]]))
                ys = model(ts)

                loss = loss_func(ys,real_ys)
                loss.backward()
                optimizer.step()
            real_ys = torch.FloatTensor(np.array(y_train))
            ys = model(torch.FloatTensor(x_train.values))
            val_ys = torch.FloatTensor(np.array(y_valid))
            vys = model(torch.FloatTensor(x_valid.values))
            print(roc_auc_score(y_valid,vys.detach().numpy()))
            print("train loss: {}, validation loss: {}".format(loss_func(ys,real_ys),loss_func(vys,val_ys)))


        val_ys = model(torch.FloatTensor(x_valid.values)).detach().numpy()
        print(roc_auc_score(y_valid,val_ys))

        #test on the required set
        preds = model(torch.FloatTensor(test.values)).detach().numpy()
        
        np.savetxt("target{}_test_predictions.csv".format(current_num),np.concatenate((testIDs.to_numpy().reshape((testIDs.shape[0],1)), preds),axis=1), delimiter = ',')


if __name__ == '__main__':
    main()  