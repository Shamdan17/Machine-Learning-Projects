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


def getNonNumCols(df):
    nonNumericCols = []
    df = df.fillna(0)
    for col in df.columns:
        if(not pd.to_numeric(df[col], errors='coerce').notnull().all()):
            nonNumericCols.append(col)
    return nonNumericCols


class MLP(nn.Module):
    def __init__(self, N):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(N, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
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

    # read inputs
    training = pd.read_csv("training_data.csv")
    training_labels = pd.read_csv("training_label.csv")
    test = pd.read_csv("test_data.csv")

    # keep aside the IDs for printing
    testIDs = test["ID"]

    # This will keep track of all the outputs to be printed into the csv file (starting with the indices)
    output_cols = testIDs.to_numpy().reshape((testIDs.to_numpy().shape[0],1))
    output_cols

    # extract non numerical cols
    nonNumCols = getNonNumCols(training)

    # drop non numeric columns
    training = training.drop(nonNumCols, axis=1)
    test = test.drop(nonNumCols, axis=1)

    # fill null values with the mean
    test = test.fillna(training.mean())
    training = training.fillna(training.mean())

    # if a column has less than 40 or more than 2 unique values consider it a category
    cats = training.loc[:,training.nunique()<40].columns.intersection(training.loc[:,training.nunique()>2].columns)

    #extract dummies
    training = pd.concat((training.drop(cats, axis=1), pd.get_dummies(training[cats].astype(str))), axis=1)
    test = pd.concat((test.drop(cats,axis=1), pd.get_dummies(test[cats].astype(str))), axis=1)
    test = test.T.reindex(training.columns).T.fillna(0)

    # drop ID as it is not needed
    training.drop("ID", axis = 1, inplace=True)
    test.drop("ID", axis = 1, inplace=True)
    training_labels = training_labels.drop("ID", axis = 1)

    # normalize the numeric values
    test = (test-training.min())/(training.max()-training.min())
    training=(training-training.min())/(training.max()-training.min())

    # make sure both test and training have same columns in order
    test = test.T.reindex(training.columns).T

    for featureNo in range(6):
        print("Current feature: {}".format(featureNo+1))

        # extract the required labels 
        set1 = training_labels[training_labels.columns[featureNo]].dropna()
        # extract the features corresponding to the required labels 
        training_set = training.iloc[set1.index]
        training_set_labels = set1

        # hyperparameters
        epoch = 6
        batchsize = 64

        # split to training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(training_set,training_set_labels, test_size=0.2, random_state=42, stratify=training_set_labels)


        # model and optimizer
        model = MLP(x_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_func = nn.BCELoss()

        for trial in range(epoch):
            # Train the model
            numBatches = math.ceil(x_train.shape[0]/batchsize)
            for el in range(numBatches):
                optimizer.zero_grad()
                batch = x_train[el*batchsize:((el+1)*batchsize)]
                ts = torch.FloatTensor(batch.values.squeeze())
                real_ys = torch.unsqueeze(torch.FloatTensor(np.array(y_train[el*batchsize:el*batchsize+batch.shape[0]])),1)
                ys = model(ts)
                loss = loss_func(ys,real_ys)
                loss.backward()
                optimizer.step()
            real_ys = torch.FloatTensor(np.array(y_train))
            ys = model(torch.FloatTensor(x_train.values))
            val_ys = torch.FloatTensor(np.array(y_valid))
            vys = model(torch.FloatTensor(x_valid.values))
            print(roc_auc_score(y_valid,vys.detach().numpy()))
            print("train loss: {}, validation loss: {}".format(loss_func(ys,torch.unsqueeze(real_ys, 1)),loss_func(vys,torch.unsqueeze(val_ys, 1))))

        test_tns = torch.FloatTensor(test.values.squeeze())
        output_cols = np.concatenate((output_cols,model(test_tns).detach().numpy()),axis=1)


    np.savetxt("test_predictions.csv",output_cols, delimiter = ',')


if __name__ == '__main__':
    main()  