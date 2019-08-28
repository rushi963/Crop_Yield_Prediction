'''
- - - - - - Team Members and IDs:- - - - - - - - 

Rushikesh Nalla - 111971011
Purvik Shah - 112045155
Sriram Gattu - 112007687
Kaustubh Sivalenka - 111987216

The purpose of this program is to build traindata and testdata,build a ridge regresion model,predict crop yield for 2013 year,and report the rmse error for the predictions made using the best regularisation parameter

Data concept "Ridge regression" is used in this file

- - - - - - - Type of System used:- - - - - - - 

NVIDIA DGX-1 was used for computational requirements.

A Singularity image was built and was mounted to the working directory (on DGX-1) to use the requisite
software libraries to realize  project objectives.

https://drive.google.com/file/d/1WATPH8xna_xSQyvWrnMhAdm98rwYC5-L contains the singularity image
tf_writable.simg - Singularity image mounted to working directory.
 
''' 
import sys
import os
import numpy as np
from pyspark import SparkContext
from io import BytesIO
import csv
import pandas as pd
import subprocess
from sklearn.linear_model import Ridge
print("Loading pandas pickle object which ecompasses crop information of form (1)year,2)Histogram Array,3)Crop Yeild")
data = pd.read_pickle("trainingdata.txt")
print("Crop data succesfully loaded")
print("Train data includes all crop information for all years < 2013")
print("Test data includes information for year= 2013 , on whch we are going to make predictions on")
traindata = data[data['Year'] < "2013" ] #train data,we will build our model on crop infrmation for all years less than 2013
testdata = data[data['Year'] == "2013" ] #test data,we will predict crop yield for 2013 year and report RMSE error for that

#--------------------------------------------------------------------------------
#build trainX and trainY data
histoftrain = traindata['Histogram Array'].values 
cropyeildoftrain = traindata['Crop Yeild'].values
rows = len(histoftrain)
a,b,c = histoftrain[0].shape
cols = a*b*c
trainXdata = np.ndarray(shape = (rows,cols),dtype = float)
trainYdata = np.array(traindata['Crop Yeild'].values)
for i in range(len(histoftrain)):
	trainXdata[i,:] = histoftrain[i].flatten() 
print("Succesfully built training data")

#---------------=----------------------------------------------------------------
#build testX and testY data
histoftest = testdata['Histogram Array'].values 
cropyeildoftest = testdata['Crop Yeild'].values
rows = len(histoftest)
a,b,c = histoftest[0].shape
cols = a*b*c
testXdata = np.ndarray(shape = (rows,cols),dtype = float)
testYdata = np.array(testdata['Crop Yeild'].values) 
for i in range(len(histoftest)):
	testXdata[i,:] = histoftest[i].flatten() 
print("Succesfully built test data")
#---------------------------------------------------------------------------------

#Paramater hypertuning of ridge regression strength lambda
lambdavec = [0.001,0.01,0.05,0.1,0.5,0.9,1,10,50,100,200,300,375,450,525,575,600,700,775,800,1000,1200,1700,2500, 5000, 10000,50000,100000,500000,750000,1000000]
#---------------------------------------------------------------------------------
print("Training ridge regression model using scikit learn")
error = 9999999999999
bestlambda = 0
for alpha in lambdavec:
	print("L2-Regularization param lambda", alpha)
	ridgemodel = Ridge(alpha=alpha)
	ridgemodel.fit(trainXdata, trainYdata)
	testY = ridgemodel.predict(testXdata)
	curerror = np.sqrt(np.mean((testYdata - testY)**2))
		 
	if(curerror < error):
		error = curerror
		bestlambda = alpha
	print ("RMSE error on test data", curerror)
	
print("Best L2-Regularization parameter", bestlambda)
print("RMSE Error on test data for best lambda", error)