'''
- - - - - - Team Members and IDs:- - - - - - - - 

Rushikesh Nalla - 111971011
Purvik Shah - 112045155
Sriram Gattu - 112007687
Kaustubh Sivalenka - 111987216

The purpose of this file is to run a ermutation test to verify the validity and performance of the formerly built ridge regression model. Our Hypothesis is - "Are these features a good representation for predicting the crop yield". We also present a p-value plot. 

Data Concept "Hypothesis testing (Permutation Test)" is used in this file

- - - - - - - Type of System used:- - - - - - - 

NVIDIA DGX-1 was used for computational requirements.

A Singularity image was built and was mounted to the working directory (on DGX-1) to use the requisite
software libraries to realize  project objectives.

https://drive.google.com/file/d/1WATPH8xna_xSQyvWrnMhAdm98rwYC5-L contains the singularity image
tf_writable.simg - Singularity image mounted to working directory.

'''
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from pyspark import SparkContext
from io import BytesIO
import csv
import pandas as pd
import subprocess
from sklearn.linear_model import Ridge
print("Training data=> data for all years < 2013;Test data=>data for year=2013")
print("Loading trainingdata,testdata......")

data = pd.read_pickle("trainingdata.txt")
traindata = data[data['Year'] < "2013" ]
testdata = data[data['Year'] == "2013" ]
#--------------------------------------------------------------------------------
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
print("succesfully built training and test data")
print(" Initiating Permutation test for 20 permutations")
 
error = []
'''
Best L2-Regularization parameter 100000
RMSE Error on test data for best lambda 10.072676333391385

'''
bestlambda = 10000 #obtained this value from previous ridge regression code via parameter tuning
besterror = 10.072676333391385 #Best Error reported from Ridge Regression
for i in range(0,20):
	ridgemodel = Ridge(alpha=bestlambda)
	permutedtrainYdata = np.random.permutation(trainYdata)
	ridgemodel.fit(trainXdata, permutedtrainYdata)
	testY = ridgemodel.predict(testXdata)
	curerror = np.sqrt(np.mean((testYdata - testY)**2))
	print("RMSE error for permutation = ",i+1,"is",curerror)
	error.append(curerror)
print("List of errors for all permutations",error)
print("Min error",min(error))
print("Max error",max(error))
num = sum(temp <= besterror for temp in error)
print("pvalue = ",num/20)


plt.hist(x=error, bins=20, range=[10, 11.5], color='orangered',alpha=0.75, rwidth=0.999, label='RMSE Histogram for 20 Permutations')
plt.axvline(besterror, color='blue', label='Best Error from Ridge Regression')
plt.xticks(np.arange(min(error), max(error)+1, 0.5))
plt.legend()
plt.xlabel('RMSE Errors for permutation tests')
plt.ylabel('Frequency')
plt.title('Histogram Plot for RMSE Errors of various permutations')
plt.savefig("permutation_test_ridge.png")
print(plt.show())
