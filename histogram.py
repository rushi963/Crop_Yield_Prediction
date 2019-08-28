'''
- - - - - - Team Members and IDs:- - - - - - - - 

Rushikesh Nalla - 111971011
Purvik Shah - 112045155
Sriram Gattu - 112007687
Kaustubh Sivalenka - 111987216

The purpose of this program is to fetch all the information of all mutispectral images stored in .npy files, convert them into histograms(dimensionality reduction), and map the information in the   form of (year,histogram array,crop yeild) and store the resultant spark rdd as a pandas pickle object which would used by LSTM and ridge regression models.

DataFrame used in this program is Spark

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
sc = SparkContext.getOrCreate()
from io import BytesIO
import csv
import pandas as pd
import subprocess
'''
Function to convert images represented as nupy array of  pixels to histograms
This function performs dimensionality reduction
'''
def convertimagehistogram(matrix,histogrambins,binsbandsinfo):
	rows,cols,dims = matrix.shape
	histogramarrayforoneband = np.zeros((binsbandsinfo[0],int(dims/binsbandsinfo[1])),dtype = float)
	histogramarrayforallbands = [histogramarrayforoneband for i in range(binsbandsinfo[1])]
	for i in range(0,dims):
		tempband = int(i%numbands)
		temptime = int(i/binsbandsinfo[1])
		tempimage = matrix[:,:,i]
		temphistogram = np.transpose(np.array((np.histogram(tempimage,histogrambins)[0])))
		temphistogram = temphistogram/(1.0*np.sum(temphistogram))
		histogramarrayforallbands[tempband][:,temptime] = temphistogram
	finalhistograms = np.dstack(histogramarrayforallbands)
	return finalhistograms

'''
Modis_dir specfies the folder encompassing .npy fils which encompasses images of shape (48,48,414)
(48,48) constitutes to one image
(414) = (46*9)=> Each image is captured 46 times a year(i.e once every 8 days) and each image is captured using 9 multispectral bands
'''
MODIS_dir="zoom_output"
files = []
x = []
y = []
z = []
'''
Code to fetch all numpy  file names present in a given folder
Each numpy(.npy) file has information about MODIS Reflctance and temperature in the form of numpy    array of pixels.
Format of .npy file => yyyy_stateid_countyid 
'''
'''
numbins specifies the bins of histogram of pixels
numbands specifies the number of multispectral bands
'''
print("Fetching Crop data from MODIS_Directory.....................")
data_df = pd.DataFrame()
for  _, _, fnames in os.walk(MODIS_dir):
	files.append(fnames)
files = files[0]
for i in range(0,len(files),100):
	'''
	processing 100 files at a time to resolve memeory issues
	'''
	temp = files[i:i+100]
	print("Current file list being processed....",temp)
	images = sc.parallelize(temp)
	images = images.map(lambda x:(x,np.load(os.path.join(MODIS_dir,x))))
	numbins = 32
	numbands = 9
	histogrambins = sc.broadcast(list(np.linspace(0,6000,numbins+1)))
	binsbandsinfo = sc.broadcast([numbins,numbands])
	images = images.map(lambda x:(x[0],convertimagehistogram(x[1],histogrambins.value,binsbandsinfo.value)))
	predictinfo  = sc.textFile("yield_final.csv").mapPartitions(lambda line: csv.reader(line))
	predictinfo = predictinfo.map(lambda x:(x[0]+"_" + x[1]+"_" + x[2] + ".npy",(x[0],float(x[3]))))
	finalrdd = images.join(predictinfo).map(lambda x:x[1])
	finalrdd = finalrdd.map(lambda x :(x[1][0],x[0],x[1][1]))
	data = finalrdd.collect()
	for i in data:
		x.append(i[0])
		y.append(i[1])
		z.append(i[2])

print("Succesfully processed all files")
print("Succesfully converted spark rdd object of form (year,histogram array,crop yeild) to a pandas object with columns..1)year,2)Histogram Array,3)Crop Yeild")
'''
Storing pandas array as pickle object which encompasses year,histogram information and crop yeild
'''
data_df['Year'] = x
data_df['Histogram Array'] = y
data_df['Crop Yeild'] = z
data_df.to_pickle("trainingdata.txt")
print("Succesfully stored crop information in trainingdata_temp.txt  of the form ('Year','histogram','Crop Yeild') as a pandas pickle object which will be used by ridge regression and LSTM models to make predictions and report RMSE")