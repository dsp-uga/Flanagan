"""
Resource for scikit-learn SVM: http://scikit-learn.org/stable/modules/svm.html
"""

import os
import numpy as np
from glob import glob
from scipy import misc
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

import time


def flatten_matrices(dirs):
	"""
	This function flattens the images of entire dataset
	For a sample of N images of dimension (X, Y), this function stores them in matrix of dimension (X*Y, N)
	Appends flat matrices over all the samples
	Parameters:
		dirs: path to dataset with frames
	Return:
		T: a flat matrix of total images. Dimension: (\sum (X_i * Y_i), 100) <- for cilia segmentation dataset
	"""
	T = np.array([])	# total images

	fcount = 0
	initialT = 0
	for d in dirs:
		print("dir ",fcount,d)
		V = np.array([])
		initialV = 0
		for i in range(100):
			new_image = misc.imread(d+"frame00"+str(i).zfill(2)+".png", flatten="true", mode = "L")
	#		print(new_image.shape)
			img = np.reshape(new_image, new_image.shape[0]*new_image.shape[1])
			
			if initialV:
				V = np.vstack((V, img))
			else:
				V = img
				initialV = 1
		V = np.transpose(V)
		if initialT:
			T = np.concatenate((T,V))
		else:
			T = V
			initialT = 1

	return T


def flatten_masks(dirs):
	"""
	This function converts masks to one-dimension and appends the masks 
	over all the samples
	Parameters:
		Path to training dataset. Assuming they include <traindataset>/<hash>/mask.png
	Returns:
		M: One-dimensional mask for training dataset. Dimension (\sum (X_i, Y_i), 1)
	"""
	M = np.array([])
	initialM = 0
	for d in dirs:
		m = misc.imread(d+"mask.png")
		m = np.reshape(m, m.shape[0]*m.shape[1])
		
		if initialM:
			M = np.concatenate((M,m))
		else:
			M = m
			initialM = 1

	return M


def flatten_testImages(d):
	"""
	This function flattens images in a single directory (test directories)
	Parameters:
		d: Path to directory containing images
	Return values:
		V: flattened matrix
		s: shape of image in the directory 
	"""
	V = np.array([])
	initialV = 0
	for i in range(100):
		new_image = misc.imread(d+"frame00"+str(i).zfill(2)+".png", flatten="true", mode = "L")
		s = new_image.shape
#		print(new_image.shape)
		img = np.reshape(new_image, new_image.shape[0]*new_image.shape[1])
		
		if initialV:
			V = np.vstack((V, img))
		else:
			V = img
			initialV = 1
	V = np.transpose(V)

	return V,s



def main(args):
	"""
	This function includes the entire flow of algorithm
	Parameters (args):
		trainset, testset, output directories 
	"""

	# Input
	trainpath = args.trainset
	traindirs = glob(trainpath+"*/")
	testpath = args.testset
	testdirs = glob(testpath+"*/")

	# Training
	print("Training:\n\n")
	T = flatten_matrices(traindirs)
	M = flatten_masks(traindirs)
	# print("T.shape: ",T.shape)
	# print("M.shape: ",M.shape)

	print("Scaler transform of train images")
	trainscaler = MinMaxScaler(feature_range=(0,10)).fit(T)		# normalising pixel values
	T = trainscaler.transform(T)

	print("SVC fit in process")
	stime = time.time()
	clf = svm.SVC(decision_function_shape = 'ovo', cache_size = 5000)
	clf.fit(T,M)
	etime = time.time()
	print("SVM fit done", "time taken:", etime-stime)

	if not os.path.exists(args.output):	# create output directory if does not exists
		os.makedirs(args.output)

	print("\n\nPrediction:\n\n")
	for i in testdirs:
		TestI,s = flatten_matrices(testdirs)
		print("Scaler transform of test images")

		scaler = MinMaxScaler(feature_range=(0,10)).fit(TestI)	# normalising pixel values
		TestI = scaler.transform(TestI)

		print("prediction in process")
		stime = time.time()
		prediction = clf.predict(TestI)
		prediction = np.resize(prediction, (s[0], s[1])) # just testing these images for now
		etime = time.time()
		print("prediction of sample", i, " in time", etime-stime)

		oImg = image.fromarray(prediction)
		oImg.save('output/'+i[-65:-1]+'.png')	# saving output image

