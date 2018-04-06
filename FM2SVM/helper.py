
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
