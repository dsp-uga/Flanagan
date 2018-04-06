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
import helper

import time

def fm2svm(args):
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
	T = helper.flatten_matrices(traindirs)
	M = helper.flatten_masks(traindirs)
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
		TestI,s = helper.flatten_matrices(testdirs)
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

