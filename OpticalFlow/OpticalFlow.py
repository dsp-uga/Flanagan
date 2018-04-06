"""
Resource for optical flow: https://docs.opencv.org/3.3.1/db/d7f/tutorial_js_lucas_kanade.html
"""

import cv2
import os
import numpy as np
from glob import glob
from PIL import Image


def OFlow(testdirs, opath):
	"""
	Implements optical flow using OpenCV's `calcOpticalFlowFarneback`
	Optical flow of each test sample is calculated
	Mask for each sample is found based on scaling of pixel distribution
	Parameters:
		testdirs: list of test sample directories
		opath: path to directory where output is to be stored
	"""

	# create directories if doesn't exist
	if not os.path.exists(opath):
		os.makedirs(opath)

	if not os.path.exists(opath+'/each_image'):
		os.makedirs(opath+'/each_image')
		
	if not os.path.exists(opath+'/masks'):
		os.makedirs(opath+'/masks')
		
	if not os.path.exists(opath+'/scaled_masks'):
		os.makedirs(opath+'/scaled_masks')

	dcount = 0	# directory count

	for d in testdirs:

		h = d[-65:-1]	# hash

		# create directory for each hash to save individual optical flow images separately
		if not os.path.exists(opath+'/each_image/'+h):
			os.makedirs(opath+'/each_image/'+h)

		prvs = cv2.imread(d + 'frame0000.png', 0)	# previous image

		s = (prvs.shape[0], prvs.shape[1], 3)	# hsv image shape

		hsv = np.zeros(s, np.uint8)
		hsv[...,1] = 255
		
		ms = (prvs.shape[0], prvs.shape[1])		# mask shape
		
		mask = np.zeros(ms, np.uint8)
		sum_mask = np.zeros(ms, np.uint8)
		scaled_mask = np.zeros(ms, np.uint8)
		
		print("dir: ",dcount," dim: ",sum_mask.shape)
		dcount += 1
		
		flag = 0

		for i in range(1,100):
			nxt = cv2.imread(d + 'frame00'+str(i).zfill(2)+'.png', 0)	# next image
			flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])	# calculates magnitude and angles of 2D vectors
			
			hsv[...,0] = ang*180/np.pi/2
			
			hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
			bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
			omg = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
	#		print(omg.dtype)
	#		print(omg.shape)
			
			cv2.imshow('frame1',omg)
			om = Image.fromarray(omg)
			om.save(opath+'/each_image/'+h+'/frame00'+str(i).zfill(2)+'.png', 0)
			
			omg2 = omg

			# scaling with random hardcoded values
			for r in range(omg2.shape[0]):
				for c in range(omg2.shape[1]):
					if omg2[r][c] < 32:
						sum_mask[r][c] += 0
					elif omg2[r][c] < 128:
						sum_mask[r][c] += 1
					else:
						sum_mask[r][c] += 2

			flag = cv2.waitKey(30) & 0xff
			if flag == 27:		# press ESC to exit
			    break

			prvs = nxt

		# generating mask based on sum_mask again based on random hardcoded values
		for r in range(prvs.shape[0]):
			for c in range(prvs.shape[1]):
				if sum_mask[r][c] > 50:
					mask[r][c] = 2
					scaled_mask[r][c] = 255
				elif sum_mask[r][c] > 15:
					mask[r][c] = 1
					scaled_mask[r][c] = 128
		omask = Image.fromarray(mask)
		omask.save(opath+'/masks/'+h+'.png', 0)
		osmask = Image.fromarray(scaled_mask)
		osmask.save(opath+'/scaled_masks/'+h+'.png', 0)

	cv2.destroyAllWindows()
