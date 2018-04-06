import argparse
import OpticalFlow
from glob import glob


def main():
	parser = argparse.ArgumentParser(description='CSCI 8360 - Project 4, Team Flanagan, module Optical Flow\
														implements optical flow in OpenCV')

	parser.add_argument("-t", "--testset", default = "data/test/smalldata/",
						help = "Path to folder containing a test data samples")

	parser.add_argument("-o", "--output", default = "output",
						help = "Path to directory where output will be written")


	args = parser.parse_args()

	testdatapath = args.testset
	testdirs = glob(testdatapath+'*/')

	opath = args.output
	OpticalFlow.OFlow(testdirs, opath)


if __name__ == '__main__':
	main()