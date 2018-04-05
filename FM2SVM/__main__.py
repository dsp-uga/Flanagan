import argparse
import FM2SVM


def main():
	parser = argparse.ArgumentParser(description='CSCI 8360 - Project 4, Team Flanagan, module FM2SVM\
														implements flatten metrics + SVM')

	parser.add_argument("-d", "--trainset", default = "data/train/smalldata/",
						help = "Path to folder containing a training data samples")

	parser.add_argument("-t", "--testset", default = "data/test/smalldata/",
						help = "Path to folder containing a training data samples")

	parser.add_argument("-o", "--output", default = "output",
						help = "Path to directory where output will be written")


	args = parser.parse_args()

	FM2SVM.fm2svm(args)

if __name__ == '__main__':
	main()
