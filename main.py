import argparse
import FM2SVM.FM2SVM

parser = argparse.ArgumentParser(description='CSCI 8360 - Project 4, Team Flanagan')

parser.add_argument("-m", "--module", default = "FM2SVM",
					help = "Choose which module to run for cilia segentation")

parser.add_argument("-d", "--trainset", default = "data/train/smalldata/",
					help = "Path to folder containing a training data samples")

parser.add_argument("-t", "--testset", default = "data/test/smalldata/",
					help = "Path to folder containing a training data samples")

parser.add_argument("-o", "--output", default = "output",
					help = "Path to directory where output will be written")


args = parser.parse_args()
FM2SVM.FM2SVM.main(args)
