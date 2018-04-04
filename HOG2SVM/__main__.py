"""
Main file for using HOG of data preprocessing and SVM of classification
"""

import sys
import argparse
import cv2
import skimage
from . import hog2svm

def info():
    """
    System information
    """
    print('Python version: ', sys.version)
    print('Skimage version: ', skimage.__version__)
    print('OpenCV version: ', cv2.__version__)

def main():
    parser = argparse.ArgumentParser(
        description = 'Cilia Segmentation',
        argument_default = argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # print information
    op = options.add_parser('info', description = 'print system information')
    op.set_defaults(func = info)

    # Optional args
    parser.add_argument("-N", default = 211, type = int,
        help = "Specify training amount of videos in training model")
    parser.add_argument("-M", default = 114, type = int,
        help = "Specify testing amount of masks to predict in testing set")
    parser.add_argument("--visualized_hash", default = "0",
        help = "Hash in training set to visualize")
    parser.add_argument("--figure", choices = ["hog", "pred"], default = "pred",
        help = "Choose the visualization of hog figures or prediction figures")
    parser.add_argument("--show", type = bool, default = False,
        help = "Option of showing the figures [Default: False]")
    parser.set_defaults(func = hog2svm.main)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

    train_txt_file = open("train.txt", "r").read()
    test_txt_file = open("test.txt", "r").read()
    train_list = train_txt_file.split('\n')[0:-1]
    test_list = test_txt_file.split('\n')[0:-1]

if __name__ == '__main__':
    main()
