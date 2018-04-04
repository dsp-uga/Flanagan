'''
This program converts a directory of images to a numpy array of images. 

There are two required parameters: 
  
    -i <SOME_INPUT_DIRECTORY> i.e. - the directory to find the image collection at 
    -n <COLLECTION_NAME>      i.e. - the name of the image collection

'''

from skimage import io
from skimage.io import collection

import argparse
import numpy as np


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Data Maker",
        epilog = "Use this program to convert a collection of images to a single 3D numpy array.",
        add_help = "How to use",
        prog = "python dataMaker.py -i <PATH_TO_DATA>")
    
    parser.add_argument("-i", "--input", required = True,
        help = "The path to the dataset to process.")

    parser.add_argument("-n", "--name", required = True,
        help = "The name to the dataset to process.")

    args     = vars(parser.parse_args())
    IN_PATH  = args['input'] + "/*"
    NAME     = args['name']
    OUT_PATH = NAME + ".npy"
    
    print("reading.")    
    imgs = io.imread_collection(IN_PATH)
    print("concatenating")
    imgs = collection.concatenate_images(imgs)
    print("reshaping")
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2])
    print("writing")
    np.save(OUT_PATH,imgs)
