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

class ImageCollectionConverter:

    def flat_convert(self,in_path,out_path):
        print("reading.")    
        imgs = io.imread_collection(path)
        print("concatenating")
        imgs = collection.concatenate_images(imgs)
        print("reshaping")
        imgs = imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2])
        print("writing")
        np.save(out_path,imgs)

    def convert_to_3D_png(self,in_path,out_path):
        print("reading.")    
        imgs = io.imread_collection(in_path)
        print("concatenating")
        imgs = collection.concatenate_images(imgs)
        print("writing")
        np.save(out_path,imgs)
