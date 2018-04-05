#################################################################################
# This class is used to split the test data for project 4 into chunks of        #
# a suitable  size in order to input them into a convolutional neural network.  #
# The data originally is heterogenous in shape (different width and heights for #
# each image in the dataset. This class takes those disparates sizes and chunks #
# them up into common sizes then saves the chunks into appropriately named      #
# files for future retrieval.                                                   #
#################################################################################

import random
import numpy as np
import argparse
from skimage import io as io
import os
import sys

DEBUG = True

def chunk_simple_fits(data,fil,top1,top2,chunk_size,chunk_save_path,out_path):
    for i in range(int(top1)):
        for j in range(int(top2)):
            r1,r2 = (chunk_size*i,chunk_size*(i+1))
            s1,s2 = (chunk_size*j,chunk_size*(j+1))
            ddata = data[:,r1:r2,s1:s2]
            path = "{}{}_{}_{}_{}_{}.npy".format(chunk_save_path,fil,r1,r2,s1,s2)
            with open("{}chunks.txt".format(out_path),'a') as f:
                f.write("{}_{}_{}_{}_{}\n".format(fil,r1,r2,s1,s2))
            np.save(path,ddata)
                
def chunk_uncovered_axis_one(data,fil,top1,top2,shape,chunk_size,chunk_save_path,out_path,uncovered_2):
    for i in range(int(top1)):
        r1,r2 = (chunk_size*i,chunk_size*(i+1))
        ddata = data[:,r1:r2,uncovered_2:]
        path = "{}{}_{}_{}_{}_{}.npy".format(chunk_save_path,fil,r1,r2,uncovered_2,shape[1])
        with open("{}chunks.txt".format(out_path),'a') as f:
            f.write("{}_{}_{}_{}_{}\n".format(fil,r1,r2,uncovered_2,shape[1]))
        np.save(path,ddata)
        
def chunk_uncovered_axis_two(data,fil,top1,top2,shape,chunk_size,chunk_save_path,out_path,uncovered_1):     
    for j in range(int(top2)):
        s1,s2 = (chunk_size*j,chunk_size*(j+1))
        ddata = data[:,uncovered_1:,s1:s2]
        path = "{}{}_{}_{}_{}_{}.npy".format(chunk_save_path,fil,uncovered_1,shape[2],s1,s2)
        with open("{}chunks.txt".format(out_path),'a') as f:
            f.write("{}_{}_{}_{}_{}\n".format(fil,uncovered_1,shape[1],s1,s2,shape[2]))
        np.save(path,ddata)

def chunk_uncovered_corner(data,fil,top1,top2,shape,chunk_size,chunk_save_path,out_path,uncovered_1,uncovered_2):
    ddata = data[:,uncovered_1:,uncovered_2:]
    path = "{}{}_{}_{}_{}_{}.npy".format(chunk_save_path,fil,uncovered_1,shape[0],uncovered_2,shape[1])
    with open("{}chunks.txt".format(out_path),'a') as f:
        f.write("{}_{}_{}_{}_{}\n".format(fil,uncovered_1,shape[0],uncovered_2,shape[1]))
    np.save(path,ddata)
  

class TestSplitter:
    def __init__(self,in_file,dat_path,chunk_size,out_path):
        self.in_file         = in_file
        self.dat_path        = dat_path
        self.chunk_size      = chunk_size
        self.out_path        = out_path
        self.chunk_save_path = "{}Chunks/".format(out_path)

            
    def chunk(self):
        try:
            original_umask = os.umask(0)
            os.makedirs(self.chunk_save_path, mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)

        with open(self.in_file) as f:
            files = [x.strip() for x in f.readlines()]
            print("Processing data ...")
            for fil in files:
                print("...")
    
                data_path = self.dat_path + fil + "/*"
                data = np.array(io.ImageCollection(data_path)).astype(np.float32)
                shape = data.shape

                if len(shape) == 1 :
                    print("data not found: {}".format(data_path))
                    continue
                
                top1 = np.floor(shape[1]/self.chunk_size)
                top2 = np.floor(shape[2]/self.chunk_size)
                
                chunk_simple_fits(data,fil,top1,top2,self.chunk_size,self.chunk_save_path,self.out_path)
                
                uncovered_1 = shape[1]-self.chunk_size
                uncovered_2 = shape[2]-self.chunk_size
                
                if uncovered_2               > 0 : chunk_uncovered_axis_one(data,fil,top1,top2,shape,self.chunk_size,self.chunk_save_path,self.out_path,uncovered_2)
                if uncovered_1               > 0 : chunk_uncovered_axis_two(data,fil,top1,top2,shape,self.chunk_size,self.chunk_save_path,self.out_path,uncovered_1)
                if uncovered_1 + uncovered_2 > 0 : chunk_uncovered_corner(data,fil,top1,top2,shape,self.chunk_size,self.chunk_save_path,self.out_path,uncovered_1,uncovered_2)
                
    
 
        
        
