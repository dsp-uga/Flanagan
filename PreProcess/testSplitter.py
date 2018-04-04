# Imports for the library

import random
import numpy as np
import argparse
from skimage import io as io

if __name__ == "__main__":
    ############################################################################
    # parameterizing model and defining file locations using command line args
    ############################################################################
    parser = argparse.ArgumentParser(description = "Data Maker",
        epilog = "Use this program to split the testing data for p4 into chunks.",
        add_help = "How to use",
        prog = "python testSplitter.py -i <path_to_input_names> -d <path_to_data> -n <size_of_chunks> -o <path_for_output>" )

    parser.add_argument("-i", "--infile", required=True,
        help = "The path to the file with the names of the input datasets.")
        
    parser.add_argument("-d", "--data", required=True,
        help = "The path to find the data at.")

    parser.add_argument("-n", "--chunks", required=True, type=int,
        help = "The size of the chunks to chunk the images into.")

    parser.add_argument("-o", "--outpath", required=True,
        help = "The path to use for output.")
    
    args = vars(parser.parse_args())
    
    in_file    = args['infile']
    dat_path   = args['data']
    chunk_size = args['chunks']
    out_path   = args['outpath']

    with open(in_file) as f:
        files = [x.strip() for x in f.readlines()]
    print("Processing data ...")
    for fil in files:
        print("...")
        data_path = dat_path + fil + "/*"
        print(data_path)
        data = np.array(io.ImageCollection(data_path)).astype(np.float32)
        shape = data.shape
        if len(shape) == 1 :
            print("data not found")
            continue
        top1 = np.floor(shape[1]/chunk_size)
        top2 = np.floor(shape[2]/chunk_size)

        for i,j in range(top1),range(top2):
            r1,r2 = (n*i,n*(i+1))
            s1,s2 = (n*j,n*(j+1))
            ddata = data[:,r1:r2,s1:s2]
            path = "{}{}_{}_{}_{}_{}.npy".format(out_path,fil,r1,r2,s1,s2)
            np.save(path,ddata)

        uncovered_1 = shape[1]-n
        uncovered_2 = shape[2]-n

        #cover the uncovered right edges
        if uncovered_2 > 0:
            for i in range(top1):
                r1,r2 = (n*i,n*(i+1))
                ddata = data[:,r1:r2,uncovered_2:]
                path = "{}{}_{}_{}_{}_{}.npy".format(out_path,fil,r1,r2,uncovered_2,shape[2])
                np.save(path,ddata)
        #cover the uncovered bottom edges
        if uncovered_1 > 0:
            for j in range(top1):
                s1,s2 = (n*j,n*(j+1))
                ddata = data[:,uncovered_1:,s1:s2]
                path = "{}{}_{}_{}_{}_{}.npy".format(out_path,fil,uncovered_1,shape[1],s1,s2)
                np.save(path,ddata)
        #cover the bottom right corner
        if unovered_1 + uncovered_2 > 0:
            ddata = data[:,uncovered_1:,uncovered_2:]
            path = "{}{}_{}_{}_{}_{}.npy".format(out_path,fil,uncovered_1,shape[1],s1,s2)
            np.save(path,ddata)
