import numpy as np
from skimage import io
from skimage.io import ImageCollection 
import tensorflow as tf
import argparse
from sklearn.model_selection import ShuffleSplit

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Data Maker",
        epilog = "Use this program to randomly split the training data for p4 into random sub-samples.",
        add_help = "How to use",
        prog = "python TFRecordEncoder.py -i <path_to_names> -l <path_to-labels> -d <path_to_data> -o <path_for_output>" )

    parser.add_argument("-i", "--id", required=True,
        help = "The path to find the file with the names of the input datasets.")

    parser.add_argument("-l", "--labl", required=True,
        help = "The path to find the label images at.")

    parser.add_argument("-d", "--data", required=True,
        help = "The path to find the data image sets at.")

    parser.add_argument("-o", "--output", required=True,
        help = "The path to write the output files to. ")

    parser.add_argument("-t", "--test", action="store_true"
        help = "The path to write the output files to. ")
    
    args = vars(parser.parse_args())

    name     = args['id']
    dat_path = args['data']
    lab_path = args['labl']
    out_path = args['output']
    test     = args['test']
    
    with open(name) as f:
        files = f.readlines()
        
    files = [x.strip() for x in files]


    ####################NEED TO SPLIT THE TESTING DATA INTO 128 by 128#################################
    if test:
        writer = tf.python_io.TFRecordWriter(out_path + "_test.tfrecord")
        for fil in files:
            test_data_path = dat_path + fil + ".npy"    
            data = np.load(test_data_path).astype(np.float32) 
            k,m_train,n_train = data.shape
            print("Converting numpy arrays to raw strings")
            data_raw = ddat_train.tostring()
            if (DEBUG) : 
                print("data.shape: ", data.shape)
                print("M_train: ",m_train,", N_train: ",n_train)
            print("Writing raw strings to files.")
            train_example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'ddat' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw])),
                        'm'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[m_train])),
                        'n'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[n_train]))
                    }
                )
            )
            writer.write(train_example.SerializeToString())
            writer.close()
    else:
        train_writer = tf.python_io.TFRecordWriter(out_path + "_train.tfrecord")
        test_writer  = tf.python_io.TFRecordWriter(out_path + "_validation.tfrecord")
        for train_indices,test_indices in ShuffleSplit(n_splits=1,test_size=.33).split(files):        
            print("******************************processing train files...*************************************")
            for train_index in train_indices:
                train_file = files[train_index]
                train_data_path = dat_path + train_file + "/*"    
                train_labl_path = lab_path + train_file  + ".png"
                train_data = np.array(ImageCollection(train_data_path)).astype(np.float32)
                train_labl = io.imread(train_labl_path).astype(np.float32)
                m_train,n_train = train_labl.shape
                ddat_train = np.concatenate([train_data,train_labl.reshape(1,m_train,n_train)])
                print("train_data.shape: ", train_data.shape)
                print("train_labl.shape: ", train_labl.shape)
                print("ddat_train.shape: ", ddat_train.shape)
                print("M_train: ",m_train,", N_train: ",n_train)
                print("Converting numpy arrays to raw strings")
                train_ddat_raw = ddat_train.tostring()
                print("Writing raw strings to files.")
                train_example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'ddat' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_ddat_raw])),
                            'm'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[m_train])),
                            'n'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[n_train]))
                        }
                    )
                )
                train_writer.write(train_example.SerializeToString())
            train_writer.close()
            print("******************************processing validation files...*************************************")
            for test_index in test_indices:
                test_file = files[test_index]
                test_data_path = dat_path + test_file + "/*"
                test_labl_path = lab_path + test_file + ".png"
                test_data = np.array(ImageCollection(test_data_path)).astype(np.float32)
                test_labl = io.imread(test_labl_path).astype(np.float32)
                m_test,n_test = test_labl.shape
                ddat_test = np.concatenate([test_data,test_labl.reshape(1,m_test,n_test)])
                print("test_data.shape: ", test_data.shape)
                print("test_labl.shape: ", test_labl.shape)
                print("ddat_test.shape: ", ddat_test.shape)
                print("M_test: ",m_test,", N_test: ",n_test)
                print("Converting numpy arrays to raw strings")
                test_ddat_raw  = ddat_test.tostring()
                test_example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'ddat' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[test_ddat_raw])),
                            'm'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[m_test])),
                            'n'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[n_test]))
                        }
                    )
                )
                test_writer.write(test_example.SerializeToString())
            test_writer.close()
        
