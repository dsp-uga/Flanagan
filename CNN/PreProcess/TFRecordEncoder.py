import numpy as np
from skimage import io
from skimage.io import ImageCollection 
import tensorflow as tf
import argparse
from sklearn.model_selection import ShuffleSplit
import os

DEBUG = False

class TestTFRecordEncoder:
    def __init__(self,in_file,dat_path,out_path):
        self.in_file  = in_file
        self.dat_path = dat_path
        self.out_path = out_path

    def writeTestTFRecord(self):
        try:
            original_umask = os.umask(0)
            os.makedirs(self.out_path, mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)
            
        with open(self.in_file) as f:
            files = [x.strip() for x in f.readlines()]
        writer = tf.python_io.TFRecordWriter(self.out_path + "test.tfrecord")
        for fil in files:
            print("******************************processing testing files...*************************************")
            test_data_path = "{}{}.npy".format(self.dat_path, fil)     
            data = np.load(test_data_path).astype(np.float32) 
            k,m_train,n_train = data.shape
            
            print("Converting numpy arrays to raw strings")
            data_raw = data.tostring()
            name = bytes(fil,'utf-8')

            print("Writing raw strings to files.")
            train_example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'name' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
                        'ddat' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw])),
                        'm'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[m_train])),
                        'n'    : tf.train.Feature(int64_list=tf.train.Int64List(value=[n_train]))
                    }
                )
            )
            writer.write(train_example.SerializeToString())
        writer.close()

class TrainTFRecordEncoder:
    def __init__(self,in_file,dat_path,lab_path,out_path):
        self.in_file = in_file
        self.dat_path = dat_path
        self.lab_path = lab_path
        self.out_path = out_path

    def writeTrainTFRecord(self):
        with open(self.in_file) as f:
            files = [x.strip() for x in f.readlines()]

        train_writer = tf.python_io.TFRecordWriter(self.out_path + "train.tfrecord")
        test_writer  = tf.python_io.TFRecordWriter(self.out_path + "validation.tfrecord")
        
        for train_indices,test_indices in ShuffleSplit(n_splits=1,test_size=.33).split(files):        
            print("******************************processing train files...*************************************")
            for train_index in train_indices:
                train_file = files[train_index]
                train_data_path = "{}{}/*".format(self.dat_path,train_file)
                train_labl_path = "{}{}.png".format(self.lab_path, train_file)
                train_data = np.array(ImageCollection(train_data_path)).astype(np.float32)
                train_labl = io.imread(train_labl_path).astype(np.float32)
                m_train,n_train = train_labl.shape
                ddat_train = np.concatenate([train_data,train_labl.reshape(1,m_train,n_train)])
                print("Converting numpy arrays to raw strings..")
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
            print("done.")
            train_writer.close()
            print("******************************processing validation files...*************************************")
            for test_index in test_indices:
                test_file = files[test_index]
                test_data_path = "{}{}/*".format(self.dat_path,test_file)
                test_labl_path = "{}{}.png".format(self.lab_path, test_file)
                test_data = np.array(ImageCollection(test_data_path)).astype(np.float32)
                test_labl = io.imread(test_labl_path).astype(np.float32)
                m_test,n_test = test_labl.shape
                ddat_test = np.concatenate([test_data,test_labl.reshape(1,m_test,n_test)])
                print("Converting numpy arrays to raw strings..")
                test_ddat_raw  = ddat_test.tostring()
                print("Writing raw strings to files.")
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
            print("done.")
            test_writer.close()
