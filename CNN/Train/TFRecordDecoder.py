import tensorflow as tf
import skimage.io as io
import numpy as np
import random
import argparse

DIM = 128
epochs = 1

def record_parser(record):
    keys_to_features = {
            'ddat': tf.FixedLenFeature([],tf.string),
            'm'   : tf.FixedLenFeature([],tf.int64),
            'n'   : tf.FixedLenFeature([],tf.int64)} 

    parsed = tf.parse_single_example(record, keys_to_features)

    m    = tf.cast(parsed['m'],tf.int64)
    n    = tf.cast(parsed['n'],tf.int64)

    ddat_shape = tf.stack([101,m,n])
    
    ddat = tf.decode_raw(parsed['ddat'],tf.float32)
    ddat = tf.reshape(ddat,ddat_shape)
    ddat = tf.random_crop(ddat,[101,128,128])
    
    k  = np.random.randint(100)

    data = tf.slice(ddat,[k,0,0],[1,128,128])
    data = tf.reshape(data,[128,128])

    labl = tf.slice(ddat,[99,0,0],[1,128,128])
    labl = tf.reshape(labl,[128,128])
    
    return (data,labl,m,n)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Data Maker",
        epilog = "Use this program to randomly split the training data for p4 into random sub-samples.",
        add_help = "How to use",
        prog = "python randomSampler.py -t <path_to_training_TFRecordFile> -v <path_to_validation_training_TFRecordFile>" )

    parser.add_argument("-t", "--tid", required=True,
        help = "The path to find the .tfrecord filenfor the testing data")
    parser.add_argument("-v", "--vid", required=True,
        help = "The path to find the .tfrecord file for the validation data")
    
    args = vars(parser.parse_args())
    
    validation_filename = args['vid']
    train_filename      = args['tid']

    sess = tf.Session()
    
    train_dataset = tf.data.TFRecordDataset([train_filename])
    train_dataset = train_dataset.map(record_parser)

    validation_dataset = tf.data.TFRecordDataset([validation_filename])
    validation_dataset = validation_dataset.map(record_parser)

    #iterator = dataset.make_initializable_iterator()        
    #next_element = iterator.get_next()

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    
    for i in range(epochs):
        sess.run(train_init_op)
        print("****************************TRAIN**********************************")
        while True:
            try:
                val = sess.run(next_element)
                data,labl,m,n = val
                print("data.shape: ",data.shape)
                print("labl.shape: ",labl.shape)
                print("M: ",m)
                print("N: ",n)
            except tf.errors.OutOfRangeError:
                break
        sess.run(validation_init_op)
        print("****************************VALID**********************************")
        while True:
            try:
                val = sess.run(next_element)
                data,labl,m,n = val
                print("data.shape: ",data.shape)
                print("labl.shape: ",labl.shape)
                print("M: ",m)
                print("N: ",n)
            except tf.errors.OutOfRangeError:
                break
