import numpy as np
from skimage import io
from skimage.io import ImageCollection 
import tensorflow as tf
import argparse

np.random.seed(42)

#A function for parsing TFRecords
def record_parser(record):
    keys_to_features = {
            'fil' : tf.FixedLenFeature([],tf.string),
            'm'   : tf.FixedLenFeature([],tf.int64),
            'n'   : tf.FixedLenFeature([],tf.int64)} 

    parsed = tf.parse_single_example(record, keys_to_features)

    m    = tf.cast(parsed['m'],tf.int64)
    n    = tf.cast(parsed['n'],tf.int64)

    fil_shape = tf.stack([10,m,n])
     
    fil = tf.decode_raw(parsed['fil'],tf.float32)    
    fil = tf.reshape(fil,fil_shape)
    fil = tf.random_crop(fil,[10,5,5])
    print("cropped")
    k = np.random.randint(9)

    data = tf.slice(fil,[k,0,0],[1,5,5])
    print("sliced")
    data = tf.reshape(data,[5,5])
    mask = tf.slice(fil,[9,0,0],[1,5,5])
    print("sliced")
    mask = tf.reshape(mask,[5,5])

    return (data,mask,m,n)
    
#For writing and reading from the TFRecord
filename = "test.tfrecord"

if __name__ == "__main__":

    #Create the TFRecordWriter
    data_writer = tf.python_io.TFRecordWriter(filename)

    #Create some fake data
    files = []
    i_vals = np.random.randint(low=5,high=10,size=10)
    j_vals = np.random.randint(low=5,high=10,size=10)

    print(i_vals)
    print(j_vals)
    for x in range(5):
        dat = np.random.rand(10,i_vals[x],j_vals[x]).astype(np.float32)
        files.append(dat)
    i=0
    #Serialize the fake data and record it as a TFRecord using the TFRecordWriter
    for fil in files:
        i+=1
        f,m,n = fil.shape
        fil_raw = fil.tostring()
        print(fil.shape)
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'fil' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[fil_raw])),
                    'm'   : tf.train.Feature(int64_list=tf.train.Int64List(value=[m])),
                    'n'   : tf.train.Feature(int64_list=tf.train.Int64List(value=[n]))
                }
            )
        )
        data_writer.write(example.SerializeToString())
    data_writer.close()

    #Deserialize and report on the fake data
    sess = tf.Session()
    
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(record_parser)

    iterator = dataset.make_initializable_iterator()
        
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            val = sess.run(next_element)
            data,mask,m,n = val 
            print("data.shape: ",data.shape)
            print("mask.shape: ",mask.shape)
            print("M: ",m)
            print("N: ",n)
        except tf.errors.OutOfRangeError:
            break

