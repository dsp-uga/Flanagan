from CNN.Train import Model
from CNN.PreProcess.ImageCollectionConverter import ImageCollectionConverter
from CNN.PreProcess.TFRecordEncoder import TestTFRecordEncoder, TrainTFRecordEncoder
from CNN.PreProcess.TestSplitter import TestSplitter
from . import run_module
import sys
import argparse

def main():

    splitter = TestSplitter(test_infile,test_data_path,chunk_size,test_out_path)
    splitter.chunk()

    testEnc = TestTFRecordEncoder(test_record_infile,test_data_path,record_output_path)
    testEnc.writeTestTFRecord()

    trainEnc = TrainTFRecordEncoder(train_record_infile,train_data_path,masks_data_path,record_output,path)
    trainEnc.writeTrainTFRecord()

    model = Model(ck_path)

    model.train(train_rec,validation_rec,csv_path,epochs,learning_rate)

    model.predict(test_rec,chkpnt_name,prediction_save_path)

    model.stitch_predictions(stitch_input_file,stitch_data_path,stitch_output_path)
