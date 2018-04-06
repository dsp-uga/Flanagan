from .Train.Model import Model
from .PreProcess.ImageCollectionConverter import ImageCollectionConverter
from .PreProcess.TFRecordEncoder import TestTFRecordEncoder, TrainTFRecordEncoder
from .PreProcess.TestSplitter import TestSplitter
from . import run_module
import sys
import argparse

def main(test_infile,test_data_path,chunk_size,test_out_path,
         chunk_record_infile,chunk_data_path,record_output_path,
         train_record_infile,train_data_path,masks_data_path,
         ck_path,
         train_rec,validation_rec,csv_path,epochs,learning_rate,
         test_rec,chkpnt_name,prediction_save_path,
         stitch_input_file,stitch_data_path,stitch_output_path):

    splitter = TestSplitter(test_infile,test_data_path,chunk_size,test_out_path)
    splitter.chunk()

    testEnc = TestTFRecordEncoder(chunk_record_infile,chunk_data_path,record_output_path)
    testEnc.writeTestTFRecord()

    trainEnc = TrainTFRecordEncoder(train_record_infile,train_data_path,masks_data_path,record_output_path)
    trainEnc.writeTrainTFRecord()

    model = Model(ck_path)

    model.train(train_rec,validation_rec,csv_path,epochs,learning_rate)

    model.predict(test_rec,chkpnt_name,prediction_save_path)

    model.stitch_predictions(stitch_input_file,stitch_data_path,stitch_output_path)
