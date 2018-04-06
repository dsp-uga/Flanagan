from CNN.Train.Model import Model
from CNN.PreProcess.ImageCollectionConverter import ImageCollectionConverter
from CNN.PreProcess.TFRecordEncoder import TestTFRecordEncoder, TrainTFRecordEncoder
from CNN.PreProcess.TestSplitter import TestSplitter


def main():

    print("Running brief test...")
    splitter = TestSplitter(test_in_file,test_data_path,chunk_size,test_out_path)
    splitter.chunk()

    testEnc = TestTFRecordEncoder(test_record_infile,test_data_path,record_output_path)
    testEnc.writeTestTFRecord()

    trainEnc = TrainTFRecordEncoder(train_record_infile,train_data_path,masks_data_path,record_output_path)
    trainEnc.writeTrainTFRecord()
    
    model = Model(ck_path)
    model.train(train_rec,validation_rec,eopchs,csv_path,learning_rate)
    model.predict(test_rec,chkpnt_name,prediction_save_path)
    model.stitch_predictions(stitch_input_file,stitch_data_path,stitch_output_path)
    
