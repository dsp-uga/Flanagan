from .Train import Model
from .PreProcess.ImageCollectionConverter import ImageCollectionConverter
from .PreProcess.TFRecordEncoder import TestTFRecordEncoder, TrainTFRecordEncoder
from .PreProcess.TestSplitter import TestSplitter
from . import run_module
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description = 'Cilia Segmentation',
        argument_default = argparse.SUPPRESS
    )
    options = parser.add_subparsers()

    # print information
    op = options.add_parser('info', description = 'print system information')
    op.set_defaults(func = info)

    # Optional args
    parser.add_argument("--test_infile", default="CNN/Data/test.txt",
        help = "The path to the file which contains the names of the test datasets.")
    parser.add_argument("-test_data_path", default = "CNN/Data/Test/", type = int,
        help = "The path to the test data.")
    parser.add_argument("--chunk_size", default = 128, type=int,
        help = "The size to chunk the test data into. NOTE: Any size larger than 128 will break.")
    parser.add_argument("--test_out_path", default = "CNN/Data/",
        help = "The path to send the output of the test_splitter.\n\
                NOTE: running the test splitter will create one new file: \n\
                <test_out_path>/chunks.txt\n\
                and one new directory:\n\
                <test_out_path>/Chunks/\n\
                which will contain all of the chunks of data from the test input file.")
    
    parser.add_argument("--chunk_record_infile", default = "Data/chunks.txt",
        help = "The path to find the input file containing the names of all of the chunked test data for the TestTFRecordEncoder.")
    parser.add_argument("--chunk_data_path", default = "Data/Test/",
        help = "The path to find the chunked up test data for the TestTFRecordEncoder.")

    
    parser.add_argument("--train_record_infile", default = "Data/train.txt",
        help = "The path to find the input file containing the names of all of the chunked train data for the TestTFRecordEncoder.")
    parser.add_argument("--train_data_path", default = "Data/Train/",
        help = "The path to find the chunked up train data for the TrainTFRecordEncoder.")
    parser.add_argument("--masks_data_path", default = "Data/Masks/",
        help = "The path to find the chunked up train data for the TrainTFRecordEncoder.")
    
    parser.add_argument("--record_output_path", default = "Data/TFRecords/",
        help = "The path to store the records output by the TFRecordEncoders.")

    parser.add_argument("--ck_path", default = "Data/Ckpts/",
        help = "The path to store the checkpoints from training the model in.")
    parser.add_argument("--train_rec", default = "Data/TFRecords/test.tfrecord",
        help = "The path to the training TFRecord file for training the model.")
    parser.add_argument("--validation_rec", default = "Data/TFRecords/validation.tfrecord",
        help = "The path to the validation TFRecord file for training the model.")
    parser.add_argument("--eopchs", default = 1, type=int,
        help = "The number of epochs to train the model for.")
    parser.add_argument("--csv_path", default = "Data/",
        help = "The path to use for saving the csv file for preserving model statistics.")
    parser.add_argument("--learning_rate", default = .0001,
        help = "The learning rate for training the model.")
    
    parser.add_argument("--test_rec", default = "Data/TFRecords/test.tfrecord",
        help = "The path to the test TFRecord file for making predictions.")
    parser.add_argument("--chkpnt_name", default = "epoch0",
        help = "The check point from which you want to reload the model in order to make predictions.\
                Note: This must refer to a valid checkpoint that was stored, which are stored once every 100 epochs.")
    parser.add_argument("--prediction_save_path", default = "Data/",
        help = "The path to use for saving the predictions.\
                Note: this will create a new file:\
                <prediction_save_path>prediction_chunks.txt.\
                with the name of the file containing the predictions for each chunk of the predictions, and a new directory:\
                <prediction_save_path>Predictions/ \
                where those prediction chunks are saved.")

    parser.add_argument("--stitch_input_file", default = "Data/prediction_chunks.txt",
        help = "The file containing all of the names of the prediction chunks. This file is generated automatically by the call to model.predict().")
    parser.add_argument("--stitch_data_path", default = "Data/Predictions/",
        help = "The path to the stored prediction files for each chunk of test data.")
    parser.add_argument("--stitch_output_path", default = "Data/FinalPreds",
        help = "The path to store the final predictions at.")
        
    parser.set_defaults(func = run_module.main)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()

    if __name__ == '__main__':
        main()
