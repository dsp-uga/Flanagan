import tensorflow as tf
import skimage.io as io
import numpy as np
import random
import argparse
import pandas as pd
from weighted_unet import UNet
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import os
import datetime
import glob

DEBUG = False
DIM = 128

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
    data = tf.reshape(data,[1,128,128,1])

    labl = tf.slice(ddat,[99,0,0],[1,128,128])
    labl = tf.reshape(labl,[1,128,128,1])

    return (data,labl,m,n)

def test_record_parser(record):
    keys_to_features = {
            'name': tf.FixedLenFeature([],tf.string),
            'ddat': tf.FixedLenFeature([],tf.string),
            'm'   : tf.FixedLenFeature([],tf.int64),
            'n'   : tf.FixedLenFeature([],tf.int64)}

    parsed = tf.parse_single_example(record, keys_to_features)

    m    = tf.cast(parsed['m'],tf.int64)
    n    = tf.cast(parsed['n'],tf.int64)

    ddat_shape = tf.stack([100,m,n])

    ddat = tf.decode_raw(parsed['ddat'],tf.float32)
    ddat = tf.reshape(ddat,ddat_shape)

    return (parsed['name'],ddat,m,n)
    
class Model:
    def __init__(self,chk_path):
        self.chk_path = chk_path
        self.name     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            original_umask = os.umask(0)
            os.makedirs(chk_path, mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)
            
    def train(self,train_record,validation_record,csv_path,epochs=3000,lr=.0001,opt="RMSProp",weight=1):
        try:
            stats_df = pd.read_csv(csv_path)
        except:
            stats_df = pd.DataFrame(columns=["model_name","epochs","dice","loss","precision","recall"])

        sess = tf.Session()

        train_dataset = tf.data.TFRecordDataset([train_record])
        train_dataset = train_dataset.map(record_parser)
        
        validation_dataset = tf.data.TFRecordDataset([validation_record])
        validation_dataset = validation_dataset.map(record_parser)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        next_element = iterator.get_next()

        train_init_op = iterator.make_initializer(train_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        model = UNet(128, is_training=True,k=1)
        train_op = model.train(lr, opt, weight)

        tf_writer = tf.summary.FileWriter(logdir='./')
        config = tf.ConfigProto()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        avg_dice_score = 0.0
        avg_loss =0.0
        best_dice_score = 0.0
        best_loss = 0.0
        best_precision = 0.0
        best_recall = 0.0
        sum_dice_score = 0.0
        sum_loss = 0.0
        sum_precision = 0.0
        iters = 0 if (len(stats_df.index)==1) else len(stats_df.index)
        count = 0
        print("starting training epochs...")
        for i in range(epochs):
            sess.run(train_init_op)
            while True:
                try:
                    val = sess.run(next_element)
                    data,labl,m,n = val
                    _, loss_np, gs_np, dice_acc_np, summary_np ,flat_output_mask = sess.run(
                        [train_op, model.loss, model.gs, model.dice_acc, model.merged_summary,model.flat_output_mask],
                        feed_dict={model.input: data, model.gt_mask: labl})
                    recall          = 0
                    precision       = 0
                    sum_dice_score  = sum_dice_score + dice_acc_np
                    sum_loss        = sum_loss + loss_np
                    sum_recall      = 0
                    sum_precision   = sum_precision + precision
                    ratio_predicted = np.count_nonzero(np.round(flat_output_mask)/flat_output_mask.shape[0]/(DIM*DIM))
                    ratio_actual = np.count_nonzero(np.round(labl)/labl[0]/(DIM*DIM))
                    mismatch = float(ratio_predicted/ratio_actual)
                    count = count + 1
                except tf.errors.OutOfRangeError:
                    break
            sess.run(validation_init_op)
            while True:
                try:
                    val = sess.run(next_element)
                    data,labl,m,n = val
                    _, loss_np, gs_np, dice_acc_np, summary_np ,flat_output_mask = sess.run(
                        [train_op, model.loss, model.gs, model.dice_acc, model.merged_summary,model.flat_output_mask],
                        feed_dict={model.input: data, model.gt_mask: labl})
                    recall          = 0
                    precision       = 0
                    sum_dice_score  = sum_dice_score + dice_acc_np
                    sum_loss        = sum_loss + loss_np
                    sum_recall      = 0
                    sum_precision   = sum_precision + precision
                    ratio_predicted = np.count_nonzero(np.round(flat_output_mask)/flat_output_mask.shape[0]/(DIM*DIM))
                    ratio_actual    = np.count_nonzero(np.round(labl)/labl[0]/(DIM*DIM))
                    mismatch        = float(ratio_predicted/ratio_actual)
                    count           = count + 1
                except tf.errors.OutOfRangeError:
                    break
                if i%100==0:
                    print("collecting epoch {} statistics".format(i))
                    stats_df = stats_df.append({
                        "model_name": self.name,
                        "epochs": i,
                        "dice": sum_dice_score/count,
                        "loss": sum_loss/count,
                        "precision": sum_recall/count,
                        "recall": sum_precision/count},ignore_index=True)
                    iters += 1
                    epoch_chk_path = self.chk_path + "epoch" + str(i) + ".ckpt"
                    saver.save(sess,epoch_chk_path)
                    stats_df.to_csv("{}ModelStats.csv".format(csv_path), index=False)
        chk_path = self.chk_path + "epoch" + str(i) + ".ckpt"
        saver.save(sess, chk_path)
        
    def predict(self,test_record,chkpnt_name,save_path):

        try:
            original_umask = os.umask(0)
            os.makedirs(save_path,mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)

        try:
            original_umask = os.umask(0)
            os.makedirs("{}Predictions/".format(save_path),mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)
        
        tf.reset_default_graph()
        sess = tf.Session()
        
        tf_writer = tf.summary.FileWriter(logdir='./')
    
        test_dataset = tf.data.TFRecordDataset([test_record])
        test_dataset = test_dataset.map(test_record_parser)

        iterator = test_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        model_test_predict = UNet(128, is_training=True,k=1)
        saver = tf.train.import_meta_graph("{}{}.ckpt.meta".format(self.chk_path,chkpnt_name))
        saver.restore(sess, tf.train.latest_checkpoint(self.chk_path))
        init = tf.global_variables_initializer()
        sess.run(init)
        while True:
            try:
                predict = model_test_predict.predict();
                val = sess.run(next_element)
                name,data,m,n = val
                data = data.reshape(100,m,n,1)
                print("predicting for {}".format(name))
                predict = sess.run([predict],feed_dict={model_test_predict.input:data})
                predict = np.array(predict).reshape(100,m,n)
                predict = np.sum(predict,axis=0)/100
                predict = np.round(predict)
                np.save("{}Predictions/{}.npy".format(save_path,name),predict)
                with open("{}prediction_chunks.txt".format(save_path),'a') as f:
                    f.write("{}\n".format(name))
            except tf.errors.OutOfRangeError:
                break

    def stitch_predictions(self,input_file,data_path,output_path):

        try:
            original_umask = os.umask(0)
            os.mkdir(output_path,mode=0o0777)
        except FileExistsError:
            pass
        finally:
            os.umask(original_umask)
            
        try:
            fil = open(input_file)
        except:
            Print("Input file not found.")
            system.exit(0)
            
        directories = [x.strip() for x in fil.readlines()]
    
        for directory in directories:
            print("Stitching data for {}...".format(directory))
            iterrator = glob.iglob("{}{}*".format(data_path,directory))
            maxX = 0
            maxY = 0
            results = {}
            i=0
            for fil in iterrator:
                realfil = fil[fil.find("\'")+1:fil.find("_")]
                result = {}
                m = fil.find('_')
                n = fil.rfind("\'.npy")
                coors = [int(x) for x in fil[m+1:n].split('_')]
                if DEBUG :
                    print("realfil: {}".format(realfil))
                    print("n: {}".format(n))
                    print("m: {}".format(n))
                    print("fil[m:n]: {}".format(fil[m:n]))
                    print("coors: {}".format(coors))
                xx  = coors[1]
                x = coors[0]
                yy  = coors[3]
                y = coors[2]
                if xx > maxX : maxX = xx
                if yy > maxY : maxY = yy
                result['x']=x
                result['y']=y
                result['xx']=xx
                result['yy']=yy
                result['data'] = np.load(fil)
                results[i]=result
                i+=1
            if DEBUG :print("maxX: {},maxY: {}".format(maxX,maxY))
            full = np.zeros((maxX,maxY))
            hits = np.zeros((maxX,maxY))
            for i in results.keys():
                res = results[i]
                if DEBUG :
                    print("res['x']: {}".format(res['x']))
                    print("res['y']: {}".format(res['y']))
                    print("res['xx']: {}".format(res['xx']))
                    print("res['yy']: {}".format(res['yy']))                
                for r in range(128):
                    for s in range(128):
                        if DEBUG :
                            print("R: {}, S: {}".format(r,s))
                            print("hits.shape : {}, full.shape: {}, res['data'].shape : {}".format(hits.shape , full.shape, res['data'].shape))
                        hits[r+res['x'],s+res['y']] +=1
                        full[r+res['x'],s+res['y']] += res['data'][(r,s)]
            prediction = full / hits
            np.save("{}{}.npy".format(output_path,realfil),prediction)
