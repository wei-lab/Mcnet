from dataprocess.dataprocess import *
from MultipathToRGB.MultipathToRGB import *
import random
import os
import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

MODEL_PATH='../ckpt'

BATCH_SIZE=1

LOG_DIR='../logs'

TRAIN_TFRECORDS_DIR='../data/train.trefcords'

TEST_TFRECORDS_DIR='../data/test.trefcords'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="McNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

    parser.add_argument("--train-data", type=str, default=TRAIN_TFRECORDS_DIR,
                        help="Path to the training trefcords file.")


    parser.add_argument("--test-data", type=str, default=TEST_TFRECORDS_DIR,
                        help="Path to the test trefcords file.")

    parser.add_argument("--mode-path", type=str, default=MODEL_PATH,
                        help="Where restore model parameters from.")

    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="save training log path.")

  

    return parser.parse_args()

def run_train():

    args = get_arguments()


    accuracyfile=open(os.path.join(args.log_dir,'accuracy.txt'),'a')
    lossfile=open(os.path.join(args.log_dir,'loss.txt'),'a')

    trianimage,tianlables=image_decode_from_tfrecords(TFRECORDS_DIR)

    batch_train_image,batch_tain_lables=get_batch_image(trianimage,tianlables,20)

    cnn_net=MultipathToRGB()

    inference=cnn_net.networkConstruction(batch_train_image)

    loss=cnn_net.sorfmax_loss(inference,batch_tain_lables)

    opti=cnn_net.optimer(loss)

    test_images,test_lables=image_decode_from_tfrecords(TEST_TFRECORDS_DIR)

    batch_test_image,batch_test_lable= get_test_batch( test_images,test_lables,batch_size=BATCH_SIZE,crop_size=128)

    test_inf=cnn_net.testNetwork(batch_test_image)

    correct_prediction=tf.equal(tf.cast(tf.argmax(test_inf,1),tf.int32), batch_test_lable)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init=tf.initialize_all_variables()

    with tf.Session() as session:

        session.run(init)
        
        coord=tf.train.Coordinator()

        threads=tf.train.start_queue_runners(coord=coord)

        max_iter=1000000

        iter=0
        print "start training ..."
        if os.path.exists(os.path.join(MODEL_PATH,'model.ckpt')) is True:
            tf.train.Saver(max_to_keep=None).restore(session, os.path.join(MODEL_PATH,'model.ckpt'))

        while iter<max_iter:
            loss_np,_,lable_np,image_np,inf_np=session.run([loss,opti,batch_tain_lables,batch_train_image,inference])

            if iter%10==0:
                print 'trainloss:',loss_np
                lossfile.write(str(iter)+':'+str(loss_np)+"\n")
            if iter%100==0:

                accuracy_np=session.run([accuracy])
                print '***************test accruacy:',accuracy_np,'*******************'
                accuracyfile.write(str(iter)+":"+str(accuracy_np[0])+'\n')
                tf.train.Saver(max_to_keep=None).save(session, os.path.join(MODEL_PATH,'model.ckpt'))
            iter+=1


        coord.request_stop()
        coord.join(threads)
        accuracyfile.close()
        lossfile.close()
run_train()


#generateTraindataORTestData()










