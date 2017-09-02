#coding=utf-8

import csv

from MultipathToRGBHSV import *
from cn.ynu.dataprocess.dataprocess import *
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf



def test_accuracy():
    #f=open("/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/predict/predict.txt",'w')
    f=file("HSVtestparameters.csv",'wb')
    writer = csv.writer(f)

    test_images,test_lables=image_decode_from_tfrecords('/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/traintestdata/test.tfrecords')

    batch_test_image,batch_test_lable= get_test_batch( test_images,test_lables,batch_size=212,crop_size=128)

    cnn_net=MultipathToRGBHSV()

    test_inf=cnn_net.testNetwork(batch_test_image)
    p=tf.nn.softmax(test_inf)

    predict_label=tf.cast(tf.argmax(test_inf,1),tf.int32)

    correct_prediction=tf.equal(predict_label, batch_test_lable)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init=tf.initialize_all_variables()

    with tf.Session() as session:

        session.run(init)

        coord=tf.train.Coordinator()

        threads=tf.train.start_queue_runners(coord=coord)



        #if os.path.exists(os.path.join("/home/zhanghao/PycharmProjects/CNN-fish-classfication/cn/ynu/MutipathToHSV",'model.ckpt')) is True:
        tf.train.Saver(max_to_keep=None).restore(session, "./model.ckpt")
        test_labels, test_inf,p,predict_label,accuracy_np,correct_prediction=session.run([batch_test_lable,test_inf,p,predict_label,accuracy,correct_prediction])

        writer.writerows(zip(p,predict_label,test_labels))
        f.close()


        print '***************test accruacy:',accuracy_np,'*******************'
            # print 'ROC',caculateROC(correct_prediction,test_labels)
            #accuracyfile.write(str(iter)+":"+str(accuracy_np[0])+'\n')
        # k= map(lambda item:str(item).split('/')[-1].decode("utf-8"),name)
        #
        # for i in zip(k,predict_label):
        #    f.write(str(i)+"\n")
        # f.close()
        coord.request_stop()
        coord.join(threads)
       # accuracyfile.close()

#generatepredictdata("/media/zhanghao/000ED952000587F3/2017/论文/CNN/test/大湾17.4.6/")
#test_accuracy()

def show_feature():

    test_images,test_lables=image_decode_from_tfrecords('/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/traintestdata/test.tfrecords')

    batch_test_image,batch_test_lable= get_test_batch( test_images,test_lables,batch_size=1,crop_size=128)

    cnn_net=MultipathToRGBHSV()

    hsv,rgb=cnn_net.testNetwork(batch_test_image)

    init=tf.initialize_all_variables()

    with tf.Session() as session:

        session.run(init)

        coord=tf.train.Coordinator()

        threads=tf.train.start_queue_runners(coord=coord)

        tf.train.Saver(max_to_keep=None).restore(session, "./model.ckpt")
        hsv,rgb=session.run([ hsv,rgb])
        #print '***************test accruacy:',test_inf,'*******************'
        plt.figure(num='astronaut',figsize=(1,8))  #创建一个名为astronaut的窗口,并设置大小

        for i in range(40,48):
            #print np.array(hsv[0][1]).shape
            plt.subplot(1,8,i-40+1)
            plt.axis('off')
            plt.imshow(np.array(hsv[0][i]),plt.cm.gray)
        plt.show()


        print hsv
        print '***********************************************************'
        print rgb
        coord.request_stop()
        coord.join(threads)

show_feature()


