#coding=utf-8
import os
import tensorflow as tf
def eachFile(file_root, resfile,labels):
    f=open(resfile,'w')
    parhDir=os.listdir(file_root )

    for allDir in parhDir:
         child = os.path.join('%s/%s' % (file_root, allDir))
         print child

         f.write(child+","+str(labels)+"\n")
    f.close()
import numpy as np
def unpickle(files):
    import cPickle
    res=[]
    label=[]
    for file in files:
        with open(file,'rb')as fo:
            dic=cPickle.load(fo)
        d=dic['data']
        label.extend(dic['labels'])
        c= np.reshape(d,(10000,3,32,32))

        for i in c:
            temp=np.array(i).transpose()
            res.append(temp)
    return res,label

def image_encode_to_tfrecords(img_matrix,label,data_root='',new_name="cifartest.tfrecords"):
    writer=tf.python_io.TFRecordWriter(data_root+'/'+new_name)
    count=0
    for i in zip(img_matrix,label):
        height,width,nchannel=i[0].shape
        count +=1
        l=int(i[1])

        example=tf.train.Example(features=tf.train.Features(feature={
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[i[0].tobytes()])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
            }))

        serialized=example.SerializeToString()
        writer.write(serialized)
    print'count:',count
    writer.close()

def image_decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)

    reader=tf.TFRecordReader()

    _,serialized=reader.read(filename_queue)

    example=tf.parse_single_example(serialized,features={
        #'name':tf.FixedLenFeature([],tf.string),
        'height':tf.FixedLenFeature([],tf.int64),
        'width':tf.FixedLenFeature([],tf.int64),
        'nchannel':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64)
    })
   # name=tf.cast(example['name'],tf.string)
    label=tf.cast(example['label'], tf.int32)

    image=tf.decode_raw(example['image'],tf.uint8)

    image=tf.reshape(image,tf.pack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))

    return image,label

def get_batch_image(image, label, batch_size):

    distorted_image = tf.random_crop(image, [128, 128, 3])
    distorted_image = tf.image.random_flip_up_down(distorted_image)#

    distorted_image = tf.image.random_flip_up_down(distorted_image)

    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,capacity=10000,
                                                min_after_dequeue=1000)
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])

import cv2
def test():

    image,label=image_decode_from_tfrecords('..//data/traintestdata/cifartrain.tfrecords')

    batch_image,batch_label=get_batch_image(image,label,3)#batch 生成测试

    init=tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for l in range(2):#每run一次，就会指向下一个样本，一直循环
            image_np,label_np=session.run([image,label])#每调用run一次，那么
            print image_np
            #cv2.imshow("temp",image_np)
            #print label_np
           # cv2.waitKey()
            #print label_np
            #print image_np.shape


            # batch_image_np,batch_label_np=session.run([batch_image,batch_label])
            # print batch_image_np.shape
            # print batch_label_np.shape



        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)
#test()
d=unpickle(['/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/cifar-10/test_batch'])
#
image_encode_to_tfrecords(d[0],d[1],data_root='.')


#eachFile('/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/Testimg_3N',
 #        '/home/zhanghao/PycharmProjects/CNN-fish-classfication/data/datapath/Testimg_3N.txt',1)


