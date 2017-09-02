#encoding=utf-8
import tensorflow as tf
import cv2

def image_encode_to_tfrecords(lables_file,data_root,new_name="test.tfrecords",resize=None):
    writer=tf.python_io.TFRecordWriter(data_root+'/'+new_name)

    print lables_file
    i=0
    for line in lables_file:
        i+=1
        l=line.split(',')

        print (l[0])
        image= cv2.imread(l[0])



        if resize is not None:

            image=cv2.resize(image,resize)

        height,width,nchannel=image.shape

        label=int(l[1])

        example=tf.train.Example(features=tf.train.Features(feature={
                'name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[l[0]])),
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

        serialized=example.SerializeToString()
        writer.write(serialized)
    print i
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

    return image,label #name

def get_batch_image(image, label, batch_size):

    distorted_image = tf.random_crop(image, [128, 128, 3])
    distorted_image = tf.image.random_flip_up_down(distorted_image)#

    distorted_image = tf.image.random_flip_up_down(distorted_image)

    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,capacity=10000,
                                                min_after_dequeue=1000)
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def get_test_batch(image, label, batch_size,crop_size):

    #distorted_image=tf.image.central_crop(image,39./45.)

    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])

    images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def get_predict_batch(image, label,name, batch_size,crop_size):

    #distorted_image=tf.image.central_crop(image,39./45.)

    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])

    images, label_batch,name_batch=tf.train.batch([distorted_image, label,name],batch_size=batch_size)

    return images, tf.reshape(label_batch, [batch_size]),tf.reshape(name_batch,[batch_size])


def test():
    #image_encode_to_tfrecords("/home/zhanghao/PycharmProjects/Rproblem/data/data.txt","/home/zhanghao/PycharmProjects/Rproblem/data",resize=(150,150))

    image,label=image_decode_from_tfrecords('/home/zhanghao/PycharmProjects/Rproblem/data/test.tfrecords')

    batch_image,batch_label=get_batch_image(image,label,3)#batch 生成测试

    init=tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for l in range(2):#每run一次，就会指向下一个样本，一直循环
            image_np,label_np=session.run([image,label])#每调用run一次，那么
            cv2.imshow("temp",image_np)
            #print label_np
            cv2.waitKey()
            #print label_np
            #print image_np.shape


            batch_image_np,batch_label_np=session.run([batch_image,batch_label])
            print batch_image_np.shape
            print batch_label_np.shape



        coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)
#test()