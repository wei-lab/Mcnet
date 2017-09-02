#encoding=utf-8
import tensorflow as tf


class MultipathToRGBHSV(object):
    def __init__(self):

        with tf.variable_scope("weights"):
            self.weights={
                "conv11":tf.get_variable("conv11",[7,7,3,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                "conv12":tf.get_variable("conv12",[3,3,3,32],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                "conv13":tf.get_variable("conv13",[7,7,3,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),

                "conv21":tf.get_variable("conv21",[5,5,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                "conv22":tf.get_variable("conv22",[2,2,32,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                "conv3":tf.get_variable("conv3",[2,2,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
               
                'fc1':tf.get_variable('fc1',[32*32*128,1024],initializer=tf.contrib.layers.xavier_initializer()),
                'fc2':tf.get_variable('fc2',[1024,2],initializer=tf.contrib.layers.xavier_initializer()),
            }

        with tf.variable_scope("biases"):
            self.biases={
                'conv11':tf.get_variable('conv11',[32,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv12':tf.get_variable('conv12',[32,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv13':tf.get_variable('conv13',[64,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

                'conv21':tf.get_variable('conv21',[64,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv22':tf.get_variable('conv22',[64,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3':tf.get_variable('conv3',[128,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
          
                'fc1':tf.get_variable('fc1',[1024,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2':tf.get_variable('fc2',[2,],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            }

    def networkConstruction(self,image):

        images=tf.reshape(image,[-1,128,128,3])
        rgbimages=(tf.cast(images,tf.float32)/255.-0.5)*2
        images=(tf.cast(images,tf.float32)/255)
        images=tf.image.rgb_to_hsv(tf.cast(images,tf.float32))
        conv11=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv11'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv11'])

        pool11=tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv12=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv12'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv12'])
        pool12=tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv21=tf.nn.bias_add(tf.nn.conv2d(pool11, self.weights['conv21'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv21'])

        conv22=tf.nn.bias_add(tf.nn.conv2d(pool12, self.weights['conv21'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv21'])

       # conv1andconv2=tf.cast(0,[conv21,conv22])
        conv13=tf.nn.bias_add(tf.nn.conv2d(rgbimages, self.weights['conv13'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv13'])
        pool13=tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        reluconv1andconv2= tf.add(conv21,conv22)

        conv1andconv2andconv3=tf.add(reluconv1andconv2,pool13)

        conv3=tf.nn.bias_add(tf.nn.conv2d(conv1andconv2andconv3, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv3'])

        pool3=tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        flatten=tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        dropfc1=tf.nn.dropout(flatten,0.5)

        fc1=tf.matmul(dropfc1, self.weights['fc1'])+self.biases['fc1']

        fc_relu1=tf.nn.relu(fc1)

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']


        return  fc2

    def testNetwork(self,images):

        images=tf.reshape(images,[-1,128,128,3])
        rgbimages=(tf.cast(images,tf.float32)/255.-0.5)*2
        images=(tf.cast(images,tf.float32)/255)
        images=tf.image.rgb_to_hsv(tf.cast(images,tf.float32))
        conv11=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv11'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv11'])

        pool11=tf.nn.max_pool(conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv12=tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv12'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv12'])
        pool12=tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv21=tf.nn.bias_add(tf.nn.conv2d(pool11, self.weights['conv21'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv21'])

        conv22=tf.nn.bias_add(tf.nn.conv2d(pool12, self.weights['conv21'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv21'])

       # conv1andconv2=tf.cast(0,[conv21,conv22])
        conv13=tf.nn.bias_add(tf.nn.conv2d(rgbimages, self.weights['conv13'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv13'])
        pool13=tf.nn.max_pool(conv13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        reluconv1andconv2= tf.add(conv21,conv22)

        conv1andconv2andconv3=tf.add(reluconv1andconv2,pool13)

        conv3=tf.nn.bias_add(tf.nn.conv2d(conv1andconv2andconv3, self.weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),
                             self.biases['conv3'])

        pool3=tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        flatten=tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

       #

        fc1=tf.matmul(flatten, self.weights['fc1'])+self.biases['fc1']

        fc_relu1=tf.nn.relu(fc1)

        fc2=tf.matmul(fc_relu1, self.weights['fc2'])+self.biases['fc2']
        hsvres=tf.cast(conv22,tf.float32)*255
        rgbres=(tf.cast(pool3,tf.float32)/2+1)*255
        return  hsvres,pool3
    def sorfmax_loss(self,predicts,labels):


        predict=tf.nn.softmax(predicts)

        label=tf.one_hot(labels,self.weights['fc2'].get_shape().as_list()[1])

        loss =-tf.reduce_mean(label * tf.log(predict))# tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost= loss
        return self.cost

    def optimer(self,loss,lr=0.0001):

        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer

