import tensorflow as tf
from decorator import lazy_method
import numpy as np
from functools import reduce
from dependency import *
from tensorflow.contrib.layers import xavier_initializer


class Vgg16:
    """
    A trainable version VGG19.
    """

    def __init__(self, data_mean, vgg19_npy_path=None, trainable=3, skippable=3, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        self.name = "vgg16"
        self.data_mean = data_mean
        self.var_dict = {}
        # trainable
        if isinstance(trainable, bool):
            self.trainable = [trainable] * 16
        elif isinstance(trainable, int):
            self.trainable = [False] * (16-trainable) + [True] * (trainable)
        else:
            raise ValueError('trainable')

        # skippable
        if isinstance(skippable, bool):
            self.skippable = [skippable] * 16
        elif isinstance(skippable, int):
            self.skippable = [False] * (16-skippable) + [True] * (skippable)
        else:
            raise ValueError('skippable')
        
        self.dropout = dropout

    #@lazy_method("VGG19_MODEL")
    def __call__(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb #* 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1]
        assert green.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1]
        assert blue.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.data_mean[0],
            green - self.data_mean[1],
            red - self.data_mean[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", 0) # [batch_size, 64, 64, 64]
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", 1) # [batch_size, 64, 64, 64]
        self.pool1 = self.max_pool(self.conv1_2, 'pool1') # [batch_size, (64-2)/2+1, (64-2)/2+1, 64]

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", 2) # [batch_size, 32, 32, 128]
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", 3) # [batch_size, 32, 32, 128]
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')  # [batch_size, (32-2)/2+1, (32-2)/2+1, 128]

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", 4) # [batch_size, 16, 16, 256]
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", 5) # [batch_size, 16, 16, 256]
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", 6) # [batch_size, 16, 16, 256]
        #self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", 7) # [batch_size, 16, 16, 256]
        self.pool3 = self.max_pool(self.conv3_3, 'pool3') # [batch_size, (16-2)/2+1, (16-2)/2+1, 256]


        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", 7) # [batch_size, 8, 8, 512]
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", 8) # [batch_size, 8, 8, 512]
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", 9) # [batch_size, 8, 8, 512]
        #self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", 11) # [batch_size, 8, 8, 512]
        self.pool4 = self.max_pool(self.conv4_3, 'pool4') # [batch_size, 4, 4, 512]

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", 10) # [batch_size, 4, 4, 512]
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", 11) # [batch_size, 4, 4, 512]
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", 12) # [batch_size, 4, 4, 512]
        #self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", 15) # [batch_size, 4, 4, 512]
        self.pool5 = self.max_pool(self.conv5_3, 'pool5') # [batch_size, 2, 2, 512]

        self.fc6 = self.fc_layer(self.pool5, 2048, 4096, "fc6", 13)  # 2048 = ((64 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if self.trainable[13]:
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        """if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable[16]:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)"""

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", 14)
        self.relu7 = tf.nn.relu(self.fc7)
        if self.trainable[14]:
            if train_mode is not None:
                self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        """if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable[17]:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)"""

        self.fc8 = self.fc_layer(self.relu7, 4096, 200, "fc8", 15 )

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

        return self.fc8, self.prob # logits and prob

    #@lazy_method("AVG_POOLING")
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #@lazy_method("MAX_POOLING")
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #@lazy_method("CONVOLUTION")
    def conv_layer(self, bottom, in_channels, out_channels, layer_name, layer_idx):
        with tf.variable_scope(layer_name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, layer_name, layer_idx)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    #@lazy_method("FULLY_CONN")
    def fc_layer(self, bottom, in_size, out_size, layer_name, layer_idx):
        with tf.variable_scope(layer_name):
            weights, biases = self.get_fc_var(in_size, out_size, layer_name, layer_idx)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    

    def get_conv_var(self, filter_size, in_channels, out_channels, layer_name, layer_idx):
        #initial_value = xavier_initializer(uniform = False)
        initial_value = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
        filters = self.get_var(initial_value, [filter_size, filter_size, in_channels, out_channels], 
                               layer_name, 0, "filters", layer_idx)

        initial_value = tf.zeros_initializer()
        biases = self.get_var(initial_value, [out_channels], layer_name, 1, "biases", layer_idx)

        return filters, biases

    def get_fc_var(self, in_size, out_size, layer_name, layer_idx):
        initial_value = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
        weights = self.get_var(initial_value, [in_size, out_size], layer_name, 0, "weights", layer_idx)

        initial_value = tf.zeros_initializer()
        biases = self.get_var(initial_value, [out_size], layer_name, 1, "biases", layer_idx)

        return weights, biases

    def get_var(self, initial_value, shape, layer_name, idx, var_name, layer_idx):
        trainable = self.trainable[layer_idx]
        skippable = self.skippable[layer_idx]
        
        if self.data_dict is not None and not skippable:
            value = self.data_dict[layer_name][idx]
            shape = None # If initializer is a constant, do not specify shape.
        else:
            value = initial_value

        if trainable:
            var = tf.get_variable(name=var_name, shape=shape, dtype=tf.float32, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(layer_name, idx)] = var

        return var

    def save_npy(self, sess, npy_path):
        assert isinstance(sess, tf.InteractiveSession)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("Model file saved at ", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

    
