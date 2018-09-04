
# coding: utf-8

# In[ ]:

from dependency import *
from resnet18.resnet_model import Model
import tensorflow as tf
from tf_utils import tf_train, tf_test_error_rate, set_model_flags
import matplotlib.pyplot as plt
from data_utils import dataset
import os
from load_data import load_images
from foolbox.models import TensorFlowModel
import random



# In[ ]:

# padding
def padding_layer_iyswim(inputs, shape, name=None):### inputs.shape = [_, 85, 85, 3]; shape = [11,11,96]
    h_start = shape[0] # 11
    w_start = shape[1] # 11
    output_short = shape[2]# 96
    input_shape = tf.shape(inputs) # [_, 85, 85, 3]
    input_short = tf.reduce_min(input_shape[1:3]) # 3
    input_long = tf.reduce_max(input_shape[1:3]) # 85
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short))) # 1.0*96.0*85.0/3.0=2720
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +        tf.to_int32(input_shape[1] < input_shape[2]) * output_short # 2720
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +        tf.to_int32(input_shape[1] < input_shape[2]) * output_long # 96
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)


# In[ ]:

def adv_model_resnet():
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('target'):
            images = tf.placeholder(tf.float32, (None, 64, 64, 3))

            # preprocessing
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
            features = images - tf.constant(_CHANNEL_MEANS)
            """print(features.get_shape())

            resize_max = 65
            resize_min = 63
            resize_shape = np.random.randint(resize_min, resize_max) # random resize shape
            img_resize_tensor = [resize_shape]*2
            shape_tensor = np.array([np.random.randint(0, resize_max - resize_shape), 
                                    np.random.randint(0, resize_max - resize_shape), 
                                    resize_max])

            img_resize = tf.image.resize_images(features, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            padded_input = padding_layer_iyswim(img_resize, shape_tensor)
            padded_input.set_shape((img_resize.get_shape()[0], resize_max, resize_max, 3))"""

            model = Model(
                resnet_size=18,
                bottleneck=False,
                num_classes=200,
                num_filters=64,
                kernel_size=3,
                conv_stride=1,
                first_pool_size=0,
                first_pool_stride=2,
                second_pool_size=7,
                second_pool_stride=1,
                block_sizes=[2, 2, 2, 2],
                block_strides=[1, 2, 2, 2],
                final_size=512,
                version=2,
                data_format=None)

            logits = model(features, False)

            with tf.variable_scope('utilities'):
                saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target'))

    return graph, saver, images, logits


# In[ ]:

def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)
    #print("Predictions:", np.argmax(predictions, 1))
    #print("Labels:", np.argmax(labels, 1))

    # return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / len(predictions))





# In[ ]:

#set_model_flags(False)

graph, saver, images, logits = adv_model_resnet()



sess = tf.Session(graph = graph)
#sess.run(tf.global_variables_initializer())
#model.tf_load(sess, "./resnet18/checkpoints/model/")

path = os.path.join('resnet18', 'checkpoints', 'model')
saver.restore(sess, tf.train.latest_checkpoint(path))

data = dataset('../Defense_Model/tiny-imagenet-200/', normalize = False)
batch_size = 256
x_test, y_test = data.next_test_batch(batch_size)

with sess.as_default():
    model = TensorFlowModel(images, logits, bounds=(0, 255))
    y_logits = model.batch_predictions(x_test)
    #y_prob=np.softmax(y_logits)
    y_pred=np.argmax(y_logits, axis=1)
    #print(y_pred)
    y_label=np.argmax(y_test, axis=1)
    print(np.sum(np.equal(y_pred, y_label))/x_test.shape[0])
    
    


# In[ ]:

"""from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=tf.train.latest_checkpoint(path), tensor_name='', all_tensors=True)"""


# In[ ]:



