from dependency import *

from keras.models import save_model
from tf_utils import tf_train, tf_test_error_rate, set_model_flags
from model_utils import get_data_mean
import vgg16
import vgg16_v2
import vgg19
from data_utils import dataset
from resnet18.resnet_model import Model
import tiny_imagenet_loader

def adv_model_vgg19(x, train_mode):
    model = vgg19.Vgg19(get_data_mean(),'./models/vgg19-pre.npy')
    logits, _ = model(x, train_mode)
    return logits, model

def adv_model_vgg16(x, train_mode):
    model = vgg16.Vgg16(get_data_mean(), './models/vgg16-pre.npy')
    logits, _ = model(x, train_mode)
    return logits, model

def adv_model_resnet(x):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
    features = x - tf.constant(_CHANNEL_MEANS)

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

    logits = model(features, True)
    return logits, model

def adv_models(type):
    adv_dict = {}
    adv_dict ["vgg19"] = adv_model_vgg19
    adv_dict ["vgg16"] = adv_model_vgg16
    adv_dict ["resnet"] = adv_model_resnet
    return adv_dict[type]

def main():
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_model_flags(False)

    tf.reset_default_graph()
    g = tf.get_default_graph()

    x = tf.placeholder(tf.float32, 
                       shape=[None, 
                              FLAGS.IMAGE_ROWS, 
                              FLAGS.IMAGE_COLS, 
                              FLAGS.NUM_CHANNELS]
                      ) 
    y = tf.placeholder(tf.float32, 
                       shape=[None, FLAGS.NUM_CLASSES]
                      )
    train_mode = tf.placeholder(tf.bool)
    adv_model = adv_models(FLAGS.TYPE)
    
    ata = dataset('../Defense_Model/tiny-imagenet-200/', normalize = False)
    
    sess, graph_dict = tf_train(g, x, y, data, adv_model, train_mode)
    #tf_train returns the sess and graph_dict
    #tf_test_error_rate also need to run the sess and use the feed_dict in the tf_train
    #graph_dict is the dictiorary that contains all the items that is necessary on the graph

    # Finally print the result!
    test_error = tf_test_error_rate(sess, graph_dict, data)
    print('Test error: %.1f%%' % test_error)
    sess.close()
    del(g)

if __name__ == '__main__':
    main()
