from dependency import *

from keras.models import save_model
from mnist import *
#from tf_utils import tf_train, tf_test_error_rate, set_model_flags
from tf_utils_wopadding import tf_train, tf_test_error_rate, set_model_flags
from model_utils import get_data_mean
from data_utils import dataset
from attack_utils import gen_grad
from fgs import symbolic_fgs
import vgg16
import vgg16_v2
import vgg19
from resnet18.resnet_model import Model

def defense_model(x):

    with tf.variable_scope('target'):
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

def load_model(name, path):
    if(name == "vgg19"):
        model = vgg19.Vgg19(get_data_mean(), path, trainable=False, skippable = False)
    if(name == "vgg16"):
        model = vgg16.Vgg16(get_data_mean(), path, trainable=False, skippable = False)
    return model

def main(adv_model_names):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_model_flags()
    
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
    eps = FLAGS.EPS

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models
    # load from out source
    adv_models = [None] * len(adv_model_names)
    for i, name in enumerate(adv_model_names):
        adv_models[i] = load_model(name, path="./models/"+name+"-save.npy")
    x_advs = [None] * (len(adv_models))

    for i, m in enumerate(adv_models):
        logits, _ = m(x)
        grad = gen_grad(x, logits, y, loss='training')
        x_advs[i] = symbolic_fgs(x, grad, eps=eps)

    data = dataset(FLAGS.DIR, normalize = False)
    sess, graph_dict = tf_train(g, x, y, data, defense_model, train_mode, x_advs=x_advs)

    # Finally print the result!
    test_error = tf_test_error_rate(sess, graph_dict, data, x_advs)
    print('Test error: %.1f%%' % test_error)


if __name__ == '__main__':
    main(['vgg19', 'vgg16'])
