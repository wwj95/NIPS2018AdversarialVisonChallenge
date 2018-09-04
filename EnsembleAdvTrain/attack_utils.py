from dependency import *
import numpy as np
import keras.backend as K

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """


    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = K.cast(K.equal(logits, K.max(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    elif loss == 'logloss':
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = K.mean(out)
    else:
        out = K.sum(out)
    return out

    
    #return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
    """softmax = tf.nn.softmax(logits)
    loss = - tf.reduce_sum(labels * tf.log(softmax))
    return loss"""
    


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad = K.gradients(adv_loss, [x])[0]
    return grad
