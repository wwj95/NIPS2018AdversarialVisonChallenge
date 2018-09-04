from dependency import *
from attack_utils import gen_adv_loss

import time
import sys
import  os
import matplotlib 
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data_utils import dataset

def set_model_flags(adv=True):
    if adv:
        flags.DEFINE_string('ADV_MODELS', '', 'Path to adv model(s)')
        flags.DEFINE_float('EPS', 0.3, 'FGS attack scale')
        
    flags.DEFINE_string('f', '', 'ipynb kernel') # for ipynb

    flags.DEFINE_string('TYPE', 'vgg19', 'Type of model')   
    #flags.DEFINE_string('MODEL', './models/vgg19-save.npy', 'Path to model')
    #flags.DEFINE_string('TRAIN_DIR', '../Defense_Model/tiny-imagenet-200/train', 'Training data dir')
    flags.DEFINE_string('DIR', '../Defense_Model/tiny-imagenet-200', 'Data dir')
    flags.DEFINE_integer('EVAL_FREQUENCY', 1, 'Frequency of evaluation')
    flags.DEFINE_string('DATA_MEAN_FILE', 'data_mean.txt', 'File name of data mean')
    flags.DEFINE_integer('NUM_EPOCHS', 1, 'Number of epochs')
    flags.DEFINE_integer('BATCH_SIZE', 1 , 'Size of training batches')
    flags.DEFINE_integer('NUM_CLASSES', 200, 'Number of classification classes')
    flags.DEFINE_integer('IMAGE_ROWS', 64, 'Input row dimension')
    flags.DEFINE_integer('IMAGE_COLS', 64, 'Input column dimension')
    flags.DEFINE_integer('NUM_CHANNELS', 3, 'Input depth dimension')
    flags.DEFINE_integer('IMAGE_RESIZE', 96, 'Resize of image size')
    flags.DEFINE_integer('AUG_RATIO', 10, 'Augmentation Ratio')
    flags.DEFINE_float('REG_SCALE', 0.005, 'The scale of regularization')


def augmentX(Xbatch, multi):
    tiled_Xbatch = augmentTensor(Xbatch, multi, "TILED_INPUTS")
    noises = tf.random_normal(tf.shape(tiled_Xbatch), 
                              mean=0.0,
                              stddev=0.1,
                              dtype=tf.float32,
                              name="NOISE",
                              ###seed=1234
                             )
    return tf.add(tiled_Xbatch, noises)

def augmentY(YBatch, multi):
    return augmentTensor(YBatch, multi, name="TILED_LABELS")

def augmentTensor(tensor, multi, name="TILED_TENSOR"):
    tile_shape = [multi] + [1] * (len(tensor.get_shape())-1)###tile_shape = [10,1,1,1]
    return  tf.tile(tensor, tile_shape, name=name)#####multiple the tensor tile_shape times


def tf_train(graph, x, y, data, model_fn, train_mode=None, x_advs=None):
    old_vars = set(tf.all_variables())

    if x_advs is not None: # ensemblem adversarial training
        #img_resize_tensor = tf.placeholder(tf.int32, [2])
        #shape_tensor = tf.placeholder(tf.int32, [3])
        idx = tf.placeholder(tf.int32)
        x_adv = tf.stack(x_advs)[idx]
        # padding and resize, 2nd place NIPS 2017
        #x_resize = tf.image.resize_images(x, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #x_adv_resize = tf.image.resize_images(x_adv, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #padded_input = padding_layer_iyswim(x_resize, shape_tensor)
        #padded_input.set_shape((FLAGS.BATCH_SIZE, FLAGS.IMAGE_RESIZE, FLAGS.IMAGE_RESIZE, 3))

        #padded_adv_input = padding_layer_iyswim(x_adv_resize, shape_tensor)
        #padded_adv_input.set_shape((FLAGS.BATCH_SIZE, FLAGS.IMAGE_RESIZE, FLAGS.IMAGE_RESIZE, 3))


        # Augmentation is only applied on the clean image
        def aug_1():
            print ("Augmentated")
            return augmentX(x, FLAGS.AUG_RATIO), x_adv, augmentY(y, FLAGS.AUG_RATIO)

        def aug_2():
            return x, x_adv, y
        #if in the test mode, do not need to do augmentation, so apply aug_2()
        if train_mode is None:
            aug_X = x
            X_adv = x_adv
            aug_Y = y
        else:
            aug_X, X_adv, aug_Y = tf.cond(tf.equal(train_mode, True), aug_1, aug_2)

        inputs = tf.concat([aug_X, X_adv], 0)
        labels = tf.concat([aug_Y, y], 0)

    else:
        inputs = x
        labels = y

    # Generate cross-entropy loss for training
    if x_advs is not None:
        logits, model = model_fn(inputs)
    else:
        logits, model = model_fn(inputs,train_mode)
    preds = tf.nn.softmax(logits)
    loss = gen_adv_loss(logits, labels, mean=True)
    
    loss *= 16
    if FLAGS.REG_SCALE is not None:
        # if you want regularization
        regularize = tf.contrib.layers.l2_regularizer(FLAGS.REG_SCALE)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print tf.GraphKeys.TRAINABLE_VARIABLES
        reg_term = sum([regularize(param) for param in params])
        loss += reg_term
    
    learning_rate = 1e-5
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(loss)
    #"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config, graph = graph)
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_variables(set(tf.all_variables()) - old_vars))
    if model.name == 'resnet' :
        model.tf_load(sess, "./resnet18/checkpoints/model/") ##give only the directory not the file

    graph_dict = {}
   
    train_size = 1000000
    start_time = time.time()
    print('Initialized!')
   
    # Loop through training steps.
    num_steps = int(FLAGS.NUM_EPOCHS * train_size + FLAGS.BATCH_SIZE - 1) // FLAGS.BATCH_SIZE
    print('Number of Iteration: %.1f' % num_steps)
    print('Number of Epoches for Iteration: %.1f' % FLAGS.NUM_EPOCHS)
    print('Batch Size: %.1f' % FLAGS.BATCH_SIZE)

    step = 0
    while step < 1:
        batch_data, batch_labels = data.next_train_batch(FLAGS.BATCH_SIZE)
        #batch_data, batch_labels = data.load_n_images_all_classes(2)       
        fetches = [optimizer, loss, preds, labels, logits]
        #if x_advs is None:
        feed_dict = {
                    x: batch_data,
                    y: batch_labels,
                    train_mode: True
            }
        """else:
            resize_shape = np.random.randint(85, 96) # random resize shape
            feed_dict = {
                        x: batch_data,
                        y: batch_labels,
                        train_mode: True,
                        img_resize_tensor: [resize_shape]*2,
                        shape_tensor: np.array(
                                                [np.random.randint(0, FLAGS.IMAGE_RESIZE - resize_shape), 
                                                np.random.randint(0, FLAGS.IMAGE_RESIZE - resize_shape), 
                                                FLAGS.IMAGE_RESIZE])"""
            #}
        

        # choose source of adversarial examples at random
        # (for ensemble adversarial training)
        if x_advs is not None:
            feed_dict[idx] = np.random.randint(len(x_advs))
            graph_dict["idx"] = idx
            #graph_dict["img_resize_tensor"] = img_resize_tensor
            #graph_dict["shape_tensor"] = shape_tensor
            #graph_dict["resize_shape"] = resize_shape

        # Run the graph
        _, curr_loss, curr_preds, curr_labels, curr_logits = \
            sess.run(fetches=fetches, feed_dict=feed_dict)
        
        if step % FLAGS.EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.2f s' %
                (step, float(step) * FLAGS.BATCH_SIZE / train_size,
                elapsed_time))
            print ('Step : %d ' % step)
            print('Minibatch loss: %.3f' % curr_loss)

            print('Minibatch error: %.1f%%' % error_rate(curr_preds, curr_labels))

            sys.stdout.flush()

        step += 1
    if model.name == 'resnet' :
        model.tf_save(sess, "./models/res_check0901/resnet18.ckpt-0901")
    else:
        model.save_npy(sess, npy_path="./models/"+model.name+"-save.npy")

    
    graph_dict["x"] =  x
    graph_dict["y"] =  y
    graph_dict["inputs"] = inputs
    graph_dict["labels"] = labels
    graph_dict["train_mode"] = train_mode
    graph_dict["logits"] = logits
    graph_dict["preds"] = preds
    graph_dict["optimizer"] = optimizer
    graph_dict["loss"] = loss

    
    # ERROR: NameError: name 'pickle_dict' is not defined
    """pickle_dict = { "step_list": step_list,
                    "loss_list": loss_list,
                    "l1_list": l1_list,
                    "l2_list": l2_list,
                    "err_list": err_list
                    }"""
    
    

    return sess, graph_dict

def tf_test_error_rate(sess, graph_dict, data, x_advs=None):
    """
    Compute test error.
    """
    total_itr = 1
    itr = 0
    err = 0
    while itr < total_itr:
        batch_data, batch_labels = data.next_test_batch(FLAGS.BATCH_SIZE)

        fetches = [graph_dict["preds"], graph_dict["labels"]]
        if x_advs is None:
            feed_dict = {
                        graph_dict["x"]: batch_data,
                        graph_dict["y"]: batch_labels,
                        graph_dict["train_mode"]: False
                }
        else:
            #resize_shape = np.random.randint(85, 96) # random resize shape
            feed_dict = {
                        graph_dict["x"]: batch_data,
                        graph_dict["y"]: batch_labels,
                        graph_dict["train_mode"]: False,
                        graph_dict["idx"]: np.random.randint(len(x_advs))
                        }  
        """graph_dict["img_resize_tensor"]: [resize_shape]*2,
        graph_dict["shape_tensor"]: np.array(
                                [np.random.randint(0, FLAGS.IMAGE_RESIZE - resize_shape), 
                                np.random.randint(0, FLAGS.IMAGE_RESIZE - resize_shape), 
                                FLAGS.IMAGE_RESIZE])"""

        # Run the graph
        test_preds, test_labels = sess.run(fetches=fetches, feed_dict=feed_dict)
        err += error_rate(test_preds, test_labels)
        itr += 1
    return err/total_itr



def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)
    #print("Predictions:", np.argmax(predictions, 1))
    #print("Labels:", np.argmax(labels, 1))

    # return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / len(predictions))



# padding
def padding_layer_iyswim(inputs, shape, name=None):### inputs.shape = [_, 85, 85, 3]; shape = [11,11,96]
    h_start = shape[0]#11
    w_start = shape[1]#11
    output_short = shape[2]#96
    input_shape = tf.shape(inputs)##[_, 85, 85, 3]
    input_short = tf.reduce_min(input_shape[1:3])#3
    input_long = tf.reduce_max(input_shape[1:3])#85
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))#1.0*96*85/3
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short##
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long##96
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)

