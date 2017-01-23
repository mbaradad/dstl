from itertools import permutations
import numpy as np
from keras.utils.np_utils import to_categorical
import keras.backend as K
import keras.objectives as objectives
import tensorflow as tf
from math import factorial

def iou_loss(y_true, y_pred):
  y_true = K.clip(y_true, 10e-6, 1.0 - 10e-6)
  y_pred = K.clip(y_pred, 10e-6, 1.0 - 10e-6)
  actual_cost = (K.sum(tf.mul(y_true, y_pred), axis=[-2,-1])) / (
    K.sum(K.sqrt(tf.mul(y_true, y_true)), axis=[-2,-1]) +
    K.sum(K.sqrt(tf.mul(y_pred, y_pred)), axis=[-2,-1]) - K.sum(tf.mul(y_true, y_pred),
                                                                           axis=[-2,-1]))
  return 1 - K.sum(actual_cost,axis=-1)/10

if __name__ == "__main__":

  #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.333)
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess = tf.Session()
  with sess.as_default():
    #size of masks (5, 50176)
    #a = np.random.rand(1,5, 50176)

    #a = K.tensor list(np.random.rand(5, 50176))

    y_true = np.zeros([1, 6, 50176])
    y_true[:, 0, 0:10000] = 1
    y_true[:, 1, 10000:20000] = 1
    y_true[:, 2, 20000:30000] = 1
    #y_true[:, 3, 30000:35000] = 1
    #y_true[:, 4, 35000:] = 1

    #TODO: check assignment using masks vs assignment using eof from prediction!!!
    perm1 = [2, 1, 0, 4, 3, 5]
    perm2 = [1, 0, 3, 4, 2, 5]

    y_true = y_true.astype("float32")

    y_pred_masks_value = [y_true[:, perm1, :], y_true[:, perm2, :]]
    y_true_masks_value = [y_true, y_true]



    # TODO: add shapes for debugging
    y_true_masks = tf.placeholder('float32', shape=[3, 6, 50176])
    y_pred_masks = tf.placeholder('float32', shape=[3, 6, 50176])
    cost_ins = [y_true_masks, y_pred_masks]
    cost_functions = [iou_loss(y_true_masks, y_pred_masks)]
    # TODO: initilize selectively the variables of the loss
    # tf.initialize_all_variables().run()

    sess = tf.get_default_session()
    print sess.run(cost_functions, feed_dict={cost_ins[0]: y_true_masks_value,
                                                    cost_ins[1]: y_pred_masks_value})


    # NOTICE: this results are hard to interpret, but they seem to match for the tests performed
    # for example, for the permutation 2, the last example perm2[2] = 3 won't be matched, since there are only 3 instances.
    #  It will assign the y_true[2] to the predicted mask[2], though the match would be with the predicted_mask[4],
    # but this is outside the groundtruth length (3)
    print 'a'