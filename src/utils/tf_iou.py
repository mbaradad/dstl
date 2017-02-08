from itertools import permutations
import numpy as np
from keras.utils.np_utils import to_categorical
import keras.backend as K
import keras.objectives as objectives
import tensorflow as tf
from math import factorial
from keras.objectives import binary_crossentropy

def iou_loss(y_true, y_pred):
  #add epsilon*224*224, to take into account the clipping of L1 y_pred in the denominator
  #when y_true is zero

  zero_masks_percentage_per_class = np.asarray([0.78762727, 0.71392209, 0.93249225, 0.52611775, 0.05146082,
                                     0.65692784, 0.97864099, 0.97211155, 0.99081452, 0.94876051])

  #used as a 'groundtruth l1' for smoothing
  area_occupied_per_class = np.asarray([3.51788670e-02, 9.12233361e-03, 7.56076172e-03, 3.02536476e-02,
                                        1.08549209e-01, 2.78849391e-01, 4.96184210e-03, 1.74715459e-03,
                                        4.48981383e-05, 1.90685129e-04])



  l1_pred = K.sum(K.abs(y_pred), axis=[-2,-1])
  l1_true = K.sum(K.abs(y_true), axis=[-2,-1])
  is_zero = tf.to_float(tf.equal(l1_true, 0))

  #also take into account zero_masks_percentage, to avoid 'easy wins' for hard classes
  #so that the 'wins' for empty masks equal the 'wins' for perfectly predicted masks = 1*percentage_of_non_zero
  #for perfect prediction, the mean of all actual costs should be 1

  #actual_cost = (area_occupied_per_class * (1 - zero_masks_percentage_per_class) * 224 * 224 * is_zero + K.sum(
  #  tf.mul(y_true, y_pred), axis=[-2, -1])) / (
  #                (area_occupied_per_class * 224 * 224 * is_zero) + l1_pred + l1_true - K.sum(tf.mul(y_true, y_pred),
  #                                                                                           axis=[-2, -1]))

  #aprox taking into account the receptive field of 32 of the resnet (though it should be 32*32/x
  smooth = [1,1,1,1,1,1,100,1,1,1]
  #TODO: scale 1 - actual cost to perfectly match the previous logic, and that the sum of all losses for perfect matching is 0
  iou = (smooth+ K.sum(tf.mul(y_true, y_pred), axis=[-2,-1])) / (
    smooth + l1_pred + l1_true - K.sum(tf.mul(y_true, y_pred), axis=[-2,-1]))

  #is_zero = tf.to_float(tf.equal(l1_true, 0))
  #not_is_zero = K.abs(1 - is_zero)

  #zero_no_important_factor = 10
  #weights = not_is_zero / (np.ones(10) - zero_masks_percentage_per_class)
  #for sample normalization of weights
  #weights = weights / K.transpose(K.reshape(K.tile(K.sum(weights, axis=1), 10), [10, 16])) * 10
  #for batch normalization of weights (take into account that some sampels may have a lot of non-zero masks while others not)
  #weights = weights / K.sum(weights) * 10*16

  #use iou when groundtruth not zero and l1 to zero when it is zero, and add proportional weights, to account for both
  #actual_cost = (1 - iou) #* not_is_zero + is_zero*K.sum(K.abs(y_pred), axis=[-2,-1])/224/224

  actual_cost = K.mean(K.binary_crossentropy(y_pred, y_true), axis=[-1,-2])
  actual_cost = tf.Print(actual_cost, [l1_true[0, :]], summarize=2000, message="l1 of y_true: ")
  actual_cost = tf.Print(actual_cost, [K.sum(K.abs(y_pred), axis=[-2,-1])[0, :]], summarize=2000, message="l1 of y_pred: ")
  actual_cost = tf.Print(actual_cost, [K.mean(K.binary_crossentropy(y_pred, y_true), axis=[-1,-2])[0, :]], summarize=2000,message="loss for sample 0: ")
  actual_cost = tf.Print(actual_cost, [(iou)[0, :]], summarize=2000, message="weighted_loss for sample 0: ")


  return actual_cost#*[0,0,0,0,0,0,0,0,0,1]
  #return actual_cost
  #return 1 - K.sqrt(actual_cost)

def binary_cross_entropy_loss(y_true, y_pred):
  zero_masks_percentage_per_class = np.asarray([0.78762727, 0.71392209, 0.93249225, 0.52611775, 0.05146082,
                                     0.65692784, 0.97864099, 0.97211155, 0.99081452, 0.94876051])**2

  weights = K.abs(1-y_true) + y_true / (np.ones(10) - zero_masks_percentage_per_class)
  weights = weights / K.sum(weights) * 10*16

  loss = K.binary_crossentropy(y_pred, y_true)
  loss= tf.Print(loss, [y_true[0, :]], summarize=2000,
                         message="y_true not_is_zero")
  loss = tf.Print(loss, [y_pred[0, :]], summarize=2000,
                         message="y_pred not_is_zero")
  loss = tf.Print(loss, [loss[0, :]], summarize=2000,
                  message="binary_cross_entropy")
  loss = loss
  loss = tf.Print(loss, [weights[0, :]], summarize=2000,
           message="weights")
  loss = tf.Print(loss, [loss[0, :]], summarize=2000,
           message="weighted_loss")

  return K.mean(loss, axis=-1)


def iou_loss_with_sample_weight(y_true, y_pred):
  # percentage of zero masks for crop 224x224
  zero_masks_percentage_per_class = [0.78762727, 0.71392209, 0.93249225, 0.52611775, 0.05146082,
                                     0.65692784, 0.97864099, 0.97211155, 0.99081452, 0.94876051]**2

  actual_cost = (10e-6 + K.sum(tf.mul(y_true, y_pred), axis=[-2,-1])) / (
    10e-6 + K.sum(K.abs(y_true), axis=[-2,-1]) +
    K.sum(K.abs(y_pred), axis=[-2,-1]) - K.sum(tf.mul(y_true, y_pred), axis=[-2,-1]))
  # sample weights, to take into consideration cropping and zero masks, assigning more weight to non-zero for sparse classes:

  is_zero = tf.to_float(tf.equal(tf.reduce_sum(y_true, reduction_indices=( 2,3)), 0))
  not_zero = tf.to_float(tf.not_equal(tf.reduce_sum(y_true, reduction_indices=( 2,3)), 0))
  weights = is_zero / zero_masks_percentage_per_class + not_zero / (np.ones(10) - zero_masks_percentage_per_class)

  return (1 - actual_cost)*weights

if __name__ == "__main__":

  #gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.333)
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess = tf.Session()
  with sess.as_default():
    #size of masks (5, 50176)
    #a = np.random.rand(1,5, 50176)

    #a = K.tensor list(np.random.rand(5, 50176))

    y_true = np.zeros([6, 50176])
    y_true[0, 0:10000] = 1
    y_true[1, 10000:20000] = 1
    y_true[2, 20000:30000] = 1
    #y_true[:, 3, 30000:35000] = 1
    #y_true[:, 4, 35000:] = 1

    #TODO: check assignment using masks vs assignment using eof from prediction!!!
    perm1 = [2, 1, 0, 4, 3, 5]
    perm2 = [1, 0, 3, 4, 2, 5]

    y_true = y_true.astype("float32")

    y_pred_masks_value = [y_true[perm1, :], y_true[perm2, :]]
    y_true_masks_value = [y_true, y_true]



    # TODO: add shapes for debugging
    y_true_masks = tf.placeholder('float32', shape=[2, 10, 224,224])
    y_pred_masks = tf.placeholder('float32', shape=[2, 10, 224,224])
    cost_ins = [y_true_masks, y_pred_masks]
    cost_functions = [iou_loss(y_true_masks, y_pred_masks)]

    sess = tf.get_default_session()
    print sess.run(cost_functions, feed_dict={cost_ins[0]: y_true_masks_value,
                                                    cost_ins[1]: y_pred_masks_value})


    # NOTICE: this results are hard to interpret, but they seem to match for the tests performed
    # for example, for the permutation 2, the last example perm2[2] = 3 won't be matched, since there are only 3 instances.
    #  It will assign the y_true[2] to the predicted mask[2], though the match would be with the predicted_mask[4],
    # but this is outside the groundtruth length (3)
    print 'a'
