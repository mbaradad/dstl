from keras.layers import Input
from keras.layers.convolutional import Convolution2D, UpSampling2D, Deconvolution2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Reshape, Activation, Merge, RepeatVector, Dropout, Lambda
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import keras.backend as K

from classifiers.fcn8_keras.custom_layers.scale import Scale
import numpy as np
from utils.utils import *
from math import log
from classifiers.resnet_keras.ResnetMod import ResNet50
from keras.models import Model
from keras.optimizers import Adam
import os, datetime
import utils.dirs as dirs
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
from utils.tf_iou import iou_loss

def create_model():
  chunk_size = 18
  image_height = 224
  image_width = 224

  hidden_dim = 2048
  feat_size = 7
  filter_dim = 3


  features_model = ResNet50(weights=None, input_shape=(3, 224,224))
  features_input = features_model.input
  features_output = features_model.get_layer('activation_49')

  #TODO: add batch norms

  #TODO: set to not trainable
  skip_candidates = dict()
  skip_scale_init_weight = 0.1

  skip_candidates['224'] = features_model.input
  skip_candidates['112'] = features_model.get_layer('activation_1').output
  skip_candidates['56'] = ZeroPadding2D(padding=(1,0,1,0))(features_model.get_layer('activation_10').output)
  skip_candidates['28'] = features_model.get_layer('activation_22').output
  skip_candidates['14'] = features_model.get_layer('activation_40').output
  #as we use 14x14 in some tests, check that the model contains it
  if not features_model.get_layer('activation_49') is None:
     skip_candidates['7'] = features_model.get_layer('activation_49').output
  skip_candidates = add_skip_network(skip_candidates, hidden_dim, feat_size, skip_scale_init_weight)

  # Deconv layers
  pre_hidden_dim = get_num_dimension_per_features_map_size(feat_size, feat_size, hidden_dim)

  a = Convolution2D(pre_hidden_dim, filter_dim, filter_dim, border_mode='same')(features_output.output)  # (10,224,224)
  a = BatchNormalization()(a)
  for i in range(int(log(image_height/feat_size, 2))):
      # at each iteration (tsteps,256,*=2,*=2)
      actual_size = feat_size * 2**i
      #Because deconvolution requires output_shape before the call(), we perform # params['chunk_size']*tsteps convolutions
      #deconv_hidden_dim = hidden_dim/2**i
      if str(actual_size) in skip_candidates.keys():
          actual_skip = skip_candidates[str(actual_size)]
          a = Merge(mode='sum', name='merge_deconv_res_' +str(i))([actual_skip, a])
          a = Activation('relu')(a)
          a = BatchNormalization()(a)
      else:
        print "Some skipped candidates couldn't be fetched (this is normal when not using resnet). Check dettention.py for skip candidates definition"
      deconv_hidden_dim = get_num_dimension_per_features_map_size(actual_size*2, feat_size, hidden_dim)
      b = Deconvolution2D(deconv_hidden_dim,filter_dim,filter_dim,
                          (chunk_size, deconv_hidden_dim,feat_size*2**(i+1), feat_size*2**(i+1))
                          , border_mode='same', subsample=(2,2), bias=False, input_shape=(chunk_size, pre_hidden_dim, feat_size*2**(i), feat_size*2**(i)))(a)
      pre_hidden_dim = deconv_hidden_dim
      a = Activation('relu')(b)
      a = BatchNormalization()(a)

  a = Convolution2D(10,filter_dim,filter_dim,border_mode='same')(a) # (10,224,224)
  a = BatchNormalization(name='bn_end')(a)

  '''
  w = int(a.get_shape()[3])
  h = int(a.get_shape()[4])
  a = Reshape((tsteps, w*h))(a)
  a = Activation('softmax')(a)
  a = Lambda(lambda x: K.log(x))(a)
  #Change for a simple bias, instead of a conv
  a = TimeDistributed(Bias(beta_init='one'))(a)
  a = Reshape((tsteps, 1, w, h))(a)
  '''


  masks = Activation('sigmoid', name='mask_output')(a)

  model = Model(input=features_input,output=[masks])
  return model

def add_skip_network(skip_candidates, hidden_dims, feat_size, skip_scale_init_weight):
    processed_skip_candidates = dict()
    for k, layer in skip_candidates.items():
        a = Convolution2D(get_num_dimension_per_features_map_size(int(layer.get_shape()[3]), feat_size, hidden_dims), 3, 3, border_mode='same', name='skip_conv_dim_' + k)(layer)
        a = BatchNormalization(name='skip_bn_dim_' + k)(a)
        #init the gamma close to zero, and beta to zero so the skip network is not used at the begining
        size_scale = int(a.get_shape()[1])
        a = Scale(name='skip_scale_dim_' + k, weights=[np.asarray([skip_scale_init_weight]*size_scale), np.asarray([0]*size_scale)])(a)
        a = Activation('relu', name='skip_relu_dim_' + k)(a)
        processed_skip_candidates[k] = a
    return processed_skip_candidates

def get_num_dimension_per_features_map_size(size, feat_size, hidden_dim):
    i = int(log(size/feat_size, 2))
    #return max(hidden_dim / 2 ** (i + 3), 1)
    return max(hidden_dim / 2 ** (i + 1), 1)


if __name__ == "__main__":

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  sess = tf.Session()
  with sess.as_default():
    model = create_model()

    timestamp = str(datetime.datetime.now())

    model.summary()

    save_path = dirs.RESNET_KERAS_OUTPUT + '/execution_' + timestamp
    os.mkdir(save_path)

    ep = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)


    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.)

    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss=iou_loss)

    d = Dataset(train=True, augmentation=True)

    generator_train = d.cropped_generator(chunk_size=18, crop_size=(224,224), overlapping_percentage=0.2, subset='train')
    generator_val = d.cropped_generator(chunk_size=18, crop_size=(224,224), overlapping_percentage=0.2, subset='val')
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=d.get_n_samples(subset='train', crop_size=(224,224)),
                        validation_data=generator_val, nb_val_samples=d.get_n_samples(subset='val', crop_size=(224,224)),
                        callbacks=[mc, ep, tb])


