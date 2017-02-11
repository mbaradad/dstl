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
from keras.optimizers import Adam, RMSprop
import os, datetime
import utils.dirs as dirs
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data.result_generator import ResultGenerator
from keras.preprocessing.image import ImageDataGenerator
from utils.tf_iou import ensembler_loss, iou_loss
from classifiers.fcn8_keras.fcn8_keras import get_model_from_json
from classifiers.densenet_keras.densenet import create_dense_net
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.models import load_model
from keras.layers import Merge
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import AtrousConvolution2D

models = ['']
def create_model(crops_per_dim=3):
  filter_dim = 15

  input_a = Input((3, 224*crops_per_dim, 224*crops_per_dim))
  input_b = Input((10, 224*crops_per_dim, 224*crops_per_dim))

  input = Merge(mode='concat', concat_axis=1)([input_a, input_b])
  #a = Flatten(name='flat_feats')(input_a)
  #a = RepeatVector(tsteps, name='rep_feats')(a)
  #a = Reshape((tsteps, input_dims[0], input_dims[1], input_dims[2]))(a)

  #a = ConvLSTM2D(100, 10, 10, dropout_U=0.5, dropout_W=0.5,return_sequences=True, border_mode='same')(a)
  #a = ConvLSTM2D(100, 10, 10, dropout_U=0.5, dropout_W=0.5, return_sequences=True, border_mode='same')(a)
  #a = ConvLSTM2D(100, 10, 10, dropout_U=0.5, dropout_W=0.5, return_sequences=True, border_mode='same')(a)
  #a = ConvLSTM2D(1000, 10, 10, dropout_U=0.5, dropout_W=0.5, return_sequences=True, border_mode='same')(a)
  #a = ConvLSTM2D(1000, 10, 10, dropout_U=0.5, dropout_W=0.5, return_sequences=True, border_mode='same')(a)
  #a = ConvLSTM2D(10, 224, 224, dropout_U=0.5, dropout_W=0.5, return_sequences=True, border_mode='same')(a)
  '''
  weights = np.zeros((10,13,9,9))
  biases =  np.zeros((10))
  weights = weights + 0.05
  for i in range(10):
    weights[i,3+i,5,5] = 1
  '''

  a = AtrousConvolution2D(10, filter_dim, filter_dim, border_mode='same', atrous_rate=(2,2))(input)
  a = Activation('relu')(a)

  '''
  weights = np.zeros((10, 10, 9, 9))
  weights = weights + 0.05
  for i in range(10):
    weights[i, i, 5, 5] = 1
  '''

  a = AtrousConvolution2D(10, filter_dim, filter_dim, border_mode='same', atrous_rate=(2,2))(a)
  a = Activation('relu')(a)
  a = AtrousConvolution2D(10, filter_dim, filter_dim, border_mode='same', atrous_rate=(2,2))(a)
  a = Activation('relu')(a)
  a = AtrousConvolution2D(10, filter_dim, filter_dim, border_mode='same', atrous_rate=(2,2))(a)
  a = Activation('relu')(a)
  a = AtrousConvolution2D(10, filter_dim, filter_dim, border_mode='same', atrous_rate=(2,2))(a)
  a = Activation('sigmoid')(a)


  #def slice(x):
  #  return x[:,tsteps-1]
  #a = Lambda(slice)(a)

  return Model(input=[input_a, input_b], output=a)



if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    model_file = dirs.RESNET_KERAS_OUTPUT + "/execution_2017-02-0921:27:39.762008/model"
    crops_per_dim = 3
    model = create_model(crops_per_dim=crops_per_dim)

    timestamp = str(datetime.datetime.now())

    save_path = dirs.RESNET_KERAS_OUTPUT + '/execution_' + timestamp
    save_path = save_path.replace(' ', '')

    os.mkdir(save_path)

    sys.stdout = open(save_path + '/train.log', 'w')

    model.summary()

    ep = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)


    opt = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.00001)

    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss=ensembler_loss)
    d = ResultGenerator(train=True, augmentation=False)

    generator_train = d.cropped_generator(chunk_size=16, crop_size=(224*crops_per_dim,224*crops_per_dim), overlapping_percentage=0.80, subset='train')
    generator_val = d.cropped_generator(chunk_size=16, crop_size=(224*crops_per_dim,224*crops_per_dim), overlapping_percentage=0.80, subset='val')
    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=8000,
                        validation_data=generator_val, nb_val_samples=800,
                        callbacks=[mc, ep, tb], max_q_size=100)


