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
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.models import load_model
from keras.layers import Merge, AveragePooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import AtrousConvolution2D
from classifiers.resnet_keras.callbacks import MyTensorBoard

models = ['']
def create_model(crops_per_dim, tsteps):

  input_dims = [13, 224*crops_per_dim, 224*crops_per_dim]

  input_a = Input((3, 224*crops_per_dim, 224*crops_per_dim))
  input_b = Input((10, 224*crops_per_dim, 224*crops_per_dim))

  input = Merge(mode='concat', concat_axis=1, name='input')([input_a, input_b])
  downsampling = 2
  input = AveragePooling2D((3, 3), strides=(downsampling, downsampling), border_mode='same')(input)

  a = Flatten(name='flat_feats')(input)
  a = RepeatVector(tsteps, name='rep_feats')(a)
  a = Reshape((tsteps, input_dims[0], input_dims[1]/downsampling, input_dims[2]/downsampling))(a)

  a = ConvLSTM2D(50, 10, 10, dropout_U=0.5, dropout_W=0.5,return_sequences=True, border_mode='same')(a)


  def slice(x):
    return x[:,tsteps-1]
  a = Lambda(slice)(a)

  a = Convolution2D(10, 1, 1, border_mode='same')(a)
  a = Activation('sigmoid', name='output')(a)

  a = UpSampling2D()(a)
  return Model(input=[input_a, input_b], output=a)



if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    crops_per_dim = 2
    chunk_size = 4
    tsteps = 5
    model = create_model(crops_per_dim=crops_per_dim, tsteps=5)

    timestamp = str(datetime.datetime.now())

    save_path = dirs.RESNET_KERAS_OUTPUT + '/ensembler_' + timestamp
    save_path = save_path.replace(' ', '')

    os.mkdir(save_path)

    sys.stdout = open(save_path + '/train.log', 'w')

    model.summary()

    ep = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)

    lr = 1e-3
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.00001)

    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss=ensembler_loss)
    d = ResultGenerator('/mnt/sdd1/submissions/submission_2017-02-21_01:35:42.024698', train=True, augmentation=False)

    generator_train = d.cropped_generator(chunk_size=chunk_size, crop_size=(224*crops_per_dim,224*crops_per_dim), overlapping_percentage=0.5, subset='train')
    generator_val = d.cropped_generator(chunk_size=chunk_size, crop_size=(224*crops_per_dim,224*crops_per_dim), overlapping_percentage=0.5, subset='val')

    batch = generator_train.next()
    batch = generator_train.next()
    tb_data = [batch[0][0][0:4], batch[0][1][0:4]]
    #tb = MyTensorBoard(tb_data, log_dir=save_path + '/tensorboard_logs', histogram_freq=1, write_graph=False,
    #                   write_images=True)

    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=2000,
                        validation_data=generator_val, nb_val_samples=200,
                        callbacks=[mc, ep])


