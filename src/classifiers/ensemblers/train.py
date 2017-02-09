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
from data.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
from utils.tf_iou import iou_loss
from classifiers.fcn8_keras.fcn8_keras import get_model_from_json
from classifiers.densenet_keras.densenet import create_dense_net
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.models import load_model
from keras.layers import Merge

models = ['']
def create_model():
  models_to_concat =[]
  #if oom, predict and store masks, and then train the ensemble.
  inputs = []
  for model in models:
    actual_model = models_to_concat.append(load_model(model + '/model.json'))
    actual_model.load_weights(model + '/model.h5')
    inputs.append(actual_model.input)

  a = Merge(mode='concat')(models_to_concat)
  a = Convolution2D(10, 3, 3, border_mode='same')(a)  # (10,224,224)
  a = BatchNormalization(name='bn_end')(a)
  a = Activation('relu')
  a = a = Convolution2D(10, 3, 3, border_mode='same')(a)  # (10,224,224)
  a = BatchNormalization(name='bn_end')(a)
  a = Activation('softmax')
  return Model(input=input, output=a)


def big_generator():
  d = Dataset(train=True)
  #while True:


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    model_0 = load_model(dirs.RESNET_KERAS_OUTPUT + "execution_2017-02-0820:09:59.765910/model.h5")

    model = create_model()

    timestamp = str(datetime.datetime.now())


    save_path = dirs.RESNET_KERAS_OUTPUT + '/execution_' + timestamp
    save_path = save_path.replace(' ', '')


    os.mkdir(save_path)

    sys.stdout = open( save_path + '/train.log', 'w')

    model.summary()

    ep = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)


    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.)

    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss=iou_loss)
    d = Dataset(train=True, augmentation=False)

    generator_train = d.cropped_generator(chunk_size=16, crop_size=(224,224), overlapping_percentage=0.2, subset='train')
    generator_val = d.cropped_generator(chunk_size=16, crop_size=(224,224), overlapping_percentage=0.2, subset='val')
    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=d.get_n_samples(subset='train', crop_size=(224,224), overlapping_percentage=0.2),
                        validation_data=generator_val, nb_val_samples=d.get_n_samples(subset='val', crop_size=(224,224), overlapping_percentage=0.2),
                        callbacks=[mc, ep, tb])


