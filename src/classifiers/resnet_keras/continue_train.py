from keras.layers import Input
from keras.layers.convolutional import Convolution2D, UpSampling2D, Deconvolution2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Reshape, Activation, Merge, RepeatVector, Dropout, Lambda
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import keras.backend as K
from kerastoolbox.callbacks import TelegramMonitor
from classifiers.fcn8_keras.custom_layers.scale import Scale
import numpy as np
from utils.utils import *
from math import log
from classifiers.resnet_keras.ResnetMod import ResNet50
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import os, datetime
import utils.dirs as dirs
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
from utils.tf_iou import iou_loss, binary_cross_entropy_loss
from classifiers.fcn8_keras.fcn8_keras import get_model_from_json
from classifiers.densenet_keras.densenet import create_dense_net
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from keras.models import model_from_json
from classifiers.resnet_keras.train import create_model
import matplotlib.pyplot as plt
from keras.layers.core import Dense

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    model_file = dirs.RESNET_KERAS_OUTPUT + "/execution_2017-02-0711:32:18.472465/model"
    model = create_model(finetune=True)

    timestamp = str(datetime.datetime.now())


    save_path = dirs.RESNET_KERAS_OUTPUT + '/execution_' + timestamp
    save_path = save_path.replace(' ', '')


    os.mkdir(save_path)

    sys.stdout = open( save_path + '/train.log', 'w')

    model.summary()

    ep = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)

    api_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    api_token = os.getenv('TELEGRAM_API_TOKEN')
    tm = TelegramMonitor(api_token=api_token, chat_id=api_chat_id)

    opt = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8)



    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss={'mask_output': iou_loss})
    d = Dataset(train=True, augmentation=True)

    model.load_weights(model_file + '.h5')

    generator_train = d.cropped_generator(chunk_size=16, crop_size=(224,224), overlapping_percentage=0.5, subset='train')
    generator_val = d.cropped_generator(chunk_size=16, crop_size=(224,224), overlapping_percentage=0.5, subset='val')
    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    #Fixed samples per epoch to force model save
    #instead of d.get_n_samples(subset='train', crop_size=(224, 224)
    model.fit_generator(generator_train, nb_epoch=200, samples_per_epoch=16000,
                        validation_data=generator_val, nb_val_samples=1600,
                        callbacks=[mc, ep, tb, tm])

    #model.fit_generator(generator_train, nb_epoch=200, samples_per_epoch=16000,
    #                    validation_data=generator_val, nb_val_samples=1600,
    #                    callbacks=[mc, ep, tb, tm])


