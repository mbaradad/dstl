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
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from keras.models import model_from_json
from classifiers.resnet_keras.train import create_model
import matplotlib.pyplot as plt
from keras.layers.core import Dense
from callbacks import PlotLayer, MyTensorBoard

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    chunk_size = 12
    model_file = dirs.RESNET_KERAS_OUTPUT + "/vgg_train_2017-02-2217:10:08.822873/model"
    model = create_model(finetune=True, chunk_size=chunk_size, finetune_bands=True, finetune_upsampling=True)

    timestamp = str(datetime.datetime.now())

    lr = 1e-6

    save_path = dirs.RESNET_KERAS_OUTPUT + '/lr_' + str(lr) + '_continue_train_vgg'  + timestamp
    save_path = save_path.replace(' ', '')

    os.mkdir(save_path)

    sys.stdout = open( save_path + '/train.log', 'w')

    model.summary()

    ep = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    api_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    api_token = os.getenv('TELEGRAM_API_TOKEN')
    #tm = TelegramMonitor(api_token=api_token, chat_id=api_chat_id)


    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.0001)

    print 'using lr:' + str(lr)


    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss={'mask_output': iou_loss})
    d = Dataset(train=True, augmentation=True)

    model.load_weights(model_file + '.h5')

    generator_train = d.cropped_generator(chunk_size=chunk_size, crop_size=(224,224), overlapping_percentage=0.25, subset='train')
    generator_val = d.cropped_generator(chunk_size=chunk_size, crop_size=(224,224), overlapping_percentage=0.25, subset='val')

    tb = MyTensorBoard(generator_train, log_dir=save_path + '/tensorboard_logs', histogram_freq=1, write_graph=False, write_images=True)

    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    #Fixed samples per epoch to force model save
    #instead of d.get_n_samples(subset='train', crop_size=(224, 224)
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=8000,
                        validation_data=generator_val, nb_val_samples=1600,
                        callbacks=[mc, ep, tb])
                        #callbacks=[mc, ep, tm, p1, p2, p3, p4, p5, p6, p7, p8, p9])
