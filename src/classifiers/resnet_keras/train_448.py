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
from keras.optimizers import Adam, RMSprop
import os, datetime
import utils.dirs as dirs
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
from utils.tf_iou import iou_loss, binary_cross_entropy_for_class
from classifiers.fcn8_keras.fcn8_keras import get_model_from_json
from classifiers.fcn8_keras.custom_layers.bias import Bias
import sys
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
from ResnetMod import identity_block, conv_block
from keras.models import load_model

def adapt_to_imagenet(generator):
  while True:
    batch = generator.next()
    batch[0][0] = preprocess_input(batch[0][0]*255)
    yield batch
def create_model(finetune=False, load_resnet_weights=False, pretrained_resnet=False):
  chunk_size = 3
  image_height = 448
  image_width = 448

  hidden_dim = 2048
  feat_size = 14
  filter_dim = 3

  # inception keras
  #features_model = InceptionV3(include_top=False, weights='imagenet',
  #              input_tensor=None, input_shape=[3,224,224])
  #features_output = features_model.get_layer('activation_49')

  other_ch = Input([17, image_height, image_width], name='input_other_ch')
  features_model = ResNet50(weights=None, input_tensor=other_ch)
  other_ch_output = features_model.get_layer('activation_49').output
  with tf.device('/gpu:1'):

    #features_model = get_model_from_json('../../keras_bs/resnet_101/Keras_model_structure.json')
    features_model = get_model_from_json('../../keras_bs/resnet_101/Keras_model_structure_448.json')

    features_input = features_model.input
    features_output = features_model.get_layer('res5c_relu').output
    if load_resnet_weights:
      features_model.load_weights('../../keras_bs/resnet_101/Keras_model_weights.h5')

  if not finetune:
    for l in features_model.layers:
      l.trainable = False

  if pretrained_resnet and load_resnet_weights:
    print 'restoring from classification'
    features_model = get_model_from_json(dirs.RESNET_KERAS_OUTPUT + '/classification_2017-02-1300:58:55.958775/model.json')

    features_model.load_weights(dirs.RESNET_KERAS_OUTPUT + '/classification_2017-02-1300:58:55.958775/model.h5')
    if not finetune:
      for l in features_model.layers:
        l.trainable = False
    other_ch = features_model.input[1]
    other_ch_output = features_model.get_layer('activation_49').output
    features_input = features_model.input[0]
    features_output = features_model.get_layer('res5c_relu').output
    print 'finished restoring'

  features_model_name = 'resnet_101'
  features_output = Merge(mode='concat', concat_axis=1)([features_output, other_ch_output])

  #features_model = create_dense_net(10,(20,224,224), nb_dense_block=6)
  #                                  depth=40, nb_dense_block=6, growth_rate=20, nb_filter=20, dropout_rate=0.5,
  #                   weight_decay=1E-4, verbose=True)
  #features_model_name = 'densenet'
  #features_model.summary()
  #features_input = features_model.input
  #features_output = features_model.get_layer('activation_73')


  #TODO: set to not trainable
  skip_candidates = dict()
  skip_scale_init_weight = 0.1
  if(features_model_name == 'resnet_101'):
      #skip_candidates['224'] = features_model.input
    with tf.device('/gpu:1'):
      skip_candidates['112'] = features_model.get_layer('conv1_relu').output
      skip_candidates['56'] = ZeroPadding2D(padding=(1,0,1,0))(features_model.get_layer('res2c_relu').output)
      skip_candidates['28'] = features_model.get_layer('res3b3_relu').output
      skip_candidates['14'] = features_model.get_layer('res4b22_relu').output
      #if not features_model.get_layer('res5c_relu') is None:
      #   skip_candidates['7'] = features_model.get_layer('res5c_relu').output
      skip_candidates = add_skip_network(skip_candidates, hidden_dim, feat_size, skip_scale_init_weight)

  elif (features_model_name == 'resnet_152'):
      #skip_candidates['224'] = features_model.input
      skip_candidates['112'] = features_model.get_layer('conv1_relu').output
      skip_candidates['56'] = ZeroPadding2D(padding=(1, 0, 1, 0))(features_model.get_layer('res2c_relu').output)
      skip_candidates['28'] = features_model.get_layer('res3b7_relu').output
      skip_candidates['14'] = features_model.get_layer('res4b35_relu').output
      #if not features_model.get_layer('res5c_relu') is None:
      #  skip_candidates['7'] = features_model.get_layer('res5c_relu').output
      skip_candidates = add_skip_network(skip_candidates, hidden_dim, feat_size, skip_scale_init_weight)

  elif(features_model_name == 'asdf'):
      #skip_candidates['224'] = features_model.input
      skip_candidates['112'] = features_model.get_layer('activation_1').output
      skip_candidates['56'] = ZeroPadding2D(padding=(1,0,1,0))(features_model.get_layer('activation_10').output)
      skip_candidates['28'] = features_model.get_layer('activation_22').output
      skip_candidates['14'] = features_model.get_layer('activation_40').output
      #as we use 14x14 in some tests, check that the model contains it
      #if not features_model.get_layer('activation_49') is None:
      #   skip_candidates['7'] = features_model.get_layer('activation_49').output
      skip_candidates = add_skip_network(skip_candidates, hidden_dim, feat_size, skip_scale_init_weight)
  # Deconv layers

  #2048 instead of to compensate for not merging
  pre_hidden_dim = 2048#get_num_dimension_per_features_map_size(feat_size, feat_size, hidden_dim)
  a = features_output

  #a = Convolution2D(pre_hidden_dim, filter_dim, filter_dim, border_mode='same')(features_output)  # (10,224,224)
  #a = BatchNormalization()(a)
  for i in range(int(log(image_height/feat_size, 2))):
      # at each iteration (tsteps,256,*=2,*=2)
      actual_size = feat_size * 2**i
      #Because deconvoldution requires output_shape before the call(), we perform # params['chunk_size']*tsteps convolutions
      #deconv_hidden_dim = hidden_dim/2**i
      if str(actual_size/2) in skip_candidates.keys():
          actual_skip = skip_candidates[str(actual_size/2)]
          a = Merge(mode='concat', name='merge_deconv_res_' +str(i), concat_axis=1)([actual_skip, a])
          nb_filters = int(a.get_shape()[1])
          a = conv_block(a, 3, [nb_filters / 2, nb_filters / 2, nb_filters], strides=(1, 1), stage=0,
                         block='tu_conv_' + str(i))
          a = identity_block(a, 3, [nb_filters / 2, nb_filters / 2, nb_filters], stage=0, block='tu_i_0' + str(i))
          #a = identity_block(a, 3, [nb_filters / 2, nb_filters / 2, nb_filters], stage=0, block='tu_i_1' + str(i))

      else:
        print "Some skipped candidates couldn't be fetched (this is normal when not using resnet). Check dettention.py for skip candidates definition"
      deconv_hidden_dim = get_num_dimension_per_features_map_size(actual_size*2, feat_size, hidden_dim)
      #, input_shape = (chunk_size, pre_hidden_dim * 2, feat_size * 2 ** (i), feat_size * 2 ** (i))
      a = Deconvolution2D(deconv_hidden_dim, 3, 3,
                          (chunk_size, deconv_hidden_dim,feat_size*2**(i+1), feat_size*2**(i+1)), border_mode='same', subsample=(2,2), bias=False)(a)
      #a = UpSampling2D(size=(2, 2))(a)
      #a = Convolution2D(deconv_hidden_dim, 3, 3, border_mode='same')(a)
      pre_hidden_dim = deconv_hidden_dim
      #a = Activation('relu')(a)
      a = BatchNormalization()(a)

  #a = Convolution2D(pre_hidden_dim,filter_dim,filter_dim,border_mode='same')(a) # (10,224,224)
  #a = BatchNormalization(name='bn_end_1')(a)
  #a = Activation('relu')(a)
  a = Convolution2D(10, 1, 1, border_mode='same')(a)  # (10,224,224)
  #a = BatchNormalization(name='bn_end')(a)

  #w = int(a.get_shape()[2])
  #h = int(a.get_shape()[3])
  #a = Reshape([10, w*h])(a)
  #a = Activation('softmax')(a)
  #a = Lambda(lambda x: K.log(K.clip(x, 1e-6, 1 - 1e-6)))(a)
  #if not bias_trainable:
  #  a = Bias(beta_init=11, trainable=False)(a)
  #else:
  #  a = Bias(beta_init=11)(a)
  #a = Reshape([10, w,h])(a)

  '''TODO: simulate crf

  masks = Activation('sigmoid', name='mask_output')(a)
  merged = Merge(mode='concat')([masks, features_input])
  '''

  masks = Activation('sigmoid', name='mask_output')(a)

  #return_sequences = ConvLSTM2D(10, 30,30)

  model = Model(input=[features_input, other_ch], output=[masks], name='dstl')
  return model

def add_skip_network(skip_candidates, hidden_dims, feat_size, skip_scale_init_weight):
    processed_skip_candidates = dict()
    for k, layer in skip_candidates.items():
      a = Convolution2D(get_num_dimension_per_features_map_size(int(layer.get_shape()[3]), feat_size, hidden_dims), 3,
                        3, border_mode='same', name='skip_conv_dim_' + k)(layer)
      a = BatchNormalization(name='skip_bn_dim_' + k)(a)
      # init the gamma close to zero, and beta to zero so the skip network is not used at the begining
      size_scale = int(a.get_shape()[1])
      a = Scale(name='skip_scale_dim_' + k,
                weights=[np.asarray([skip_scale_init_weight] * size_scale), np.asarray([0] * size_scale)])(a)
      a = Activation('relu', name='skip_relu_dim_' + k)(a)
      # a = Dropout(0.5)(a)
      processed_skip_candidates[k] = a

      #Relu here doesn't mind given that identity has one
      #a = Activation('relu', name='skip_relu_dim_' + k)(a)
      #a = Dropout(0.5)(a)
    return processed_skip_candidates

def get_num_dimension_per_features_map_size(size, feat_size, hidden_dim):
    i = int(log(size / feat_size, 2))
    if size == 14:
      return 1024
    return max(hidden_dim / 4 ** (i), 30)


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  chunk_size = 3
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
  with sess.as_default():
    model = create_model(finetune=False, load_resnet_weights=True, pretrained_resnet=False)

    timestamp = str(datetime.datetime.now())


    save_path = dirs.RESNET_KERAS_OUTPUT + '/train_448_' + timestamp
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

    lr = 0.001
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, decay=0.00001)

    print 'using lr: ' + str(lr)

    # to be used when both classes and masks are being predicted
    model.compile(optimizer=opt, loss={'mask_output': iou_loss})
    d = Dataset(train=True, augmentation=True)

    generator_train = d.cropped_generator(chunk_size=chunk_size, crop_size=(448,448), overlapping_percentage=0.25, subset='train')
    generator_val = d.cropped_generator(chunk_size=chunk_size, crop_size=(448,448), overlapping_percentage=0.25, subset='val')
    json_string = model.to_json()
    open(save_path + '/model.json', 'w').write(json_string)
    #Fixed samples per epoch to force model save
    #instead of d.get_n_samples(subset='train', crop_size=(224, 224)
    model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=8000,
                        validation_data=generator_val, nb_val_samples=320,
                        callbacks=[mc, ep, tb, tm], max_q_size=100)
