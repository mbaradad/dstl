#The root folder is src, so all paths are relative to it
import os

INPUT = '../input'
SIXTEEN_BAND = INPUT + '/sixteen_band'
THREE_BAND = INPUT + '/three_band'
TRAIN_WKT = INPUT + '/train_wkt_v4.csv'
GRID_SIZES = INPUT + '/grid_sizes.csv'
SAMPLE_SUBMISSION = INPUT + '/sample_submission.csv'

PREPROCESSED_INPUT = '../preprocessed_input'
OUTPUT = '../output'
SUBMISSION = OUTPUT + '/submissions'
CLASSIFIERS = 'classifiers'
CLASSIFIERS_OUTPUT = OUTPUT + '/classifiers'

DENSNET_LASAGNE = CLASSIFIERS + '/densenet_lasagne'
DENSNET_LASAGNE_OUTPUT = CLASSIFIERS_OUTPUT + '/densenet_lasagne'

FCN8 = CLASSIFIERS + '/fcn8_caffe'
FCN8_OUTPUT = CLASSIFIERS_OUTPUT+ '/caffe'

FCN8_KERAS = CLASSIFIERS + '/fcn8_keras'
FCN8_KERAS_OUTPUT = CLASSIFIERS_OUTPUT+ '/fcn8_keras'

RESNET_KERAS = CLASSIFIERS + '/resnet_keras'
RESNET_KERAS_OUTPUT = CLASSIFIERS_OUTPUT+ '/resnet_keras'

TEMP_IMAGE_DIR = '/home/manel/Documents/'


def mkdir(dir_name):
  os.mkdir(dir_name)

if os.getcwd().endswith('src'):
  if not os.path.isdir(INPUT):
    os.mkdir(INPUT)
  if not os.path.isdir(PREPROCESSED_INPUT):
    os.mkdir(PREPROCESSED_INPUT)
  if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)
  if not os.path.isdir(SUBMISSION):
    os.mkdir(SUBMISSION)
  if not os.path.isdir(CLASSIFIERS_OUTPUT):
    os.mkdir(CLASSIFIERS_OUTPUT)
  if not os.path.isdir(FCN8_OUTPUT):
    os.mkdir(FCN8_OUTPUT)
  if not os.path.isdir(DENSNET_LASAGNE_OUTPUT):
    os.mkdir(DENSNET_LASAGNE_OUTPUT)
  if not os.path.isdir(FCN8_KERAS_OUTPUT):
    os.mkdir(FCN8_KERAS_OUTPUT)
  if not os.path.isdir(RESNET_KERAS_OUTPUT):
    os.mkdir(RESNET_KERAS_OUTPUT)
  if not os.path.isdir(TEMP_IMAGE_DIR):
    os.mkdir(TEMP_IMAGE_DIR)