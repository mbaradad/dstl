import utils.dirs as dirs
from keras.models import load_model
from classifiers.fcn8_keras.custom_layers.scale import Scale
from classifiers.fcn8_keras.custom_layers.bias import Bias
from utils.tf_iou import iou_loss
import numpy as np
from utils.dirs import RESNET_KERAS_OUTPUT
from classifiers.resnet_keras.train import create_model

class ResnetClassifier():
  def __init__(self, dir, chunk_size):
    self.model = create_model(chunk_size)
    self.model.load_weights(dir)

  def get_crop_size(self):
    return (224,224)

  def predict(self, x):
    return self.model.predict(x)