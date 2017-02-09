import utils.dirs as dirs
from keras.models import load_model
from classifiers.fcn8_keras.custom_layers.scale import Scale
from classifiers.fcn8_keras.custom_layers.bias import Bias
from utils.tf_iou import iou_loss

class ResnetClassifier():
  def __init__(self, dir):
    self.model = load_model(dir, custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss})


  def get_crop_size(self):
    return (224,224)

  def predict(self, x):
    return self.model.predict(x)