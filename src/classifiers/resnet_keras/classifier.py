import utils.dirs as dirs
from keras.models import load_model
from classifiers.fcn8_keras.custom_layers.scale import Scale
from classifiers.fcn8_keras.custom_layers.bias import Bias
from utils.tf_iou import iou_loss
import numpy as np
from utils.dirs import RESNET_KERAS_OUTPUT

class ResnetClassifier():
  def __init__(self, dir):
    self.model = load_model(dir, custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss})


  def get_crop_size(self):
    return (224,224)

  def predict(self, x):
    return self.model.predict(x)

class MultipleClassifier():
  def __init__(self):
    self.models = list()

    dir_0 = "execution_2017-02-1117:22:46.420973"
    dir_1 = "execution_2017-02-1117:22:46.420973"
    dir_2 = "execution_2017-02-1117:22:46.420973"
    dir_3 = "execution_2017-02-1117:22:46.420973"
    dir_4 = "execution_2017-02-1117:22:46.420973"
    dir_5 = "execution_2017-02-1117:22:46.420973"
    dir_6 = "execution_2017-02-1117:22:46.420973"
    dir_7 = "execution_2017-02-1117:22:46.420973"
    dir_8 = "execution_2017-02-1117:22:46.420973"
    dir_9 = "execution_2017-02-1117:22:46.420973"


    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_0+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_1+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_2+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_3+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_4+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_5+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_6+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_7+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_8+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))
    self.models.append(load_model(RESNET_KERAS_OUTPUT + "/"+dir_9+ "/model.h5", custom_objects={'Bias': Bias, 'Scale': Scale, 'iou_loss': iou_loss}))

    self.output_shape = self.models[0].output.shape

    def predict(self, x):
      y = np.zeros(self.output_shape)
      for i in len(self.models):
        y[i] = self.model.predict(x)[i]
      return y