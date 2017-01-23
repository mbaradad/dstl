from classifiers.resnet_keras.train import create_model
import utils.dirs as dirs

class ResnetClassifier():
  def __init__(self):
    self.model = create_model()
    self.model.load_weights(dirs.RESNET_KERAS_OUTPUT + "/execution _2017-01-23 20:50:08.514709/model.h5")


  def get_crop_size(self):
    return (224,224)

  def classify(self, x):
    self.model.predict(x)