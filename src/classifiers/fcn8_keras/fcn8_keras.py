from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from classifiers.fcn8_keras.custom_layers.scale import Scale
from classifiers.fcn8_keras.custom_layers.bias import Bias
import utils.dirs as dirs
import os
import datetime
from data.dataset import Dataset
from utils.tf_iou import iou_loss

def get_model_from_json(json_path):
  return model_from_json(open(json_path, 'r').read(), custom_objects={'Scale': Scale, 'Bias':Bias})


if __name__ == "__main__":

  timestamp = str(datetime.datetime.now())

  model = get_model_from_json('../../keras_bs/fcn8/Keras_model_structure.json')
  model.summary()


  save_path = dirs.FCN8_OUTPUT + '/execution _' + timestamp
  os.mkdir(save_path)

  ep = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
  mc = ModelCheckpoint(save_path + '/model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
  #tb = TensorBoard(log_dir=save_path + '/tensor.log', histogram_freq=0, write_graph=True, write_images=False)


  opt = Adam(lr=0.0001)

  # to be used when both classes and masks are being predicted
  model.compile(optimizer=opt, loss=iou_loss)

  d = Dataset(train=True, augmentation=True)
  generator_train = d.cropped_generator(chunk_size=1, crop_size=(224,224), overlapping_percentage=0.1, subset='train')
  generator_val = d.cropped_generator(chunk_size=1, crop_size=(224,224), overlapping_percentage=0.1, subset='val')
  model.fit_generator(generator_train, nb_epoch=500, samples_per_epoch=d.get_n_samples(subset='train', crop_size=(224,224)),
                      validation_data=generator_val, nb_val_samples=d.get_n_samples(subset='val', crop_size=(224,224)),
                      callbacks=[mc, ep])

