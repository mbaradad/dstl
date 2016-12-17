from utils.dirs import *
import pandas as pd
import numpy as np
import random
from image_processor import ImageProcessor

class Dataset():
  def __init__(self, train=True):
    #Place the unziped files at this path
    self.root_path = INPUT
    self.train = train

    #only those with annotations are training, it should be 22 images
    df = pd.read_csv(TRAIN_WKT)
    train_images = df['ImageId'].unique()
    if self.train:
      self.image_list = np.asarray(train_images)
    else:
      #the opposite
      df1 = pd.read_csv(GRID_SIZES)
      all_images = df1['Unnamed: 0'].unique()
      self.image_list = [x for x in all_images if x not in np.asarray(train_images)]

    #To store previoulsy loaded images
    self.preloaded_images = dict()
    self.processor = ImageProcessor()

  def generate_by_name(self, name):
    if name in self.image_list:
      return self.generate_one(self.image_list.index(name))

  def generate_one(self, idx):
    #only training images are stored in memory, test images are not
    if idx not in self.preloaded_images:
      images = self.processor.get_images(self.image_list[idx])
      if self.train:
        masks= self.processor.get_masks(self.image_list[idx])
        self.preloaded_images[idx] = [images, masks]
        return self.preloaded_images[idx]
      else:
        return [images, np.array()]


  def generate(self, idxs):
    # maybe store everything in memory
    images = list()
    masks = list()
    for id in idxs:
      images, masks = self.generate_one(id)
      images.append(images)
      masks.append(masks)
    return np.asarray(masks), np.asarray(images)

  def generate_one_cropped(self, idx, crop_percentage, i, j, overlapping_percentage = 0):
    return None

  def crop_generator(self, crop_percentage, chunk_size):
    #do the same as generator, but with crops
    return None

  def generator(self, chunk_size):
    #as image_list may be small, compute chunksize replica of index to shuffle
    idx_values = [x%len(self.image_list) for x in range(len(self.image_list)*chunk_size)]
    while True:
        # Select random indices each time, not sequential
        random.shuffle(idx_values)
        for i in range(len(idx_values) / chunk_size):
            idxs = np.sort(idx_values[i * chunk_size:i * chunk_size + chunk_size])
            yield self.generate(idxs)


if __name__ == "__main__":
  d = Dataset()
  for i in d.generator(16):
    print len(i)
    print np.shape(i[0])
    print np.shape(i[0])

