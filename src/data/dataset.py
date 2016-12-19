from utils.dirs import *
import pandas as pd
import numpy as np
import random
from image_processor import ImageProcessor
import time
import gc

#TODO: reserve some samples for validation (probably only when cropping is implemented).
class Dataset():
  def __init__(self, train=True, subset=-1):
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

    if subset != -1:
      self.image_list = self.image_list[subset]
    #To store previoulsy loaded images
    self.preloaded_images = dict()
    self.processor = ImageProcessor()

  def get_image_list(self):
    return self.image_list

  def generate_by_name(self, name):
    if name in self.image_list:
      return self.generate_one(self.image_list.index(name))

  def generate_one(self, idx):
    #only training images are stored in memory, test images are not
    if idx not in self.preloaded_images:
      images = self.processor.get_images(self.image_list[idx])
      gc.collect()
      if self.train:
        masks = self.processor.get_masks(self.image_list[idx], images.shape[1], images.shape[2])
        self.preloaded_images[idx] = [images, masks]
        gc.collect()
        return (self.preloaded_images[idx][0], self.preloaded_images[idx][1])
      else:
        return [images, np.array()]

  def generate(self, idxs):
    # maybe store everything in memory
    images = None
    masks = None
    for id in idxs:
      im, m = self.generate_one(id)
      if images is None:
        images = im
        masks = m
      else:
        images = np.append(images, im)
        masks = np.append(masks, im)
    return [images, masks]

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
  d = Dataset(subset=[0,1,2])

  start = time.clock()
  i = 0
  for j in d.generator(1):
    print "time elapsed for processing image " + str(i) + ": " + str(time.clock() - start)
    i += 1
    start = time.clock()