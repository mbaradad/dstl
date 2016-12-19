from PIL.ImageOps import crop

from utils.dirs import *
import pandas as pd
import numpy as np
import random
random.seed(1337)

from image_processor import ImageProcessor
import time

#TODO: reserve some samples for validation (probably only when cropping is implemented).
class Dataset():
  def __init__(self, train=True, subset_idx_list=[], subset=-1):
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

    if len(subset_idx_list) > 0:
      self.image_list = self.image_list[subset_idx_list]
    if subset != -1:
      self.image_list = self.image_list[:subset]
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
      if self.train:
        masks = self.processor.get_masks(self.image_list[idx], images.shape[1], images.shape[2])
        self.preloaded_images[idx] = [images, masks]
      else:
        #for test, don't store the images in memory, as it is not worth it (there are lots of images, and don't
        #require preprocessing
        return [images, np.array()]
    return (self.preloaded_images[idx][0], self.preloaded_images[idx][1])

  def generate(self, idxs):
    # maybe store everything in memory
    images = list()
    masks = list()
    for id in idxs:
      im, m = self.generate_one(id)
      images.append(im)
      masks.append(m)
    return [images, masks]

  #TODO: we are losing some pixels at the end, because of rounding issues. Check if this is a problem
  # overlapping is for each side: i.e overlapping_percentage = 1 ensembling three crops (the actual crop, the one on the
  # and the one on the right.
  def generate_one_cropped(self, idx, crops_per_axis, i, j, overlapping_percentage = 0):
    if overlapping_percentage > 1:
      raise Exception("Overlapping of cropped generator can be greater than 1")
    [image, masks] = self.generate_one(idx)
    height = image.shape[1]
    width = image.shape[2]

    #TODO: maybe make it the same for all the images?
    #TODO: Add padding when necessary? (for crops and also for overlappings)
    crop_height = height / crops_per_axis
    crop_width = width / crops_per_axis

    if i == 0:
      crop_i_beg = 0
      # if it is the first tile, get two times the overlapping
      crop_i_end = (i + 1) * crop_height + crop_height * overlapping_percentage * 2
    else:
      #if it is the last tile, get two times the overlapping
      if i == crops_per_axis - 1:
        crop_i_beg = i * crop_height - crop_height * overlapping_percentage*2
      else:
        crop_i_beg = i*crop_height - crop_height*overlapping_percentage
      crop_i_end = (i + 1) * crop_height + crop_height * overlapping_percentage

    if j == 0:
      crop_j_beg = 0
      crop_j_end = (j + 1) * crop_width + crop_width * overlapping_percentage * 2
    else:
      if j == crops_per_axis - 1:
        crop_j_beg = j * crop_width - crop_width * overlapping_percentage*2
      else:
        crop_j_beg = j * crop_width - crop_width * overlapping_percentage
      crop_j_end = (j + 1) * crop_width + crop_width * overlapping_percentage

    image_cropped = image[:, crop_i_beg:crop_i_end, crop_j_beg:crop_j_end]
    masks_cropped = masks[:, crop_i_beg:crop_i_end, crop_j_beg:crop_j_end]
    return (image_cropped, masks_cropped)

  def generate_cropped(self, idxs, crops_per_axis, overlapping_percentage=0):
    # maybe store everything in memory
    images = None
    masks = None
    for id in idxs:
      image_id = id/(crops_per_axis**2)
      i = (id %(crops_per_axis**2))/crops_per_axis
      j = (id %(crops_per_axis**2))%crops_per_axis
      [im, m] = self.generate_one_cropped(image_id, crops_per_axis, i, j, overlapping_percentage)
      if images is None:
        images = np.expand_dims(im, axis=0)
        masks = np.expand_dims(m, axis=0)
      else:
        images = np.append(images, np.expand_dims(im, axis=0), axis=0)
        masks = np.append(masks, np.expand_dims(m, axis=0), axis=0)

    return [images, masks]

  def generator(self, chunk_size):
    #as image_list may be small, compute chunksize replica of index to shuffle
    idx_values = [x for x in range(len(self.image_list))]
    while True:
        # Select random indices each time, not sequential
        random.shuffle(idx_values)
        for i in range(len(idx_values) / chunk_size):
            idxs = np.sort(idx_values[i * chunk_size:(i + 1)* chunk_size])
            yield self.generate(idxs)

  def cropped_generator(self, chunk_size, crops_per_image, overlapping_percentage=0):
    #do the same as generator, but with crops
    n_crops = crops_per_image**2
    batch_len = n_crops*len(self.image_list)
    idx_values = [x for x in range(batch_len)]
    while True:
      random.shuffle(idx_values)
      for i in range(len(idx_values) / chunk_size):
        idxs = np.sort(idx_values[i * chunk_size:(i + 1) * chunk_size])
        yield self.generate_cropped(idxs, crops_per_image, overlapping_percentage)