from PIL.ImageOps import crop

from utils.dirs import *
import pandas as pd
import numpy as np
import random
random.seed(1337)

from image_processor import ImageProcessor
import datetime

#When cropping, one out of 10 samples is reserved for validation, taking the modulus of the sample position
#and the rest 9/10 are used for training
#TODO: reserve some samples for validation (probably only when cropping is implemented).
class Dataset():
  def __init__(self, train=True, augmentation=False, subset_idx_list=[], normalize=True, subset=-1):
    if len(subset_idx_list) > 0 and subset != -1:
      raise Exception("Only idx list or subset can be -1")

    #Place the unziped files at this path
    self.root_path = INPUT
    self.train = train
    self.augmentation = augmentation
    #orig + 3 rotation + vert_flip, horizontal_flip = 6
    self.augmentation_factor = 6
    self.normalize = normalize

    #only those with annotations are training, it should be 22 images
    df = pd.read_csv(TRAIN_WKT)
    train_images = df['ImageId'].unique()
    if self.train:
      self.image_list = np.asarray(train_images)
    else:
      #the opposite
      self.grid_sizes = pd.read_csv(GRID_SIZES)
      all_images = self.grid_sizes['Unnamed: 0'].unique()
      self.image_list = [x for x in all_images if x not in np.asarray(train_images)]


    if len(subset_idx_list) > 0:
      self.image_list = self.image_list[subset_idx_list]
    elif subset != -1:
      self.image_list = self.image_list[:subset]
    #To store previoulsy loaded images
    self.preloaded_images = dict()
    self.processor = ImageProcessor()

    self.image_sizes = []
    for idx in self.image_list:
      self.image_sizes.append(self.processor.get_image_size(idx))

    self.classes = ["Buildings",
      "Manmade structures",
      "Road",
      "Track",
      "Trees",
      "Crops",
      "Waterway",
      "Standing water",
      "Vehicle Large",
      "Vehicle Small"]
    self.means = np.asarray([  329.95846619,   414.74250944,   305.02988176,  3191.11093819,
        3718.8890884 ,  3401.60485121,  3127.57398649,  2612.33625032,
        2354.97220436,  2288.29265143,  2288.09404569,   293.54224129,
         305.14513705,   414.94993739,   464.7065359 ,   330.15337527,
         449.86065868,   409.28267764,   422.98811144,   451.28799937])

    self.stds = [  25.21697092,   20.26343569,    9.84691434,  274.77193071,
        319.14481443,  292.78475428,  274.8996716 ,  270.56382503,
        236.50423548,  244.0594615 ,  255.95098918,    5.41074222,
          9.81402107,   20.19385471,   29.7182571 ,   25.04388324,
         36.20191917,   39.99403428,   41.28618233,   29.8104115 ]

  def class_id_to_name(self, id):
    return self.classes[id]

  def get_image_list(self):
    return self.image_list

  def get_grid_size(self, idx):
    imname = self.image_list[idx]
    x_max = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == imname].iloc[0, 1]
    y_min = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == imname].iloc[0, 2]
    return [x_max, y_min]

  def get_n_samples(self, subset, crop_size):
    return len(self.get_generator_idxs(crop_size, subset))*self.augmentation_factor

  def generate_by_name(self, name):
    if name in self.image_list:
      return self.generate_one(self.image_list.index(name))

  def generate_one(self, idx):
    #only training images are stored in memory, test images are not
    if idx not in self.preloaded_images:
      images = self.processor.get_images(self.image_list[idx])
      if self.normalize:
        np.transpose((np.transpose(images, [1, 2, 0]) - self.means) / self.stds, [2, 0, 1])
      if self.train:
        masks = self.processor.get_masks(self.image_list[idx], images.shape[1], images.shape[2])
        self.preloaded_images[idx] = [images, masks]
      else:
        #for test, don't store the images in memory, as it is not worth it (there are lots of images, and don't
        #require preprocessing
        return [images, None]
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
  #TODO: add data augmentation
  def generate_one_cropped(self, idx, crop_size, x_begin, y_begin):
    [image, masks] = self.generate_one(idx)
    image_cropped = image[:, x_begin:(x_begin+crop_size[0]), y_begin:(y_begin+crop_size[1])]
    masks_cropped = masks[:, x_begin:(x_begin+crop_size[0]), y_begin:(y_begin+crop_size[1])]
    return (image_cropped, masks_cropped)

  def augment(self, im):
    ims = np.expand_dims(im, axis=0)
    t_im = np.transpose(im, [1, 2, 0])
    ims = np.append(ims, np.expand_dims(np.transpose(np.rot90(t_im,1), [2, 0, 1]), axis=0), axis=0)
    ims = np.append(ims, np.expand_dims(np.transpose(np.rot90(t_im,2), [2, 0, 1]), axis=0), axis=0)
    ims = np.append(ims, np.expand_dims(np.transpose(np.rot90(t_im,3), [2, 0, 1]), axis=0), axis=0)
    ims = np.append(ims, np.expand_dims(np.transpose(np.fliplr(t_im), [2, 0, 1]), axis=0), axis=0)
    ims = np.append(ims, np.expand_dims(np.transpose(np.flipud(t_im), [2, 0, 1]), axis=0), axis=0)
    return ims

  def generate_cropped(self, idxs, crop_size):
    # maybe store everything in memory
    images = None
    masks = None
    for id in idxs:
      image_id = id[0]
      x_begin = id[1]
      y_begin = id[2]

      [im, m] = self.generate_one_cropped(image_id, crop_size, x_begin, y_begin)
      if self.augmentation:
        actual_images = self.augment(im)
        actual_masks = self.augment(m)
      else:
        actual_images = np.expand_dims(im, axis=0)
        actual_masks = np.expand_dims(m, axis=0)
      if images is None:
        images = actual_images
        masks = actual_masks
      else:
        images = np.append(images, actual_images, axis=0)
        masks = np.append(masks, actual_masks, axis=0)

    return [images, masks]

  def generator(self, chunk_size):
    #as image_list may be small, compute chunksize replica of index to shuffle
    idx_values = [x for x in range(len(self.image_list))]
    while True:
        # Select random indices each time, not sequential
        random.shuffle(idx_values)
        for i in range(len(idx_values) / chunk_size):
            idxs = idx_values[i * chunk_size:(i + 1)* chunk_size]
            yield self.generate(idxs)
  '''
  Returns a set of idx where:
    idx[0] = image_idx
    idx[1] = x_begin
    idx[2] = y_begin
  Taking into account image sizes, crop sizes and overlapping
  If subset is val, selects one out of then
  If it is train, the rest
    '''
  def get_generator_idxs(self, crop_size, subset, overlapping_percentage=0):
    idxs = []
    for idx in range(len(self.image_list)):
      width, height = self.image_sizes[idx]
      step_x = int(crop_size[0]*(1 - overlapping_percentage))
      step_y = int(crop_size[1]*(1 - overlapping_percentage))
      if step_x <= 0 or step_y <= 0:
        raise Exception("Overlapping percentage is too big")
      possible_x = [x for x in range(0, width, step_x) if x < width - crop_size[0]]
      possible_x.append(width - crop_size[0])
      possible_y = [y for y in range(0, height, step_y) if y < height - crop_size[1]]
      possible_y.append(height - crop_size[1])
      for i in range(len(possible_x)):
        for j in range(len(possible_y)):
          idxs.append([idx, possible_x[i], possible_y[j]])

    # select a subset for training and validation. For test use all the samples (though provably this generator is not used on the test set)
    if subset != "" and not self.train:
      raise Exception(
        "The subset (val or train) should not be specified for the test data")
    if subset not in ['val', 'train']:
      raise Exception(
        "The dataset has been inizialized with training data, so the subset (val or train) should be specified.")
    if subset == 'val':
      idxs = [x for x in idxs if (x[0] + x[1] + x[2]) % 10 == 9]
    if subset == 'train':
      idxs = [x for x in idxs if (x[0] + x[1] + x[2]) % 10 != 9]
    return idxs

  #set subset to val or train for the case
  def cropped_generator(self, chunk_size, crop_size, overlapping_percentage=0, subset=""):
    if self.augmentation and chunk_size % self.augmentation_factor != 0:
      raise Exception("chunks_size must be multiple of the dataset augemntation factor, if autmentation is enabled")
    if self.augmentation:
      chunk_size = chunk_size/self.augmentation_factor
    #do the same as generator, but with crops
    idx_values = self.get_generator_idxs(crop_size, subset, overlapping_percentage)
    while True:
      random.shuffle(idx_values)
      for i in range(len(idx_values) / chunk_size):
        idxs = idx_values[i * chunk_size:(i + 1) * chunk_size]
        yield self.generate_cropped(idxs, crop_size)


if __name__ == '__main__':
  d_train = Dataset(train=True, augmentation=True)
  generator = d_train.cropped_generator(chunk_size=12, crop_size=(224,224), overlapping_percentage=0.1, subset='train')
  for i in range(1000):
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    generator.next()
    end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time, '%H:%M:%S'))
    print "Time to process one batch: " + str(total_time)
