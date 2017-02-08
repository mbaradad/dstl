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
  def __init__(self, train=True, augmentation=False, normalize=True, downsampling=1):

    #Place the unziped files at this path
    self.root_path = INPUT
    self.train = train
    self.augmentation = augmentation
    #orig + 3 rotation + vert_flip, horizontal_flip = 6
    self.augmentation_factor = 8
    self.last_augmentation = 0
    self.normalize = normalize
    self.downsampling = downsampling
    #self.classes_only = [9]
    self.select_non_zero_instances = False

    #only those with annotations are training, it should be 22 images
    df = pd.read_csv(TRAIN_WKT)
    train_images = df['ImageId'].unique()
    if self.train:
      self.image_list = np.asarray(train_images)
    else:
      #the opposite
      self.grid_sizes = pd.read_csv(GRID_SIZES)
      all_images = self.grid_sizes['Unnamed: 0'].unique()
      #this three are in train, but should also be included in eval
      #TODO: maybe use groundtruth directly, to achieve small increase for the following instances
      #self.image_list = ['6070_2_3', '6010_1_2', '6040_4_4', '6100_2_2']

      self.image_list = [x for x in all_images if x not in np.asarray(train_images)]
      self.image_list.append(['6070_2_3', '6010_1_2', '6040_4_4', '6100_2_2'])

      # Read the CSV into a pandas data frame (df)
      #   With a df you can do many things
      #   most important: visualize data with Seaborn
      df = pd.read_csv(SAMPLE_SUBMISSION, delimiter=',')

      self.image_list = list(df.ix[range(0,len(df),10),0])

    #To store previoulsy loaded images
    self.preloaded_images = dict()
    self.processor = ImageProcessor(downsampling=self.downsampling)

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

    self.stds = np.asarray([  25.21697092,   20.26343569,    9.84691434,  274.77193071,
        319.14481443,  292.78475428,  274.8996716 ,  270.56382503,
        236.50423548,  244.0594615 ,  255.95098918,    5.41074222,
          9.81402107,   20.19385471,   29.7182571 ,   25.04388324,
         36.20191917,   39.99403428,   41.28618233,   29.8104115 ])

    #percentage of zero masks for crop 224x224
    self.zero_masks_percentage_per_class = [7.04225352e-02, 1.70496664e-02, 4.26982950e-01, 7.41289844e-04,
                                            0.000001e+00, 3.70644922e-03, 7.62045960e-01, 7.01260193e-01, 8.85100074e-01, 5.06300964e-01]
    #cache for test,
    self.last_im_idx = -1
    self.last_im = None

  def class_id_to_name(self, id):
    return self.classes[id]

  def get_image_list(self):
    return self.image_list

  def get_grid_size(self, idx):
    imname = self.image_list[idx]
    x_max = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == imname].iloc[0, 1]
    y_min = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == imname].iloc[0, 2]
    return [x_max, y_min]

  def get_n_samples(self, subset, crop_size, overlapping_percentage):
    if self.augmentation:
      return len(self.get_generator_idxs(crop_size, subset, overlapping_percentage)) * self.augmentation_factor
    else:
      return len(self.get_generator_idxs(crop_size, subset, overlapping_percentage))

  def generate_by_name(self, name):
    if name in self.image_list:
      return self.generate_one(self.image_list.index(name))

  def generate_one(self, idx):
    #only training images are stored in memory, test images are not
    if self.train:
      if not idx in self.preloaded_images.keys():
        images = self.processor.get_images(self.image_list[idx])
        masks = self.processor.get_masks(self.image_list[idx], images.shape[1], images.shape[2])
        self.preloaded_images[idx] = [images, masks]
      return (self.preloaded_images[idx][0], self.preloaded_images[idx][1])
    else:
      # for test, don't store the images in memory, as it is not worth it (there are lots of images, and don't
      # require preprocessing
      if idx != self.last_im_idx:
        images = self.processor.get_images(self.image_list[idx])
        self.last_im = images
        self.last_im_idx = idx
      else:
        images = self.last_im
      return [images, None]

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
    if self.train:
      masks_cropped = masks[:, x_begin:(x_begin+crop_size[0]), y_begin:(y_begin+crop_size[1])]
    else:
      masks_cropped = None
    if self.normalize:
      image_cropped = np.transpose((np.transpose(image_cropped, [1, 2, 0]) - self.means[0:len(image_cropped)]) / self.stds[0:len(image_cropped)], [2, 0, 1])*255.0

    #sample weights, to take into consideration cropping and zero masks, assigning more weight to non-zero for sparse classes:
    #is_zero = np.sum(masks_cropped, axis=(1, 2)) == 0
    #weights = (is_zero/self.zero_masks_percentage_per_class) + np.abs(is_zero - 1)/(np.ones(10) - self.zero_masks_percentage_per_class)
    #weights = weights / np.sum(weights)*10
    weights = np.ones(10)
    return (image_cropped, masks_cropped, weights)

  def augment(self, im, mask=False):
    t_im = np.transpose(im, [1, 2, 0])
    if self.last_augmentation == 0:
      im = im
    elif self.last_augmentation == 1:
      im = np.transpose(np.rot90(t_im, 1), [2, 0, 1])
    elif self.last_augmentation == 2:
      im = np.transpose(np.rot90(t_im,2), [2, 0, 1])
    elif self.last_augmentation == 3:
      im = np.transpose(np.rot90(t_im,3), [2, 0, 1])
    elif self.last_augmentation == 4:
      im = np.transpose(np.fliplr(t_im), [2, 0, 1])
    elif self.last_augmentation == 5:
      im = np.transpose(np.flipud(t_im), [2, 0, 1])
    elif self.last_augmentation == 6:
      im = np.transpose(np.rot90(np.flipud(t_im), 1), [2, 0, 1])
    elif self.last_augmentation == 7:
      im = np.transpose(np.rot90(np.fliplr(t_im), 1), [2, 0, 1])
    if mask:
      self.last_augmentation += 1
    return im

  def generate_cropped(self, idxs, crop_size):
    # maybe store everything in memory
    images = None
    masks = None
    weights = None
    for id in idxs:
      image_id = id[0]
      x_begin = id[1]
      y_begin = id[2]

      [im, m, w] = self.generate_one_cropped(image_id, crop_size, x_begin, y_begin)
      if self.augmentation:
        im = self.augment(im)
        m = self.augment(m, mask=True)

      actual_images = np.expand_dims(im, axis=0)
      actual_masks = np.expand_dims(m, axis=0)
      actual_weights = np.expand_dims(w, axis=0)
      if images is None:
        images = actual_images
        masks = actual_masks
        weights = actual_weights
      else:
        images = np.append(images, actual_images, axis=0)
        masks = np.append(masks, actual_masks, axis=0)
        weights = np.append(weights, actual_weights , axis=0)
    not_is_zero = np.asarray(np.sum(masks, axis=(-1,-2)) != 0, dtype='float32')
    not_is_zero_sum = np.transpose(np.reshape(np.tile(np.sum(not_is_zero, axis=-1),10),[10,16]))
    not_is_zero_sum = not_is_zero_sum + 1*(not_is_zero_sum == 0)
    not_is_zero = not_is_zero/not_is_zero_sum
    #return [[images[:, 0:3, :, :], images[:, 3:, :, :]], [masks, not_is_zero], [weights, weights]]
    return [[images[:,0:3,:,:], images[:,3:,:,:]], masks]

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
    elif self.train and subset not in ['val', 'train']:
      raise Exception(
        "The dataset has been inizialized with training data, so the subset (val or train) should be specified.")
    if subset == 'val':
      idxs = [x for x in idxs if (x[0] + x[1] + x[2]) % 10 == 9]
    if subset == 'train':
      idxs = [x for x in idxs if (x[0] + x[1] + x[2]) % 10 != 9]
    return idxs

  #set subset to val or train for the case
  def cropped_generator(self, chunk_size, crop_size, overlapping_percentage=0, subset=""):
    #do the same as generator, but with crops
    idx_values = self.get_generator_idxs(crop_size, subset, overlapping_percentage)
    to_yield = list()
    good_idx_found = False
    while True:
      random.shuffle(idx_values)
      good_idx = list()
      for i in range(len(idx_values) / chunk_size):
        idxs = idx_values[i * chunk_size:(i + 1) * chunk_size]
        if good_idx_found or not self.select_non_zero_instances:
          yield self.generate_cropped(idxs, crop_size)
          continue
        else:
          actual = self.generate_cropped(idxs, crop_size)
          not_is_zero = np.sum(actual[1][:,self.classes_only], axis=(-1,-2,-3)) != 0
          for i in range(len(not_is_zero)):
            if not_is_zero[i]:
              to_yield.append([[actual[0][0][i], actual[0][1][i]], actual[1][i]])
              good_idx.append(idxs[i])
          if len(to_yield) > 16:
            actual_yield = to_yield[:16]
            to_yield = to_yield[16:]
            images = np.expand_dims(actual_yield[0][0][0],axis=0)
            channels = np.expand_dims(actual_yield[0][0][1],axis=0)
            masks = np.expand_dims(actual_yield[0][1],axis=0)
            for i in range(1,16):
              images = np.append(images, np.expand_dims(actual_yield[i][0][0],axis=0), axis=0)
              channels = np.append(channels, np.expand_dims(actual_yield[i][0][1],axis=0), axis=0)
              masks = np.append(masks, np.expand_dims(actual_yield[i][1],axis=0), axis=0)
            yield [[images, channels],masks]

      if not good_idx_found and self.select_non_zero_instances and len(good_idx) > 0:
        idx_values = good_idx
        print 'GOOOOOOOOOOOOD IDX OF GENERATOR FOUND. SHOULD SPEED UP NOW, check wnvidia'
        print 'number of instances without augmentation: ' + str(len(idx_values))
        good_idx_found = True

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  d_train = Dataset(train=True, augmentation=True, normalize=True)
  generator = d_train.cropped_generator(chunk_size=16, crop_size=(224,224), overlapping_percentage=0.2, subset='train')
  is_zero_counts = np.zeros([10])
  area = np.zeros([10])
  for i in range(100000):
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    images, masks = generator.next()
    area = area + np.sum(masks, axis=(0,-1,-2))
    print 'total_area =' + str(area/((i+1)*16*224*224))
    is_zero_counts = is_zero_counts + np.sum(np.sum(masks, axis=(2, 3)) == 0, axis=0)/16.0
    if is_zero_counts[9] != 1.0:
      print 'car_found'
    print is_zero_counts/(i+1)
    end_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    total_time = (datetime.datetime.strptime(end_time, '%H:%M:%S') - datetime.datetime.strptime(start_time, '%H:%M:%S'))
    print "Time to process one batch: " + str(total_time)

  '''

  import matplotlib.pyplot as plt
  d_train = Dataset(train=True, augmentation=True, normalize=True)

  for idx in range(len(d_train.image_list)):
    print 'storing image: ' + str(idx)
    images, masks = d_train.generate_one(idx)
    plt.imshow(np.transpose(images[0:3], [1, 2, 0]))
    plt.savefig('../images/img_' + str(idx) + '.jpg')
    for i in range(0,10):
      plt.imshow(masks[i])
      plt.savefig('../images/img_' + str(idx) + '_m_' + str(i) + '_' + d_train.class_id_to_name(i) +'.jpg')
  '''
