from data.dataset import Dataset
import random
import numpy as np
import datetime
from utils.dirs import TEMP_IMAGE_DIR
from image_processor import ImageProcessor
import tifffile as tiff

class ResultGenerator():
  def __init__(self, submission_dir, train=True, augmentation=False, normalize=True, store_processed_images=True):
    #augmentation must be performed in this class, as the dataset augmentation can be extrapolated
    if augmentation:
      raise Exception('augmentation not implemented')

    self.dataset = Dataset(train=train, augmentation=False, normalize=normalize)
    self.image_list = self.dataset.get_image_list()
    self.augmentation = augmentation
    self.submission_dir = submission_dir
    self.preprocessed_images = dict()
    self.train = train
    self.store_processed_images = store_processed_images

  def generate_one_cropped(self, idx, crop_size, x_begin, y_begin):
    predicted_masks_list = self.generate_one(idx)
    predicted_masks = np.zeros((len(predicted_masks_list), crop_size[0], crop_size[1]))
    for i in range(10):
      predicted_masks[i] = predicted_masks_list[i][x_begin:(x_begin+crop_size[0]), y_begin:(y_begin+crop_size[1])]
    return predicted_masks

  def generate_one(self, idx):
    if not idx in self.preprocessed_images.keys() and self.train:
      predicted_masks = list()
      for i in range(10):
        predicted_masks.append(np.load(self.submission_dir + '/mask_' + self.image_list[idx] + '_' + str(i) + '.h5.npz')['arr_0'])
      if not self.store_processed_images:
        return predicted_masks
      self.preprocessed_images[idx] = predicted_masks
    return self.preprocessed_images[idx]

  def generate_cropped(self, idxs, crop_size):
    dataset_res = self.dataset.generate_cropped(idxs, crop_size)

    predicted_masks = np.zeros((len(idxs), 10, crop_size[0], crop_size[1]))
    i = 0
    for id in idxs:
      image_id = id[0]
      x_begin = id[1]
      y_begin = id[2]

      m = self.generate_one_cropped(image_id, crop_size, x_begin, y_begin)

      predicted_masks[i] = m
      i = i + 1
    # return [[images[:, 0:3, :, :], images[:, 3:, :, :]], [masks, not_is_zero], [weights, weights]]
    #if self.augmentation:
      #todo: perform augmentation on everything

    dataset_res.append(predicted_masks)
    return dataset_res


  def cropped_generator(self, chunk_size, crop_size=(448,448), subset="", overlapping_percentage=0):
    idx_values = self.dataset.get_generator_idxs(crop_size, subset=subset, overlapping_percentage=overlapping_percentage)
    while True:
      random.shuffle(idx_values)
      for i in range(len(idx_values) / chunk_size):
        idxs = idx_values[i * chunk_size:(i + 1) * chunk_size]
        gen = self.generate_cropped(idxs, crop_size)
        #print 'yielding'
        yield [[gen[0][0], gen[2]], gen[1]]



if __name__ == '__main__':
  import matplotlib.pyplot as plt

  r_generator = ResultGenerator(train=True, augmentation=False, normalize=True)
  generator = r_generator.cropped_generator(chunk_size=1, crop_size=(224*3,224*3), overlapping_percentage=0.2, subset='train')
  is_zero_counts = np.zeros([10])
  area = np.zeros([10])
  for i in range(100000):
    start_time = datetime.datetime.now().time().strftime('%H:%M:%S')
    images, gt_masks = generator.next()
    predicted_masks = images[1][0]
    a = ImageProcessor()
    images = a.image_for_display(images[0][0])
    tiff.imshow(np.transpose(images, [1, 2, 0]))
    plt.savefig(TEMP_IMAGE_DIR + '/img_' + str(i) + '.jpg')
    for j in range(0, 10):
      plt.imshow(predicted_masks[j])
      plt.savefig(TEMP_IMAGE_DIR + '/img_' + str(i) + '_m_' + str(j) + '_' + str(i) + '.jpg')

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
