import sys
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

import skimage.io as io
from dataset import Dataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.dirs import TEMP_IMAGE_DIR
from data.image_processor import ImageProcessor
import tifffile as tiff
from utils.utils import savefig


def post_process_crf():
  d = Dataset(train=False)
  image_list = d.get_image_list()

  id = '6170_3_3'

  np_dir = '/mnt/sdd1/submissions/submission_2017-02-18_13:11:59.298593/'

  #os.listdir(np_dir)

  predicted_maps = list()
  for i in range(10):
    actual_mask = np.load(np_dir + 'mask_' + id + '_' + str(i) + '.h5.npz')['arr_0']
    predicted_maps.append(actual_mask)
    savefig(actual_mask > 0.5, 'predicted_' + str(i))


  im, _ = d.generate_one(list(image_list).index(id))

  predicted_maps = np.resize(predicted_maps, [10, im.shape[1], im.shape[2]])

  im = im[:3]

  image = np.transpose(im, [1,2,0])

  #image = a.image_for_display(image)
  savefig(image, 'image')

  unary = np.asarray(predicted_maps, dtype='float32').squeeze()

  #tiff.imshow(image)
  #plt.savefig(TEMP_IMAGE_DIR + '/fig.png')

  #softmax = processed_probabilities.transpose((2, 0, 1))


  # The input should be the negative of the logarithm of probability values
  # Look up the definition of the softmax_to_unary for more information
  #unary = softmax_to_unary(processed_probabilities)

  # The inputs should be C-continious -- we are using Cython wrapper
  unary = np.ascontiguousarray(np.reshape(unary, [10, unary.shape[1]*unary.shape[2]]))

  d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 10)

  d.setUnaryEnergy(unary)

  # This potential penalizes small pieces of segmentation that are
  # spatially isolated -- enforces more spatially consistent segmentations
  feats = create_pairwise_gaussian(sdims=(50, 50), shape=image.shape[:2])

  d.addPairwiseEnergy(feats,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

  # This creates the color-dependent features --
  # because the segmentation that we get from CNN are too coarse
  # and we can use local color features to refine them
  feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                    img=image, chdim=2)

  d.addPairwiseEnergy(feats, compat=10,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
  Q = d.inference(5)

  res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
  map = np.asarray(Q).reshape((10, image.shape[0], image.shape[1]))

  #cmap = plt.get_cmap('bwr')

  #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

  #ax1.set_title('Segmentation with CRF post-processing')
  #probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
  #ax2.set_title('Ground-Truth Annotation')
  #a = 1
  plt.imshow(predicted_maps[5])
  plt.savefig(TEMP_IMAGE_DIR + '/original.jpg')
  plt.imshow(map)
  plt.savefig(TEMP_IMAGE_DIR + '/processed.jpg')
  plt.imshow(res == 8)
  plt.savefig(TEMP_IMAGE_DIR + '/im2.jpg')

if __name__ == "__main__":
  post_process_crf()