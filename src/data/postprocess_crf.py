import sys
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

import skimage.io as io
from dataset import Dataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def post_process_crf(id):
  d = Dataset(train=False)
  image_list = d.get_image_list()
  id = image_list[0]
  d.generate_one(0)

  id = '6030_4_4'

  submission = '/home/manel/Documents/dstl/output/submissions/submission_2017-02-07_02:24:07.905275.csv'
  np_dir = submission.replace(".csv", "/")

  os.listdir(np_dir)

  predicted_maps = list()
  for i in range(10):
    predicted_maps.append(np.load(np_dir + 'mask_' + id + '_' + str(i) + '.h5.npy'))

  im, _ = d.generate_one(362)

  predicted_maps = np.resize(predicted_maps, [10, im.shape[1], im.shape[2]])

  im = im[:3]
  image = np.transpose(im, [1,2,0])
  unary = np.asarray(predicted_maps, dtype='float32').squeeze()

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
  feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

  d.addPairwiseEnergy(feats, compat=3,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

  # This creates the color-dependent features --
  # because the segmentation that we get from CNN are too coarse
  # and we can use local color features to refine them
  feats = create_pairwise_bilateral(sdims=(50, 50), schan=(900, 900, 900),
                                    img=image, chdim=2)

  d.addPairwiseEnergy(feats, compat=10,
                      kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
  Q = d.inference(5)

  res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

  cmap = plt.get_cmap('bwr')

  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
  ax1.set_title('Segmentation with CRF post-processing')
  #probability_graph = ax2.imshow(np.dstack((train_annotation,) * 3) * 100)
  #ax2.set_title('Ground-Truth Annotation')
  plt.show()
  a = 1
  plt.savefig('asdf.jpg')

if __name__ == "__main__":
  post_process_crf('6020_1_1')