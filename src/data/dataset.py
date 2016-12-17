from utils.dirs import *
import pandas as pd
import numpy as np

class Dataset():
  def __init__(self, train=True):
    #Place the unziped files at this path
    self.root_path = INPUT
    self.train = train

    #only those with annotations are training, it should be 22 images
    df = pd.read_csv('../input/train_wkt_v2.csv')
    train_images = df['ImageId'].unique()
    if self.train:
      self.image_list = np.asarray(train_images)
    else:
      #the opposite
      df1 = pd.read_csv('../input/grid_sizes.csv')
      all_images = df1['Unnamed: 0'].unique()
      self.image_list = [x for x in all_images if x not in np.asarray(train_images)]

  def generate_by_name(self, name):
    if name in self.image_list:
      return None

  def generate_one(self, idx):
    return None

  def generate_one(self, idx, crop_percentage, i, j):
    return None

  def crop_generator(self, crop_percentage, chunk_size):
    return None
  def generator(self, chunk_size):
    return None

