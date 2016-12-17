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

  def get_idx(self, idx):

  def get_by_name(self, name):
    if name in self.image_list:

  def generate_one(self):

  def crop_generator(self, crop_percentage, chunk_size):

  def generator(self, chunk_size):


