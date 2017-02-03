from utils.dirs import *
import pandas as pd
from shapely.wkt import loads
import numpy as np
from utils.utils import resize
from tifffile import tifffile
from utils.utils import normalize_coordinates
from PIL import Image, ImageDraw
import shapely.affinity
import cv2

'''
Classes of masks
1. Buildings
2. Misc. Manmade structures
3. Road
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike
'''

class ImageProcessor():
  def __init__(self, downsampling=1):
    self.downsampling = downsampling
    self.df_wkt = pd.read_csv(TRAIN_WKT)
    self.grid_sizes = pd.read_csv(GRID_SIZES)
    self.class_types = 10
  #TODO: possibly normalize images at this point or at other point
  #TODO: check whole precisison is preserved
  #TODO: everything is resized to image size, which is the maximum size, though some channels (i.e infrared) are of lower size.
  def get_images(self, name):
    images = tifffile.imread(THREE_BAND + '/' + name + '.tif')
    for i in 'A','M','P':
      image = tifffile.imread(SIXTEEN_BAND + '/' + name + '_' + i + '.tif')
      #resize is specified in the opposite order than np.shape!
      if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
      image = np.transpose(image, [1,2,0])
      image = resize(image, images[0].shape[0], images[0].shape[1])
      image = np.transpose(image, [2, 0, 1])
      images = np.append(images, image, axis=0)

    #TODO: Noramlize and substract mean at this point?
    return images


  # TODO: maybe do it more efficiently, without requiring pyplot
  # though this is only done once
  # also maybe store the masks directly to disk
  def get_masks(self, idx, height, width):

    polygonsList = {}
    image = self.df_wkt[self.df_wkt.ImageId == idx]
    masks = np.zeros([self.class_types, height, width], dtype="bool")
    #for cType in range(self.class_types):
    for cType in range(self.class_types):
      #TODO: CHECK if image has been previously preprocessed, and use that, once we know for sure that there are no errors
      polygonsList = loads(image[image.ClassType == (cType + 1)].MultipolygonWKT.values[0])

      if len(polygonsList) == 0:
        continue

      x_max = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == idx].iloc[0,1]
      y_min = self.grid_sizes[self.grid_sizes['Unnamed: 0'] == idx].iloc[0,2]

      polygonsList = shapely.affinity.scale(
        polygonsList, xfact=normalize_coordinates(1, 0, height, width, x_max, y_min, )[0],
        yfact=normalize_coordinates(0, 1, height, width, x_max, y_min, )[1],
        origin=(0, 0, 0))

      img = np.zeros((height, width), np.uint8)

      int_coords = lambda x: np.array(x).round().astype(np.int32)
      exteriors = [int_coords(poly.exterior.coords) for poly in polygonsList]
      interiors = [int_coords(pi.coords) for poly in polygonsList
                   for pi in poly.interiors]
      cv2.fillPoly(img, exteriors, 1)
      cv2.fillPoly(img, interiors, 0)

      masks[cType] = np.asarray(np.expand_dims(img, 0), dtype="bool")

    return masks

  def get_image_size(self, idx):
    return tifffile.imread(THREE_BAND + '/' + idx + '.tif').shape[1:]
