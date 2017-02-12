from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import numpy as np

def resize(img, height, width):
  '''
  Resize a 3D array (image) to the size specified in parameters
  '''
  zoom_h = float(height) / img.shape[0]
  zoom_w = float(width) / img.shape[1]
  img = zoom(img, [zoom_h, zoom_w, 1], mode='constant', order=0)
  return img

def imshow_th(image):
  plt.imshow(np.transpose(image,[1,2,0]))
def imshow_tf(image):
  plt.imshow(image)

def imshow_one_ch(image):
  imshow_th([image[0, :, :]] * 3)


#from https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/details/data-processing-tutorial
#coordinates are returned in the range (0, im_width), (0, im_height)
def normalize_coordinates(x, y, im_height, im_width, x_max, y_min):
  im_width = float(im_width)
  im_height = float(im_height)
  w_p = im_width*(im_width/(im_width+ 1))
  x_p = x*w_p/x_max

  h_p = im_height * (im_height / (im_height + 1))
  y_p = y*h_p/y_min
  return x_p, y_p

def normal_coordinates_to_dataset_coordinates(x_p, y_p, im_height, im_width, x_max, y_min):
  im_width = float(im_width)
  im_height = float(im_height)

  w_p = im_width*(im_width/(im_width+ 1))
  x = x_p*x_max/w_p

  h_p = im_height * (im_height / (im_height + 1))
  y = y_p*y_min/h_p
  return x, y


