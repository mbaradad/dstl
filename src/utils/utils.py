from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import numpy as np

def resize(img, height, width):
  '''
  Resize a 3D array (image) to the size specified in parameters
  '''
  zoom_h = float(height) / img.shape[0]
  zoom_w = float(width) / img.shape[1]
  img = zoom(img, [zoom_h, zoom_w, 1], mode='nearest', order=0)
  return img

def imshow_th(image):
  plt.imshow(np.transpose(image,[1,2,0]))
def imshow_tf(image):
  plt.imshow(image)

def imshow_one_ch(image):
  imshow_th([image[0, :, :]] * 3)
