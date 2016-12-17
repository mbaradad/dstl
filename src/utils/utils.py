from scipy.ndimage.interpolation import zoom

def resize(img, height, width):
  '''
  Resize a 3D array (image) to the size specified in parameters
  '''
  zoom_h = float(height) / img.shape[0]
  zoom_w = float(width) / img.shape[1]
  img = zoom(img, [zoom_h, zoom_w, 1], mode='nearest', order=0)
  return img

