# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:22:50 2016

@author: ironbar

Function for dealing with topology exception errors
"""
import pandas as pd
import numpy as np
import time
from shapely.wkt import loads, dumps
import shapely.geometry
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon


def __get_valid_wkt_str(input, precision):
  """
  Function that checks that a wkt str is valid
  """
  if type(input) is str:
    wkt_str = input
  else:
    # Get the initial str
    wkt_str = dumps(input, rounding_precision=precision)
  # Loop until we find a valid polygon
  for i in range(100):
    polygon = loads(wkt_str)
    if not polygon.is_valid:
      # print debugging info
      # Use buffer to try to fix the polygon
      polygon = polygon.buffer(0)
      # Get back to multipolygon if necesary
      if polygon.type == 'Polygon':
        polygon = shapely.geometry.MultiPolygon([polygon])
      # Dump to str again
      wkt_str = dumps(polygon, rounding_precision=precision)
    else:
      return wkt_str
  raise Exception("Polygon couldn't be repaired in 100 iterations.")


def __create_square_around_point(point, side):
  """
  Creates a square polygon with shapely given
  the center point
  """
  # Create canonical square points
  square_points = np.zeros((4, 2))
  square_points[1, 0] = 1
  square_points[2] = 1
  square_points[3, 1] = 1
  # Scale the square
  square_points *= side
  # Position to have the point in the center
  for i in range(2):
    square_points[:, i] += point[i] - side / 2.
  pol = shapely.geometry.Polygon(square_points)
  return pol


def repair_topology_exception(submission_path, precision, image_id, n_class, point,
                              side=1e-4):
  """
  Tries to repair the topology exception error by creating a squared
  hole in the given point with the given side
  """
  start_time = time.time()
  # Load the submission
  print 'Loading the submission...'
  df = pd.read_csv(submission_path)
  # Loop over the values changing them
  # I'm going to use the values because I think iterating over the dataframe is slow
  # But I'm not sure
  print 'Looping over the polygons'
  polygons = df.MultipolygonWKT.values
  img_ids = df.ImageId.values
  class_types = df.ClassType.values
  for i, polygon, img_id, class_type in zip(range(len(polygons)), polygons,
                                            img_ids, class_types):
    print i
    #if img_id == image_id and n_class == class_type:
    polygon = shapely.wkt.loads(polygon)
    polygon = polygon.simplify(0.000008, preserve_topology=False)
    square = __create_square_around_point(point, side)
    processed_subpolygons = list()
    if type(polygon) == Polygon:
      p = polygon
      polygon = list()
      polygon.append(p)
    for subpolygon in polygon:
      processed_subpolygons.append(subpolygon.buffer(0))
      #processed_subpolygons.append(subpolygon.buffer(0).difference(square))
    polygon = MultiPolygon(processed_subpolygons)
    polygons[i] = __get_valid_wkt_str(polygon, precision)
  # Update the dataframe
  df.MultipolygonWKT = polygons
  # Save to a new file
  print 'Saving the submission...'
  df.to_csv(submission_path,
            index=False)
  print 'It took %i seconds to repair the submission' % (
    time.time() - start_time)


if __name__ == '__main__':
  repair_topology_exception('/home/manel/Documents/dstl/output/submissions/submission_2017-02-07_02:24:07.905275.csv',
                            precision=5,
                            image_id='6050_4_4',
                            n_class=1,
                            point=(0.002644, -0.003722),
                            side=1e-4)


