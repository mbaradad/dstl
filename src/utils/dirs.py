#The root folder is src, so all paths are relative to it
import os

INPUT = '../input'
SIXTEEN_BAND = INPUT + '/sixteen_band'
THREE_BAND = INPUT + '/three_band'
TRAIN_WKT = INPUT + '/train_wkt_v3.csv'
GRID_SIZES = INPUT + '/grid_sizes.csv'

PREPROCESSED_INPUT = '../preprocessed_input'
OUTPUT = '../output'
SUBMISSION = OUTPUT + '/submissions'

if os.getcwd().endswith('src'):
  if not os.path.isdir(INPUT):
    os.mkdir(INPUT)
  if not os.path.isdir(PREPROCESSED_INPUT):
    os.mkdir(PREPROCESSED_INPUT)
  if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)
  if not os.path.isdir(SUBMISSION):
    os.mkdir(SUBMISSION)