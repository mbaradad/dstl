from data.dataset import Dataset
import time

if __name__ == "__main__":
  d = Dataset(subset=1)


  start = time.clock()
  i = 0
  for j in d.cropped_generator(16, 20):
    print "time elapsed for processing image " + str(i) + ": " + str(time.clock() - start)
    print j[0].shape
    i += 1
    start = time.clock()
  start = time.clock()
  i = 0
  for j in d.generator(1):
    print "time elapsed for processing image " + str(i) + ": " + str(time.clock() - start)
    print j[0].shape
    i += 1
    start = time.clock()