from data.dataset import Dataset
import time

if __name__ == "__main__":
  d = Dataset()

  start = time.clock()
  i = 0
  for j in d.generator(1):
    print "time elapsed for processing image " + str(i) + ": " + str(time.clock() - start)
    i += 1
    start = time.clock()
