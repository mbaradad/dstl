import caffe
import surgery, score
import utils.dirs as dirs

import numpy as np
import os
import sys
import utils.dirs as dirs

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

#weights = '../voc-fcn16s/voc-fcn16s.caffemodel'

# init
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.AdamSolver(dirs.FCN8 + '/solver.prototxt')
#solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = None
#To store the output
#output = dirs.FCN8_OUTPUT + '/val_output'
output = False

for i in range(750):
    print 'EPOCH: ' + str(i)
    solver.step(325)
    score.seg_tests(solver, output)
    solver.net.save(dirs.FCN8_OUTPUT + '/final_fcn8.caffemodel')
