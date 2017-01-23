from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    val_samples = 400
    for idx in range(val_samples):
        net.forward()
        #hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
        #                       net.blobs[layer].data[0],
        #                       n_cl)

        if save_dir and idx < 4:
            im = Image.fromarray(net.blobs['data'].data[0][0], mode='P')
            im.save(os.path.join(save_dir, 'idx_' + str(idx) + 'input' + '.png'))
            for j in range(10):
                im = Image.fromarray(net.blobs[layer].data[0][j] > 0.5, mode='P')
                im.save(os.path.join(save_dir, 'idx_' + str(idx) + '_idx_' + str(j) + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / val_samples

def seg_tests(solver, save_format, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, None, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    #acc = np.diag(hist).sum() / hist.sum()
    #print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    #acc = np.diag(hist) / hist.sum(1)
    #print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    #iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    #freq = hist.sum(1) / hist.sum()
    #print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
    #        (freq[freq > 0] * iu[freq > 0]).sum()
    #return hist
