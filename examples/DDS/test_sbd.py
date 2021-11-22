import os
import sys
import time
import cv2
import numpy as np
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe


categories = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow', \
    'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

data_root = '../../data/SBD/'
img_root = data_root + 'img/'

with open(data_root+'val.txt') as f:
    test_lst = f.readlines()
test_lst = [x.strip() for x in test_lst]

# remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('edge_test_sbd.prototxt', 'pretrained_models/edge_cls_sbd_reweighted_loss_orig_data_iter_25000.caffemodel', caffe.TEST)

save_root = os.path.join(data_root, 'eval', 'res')
if not os.path.exists(save_root):
    os.mkdir(save_root)
for cls in categories:
    if not os.path.exists(os.path.join(save_root, cls)):
        os.mkdir(os.path.join(save_root, cls))

start_time = time.time()
for idx in range(len(test_lst)):
    im = cv2.imread(img_root+test_lst[idx]+'.jpg')
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2, 0, 1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    # get results
    label = net.blobs['sigmoid_fuse'].data[0].copy()

    # save results
    for i, cls in enumerate(categories):
        edge = 255 * (1-label[i])
        cv2.imwrite(os.path.join(save_root, cls, test_lst[idx]+'.png'), edge)

diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))
