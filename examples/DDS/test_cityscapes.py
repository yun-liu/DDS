import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from skimage import color
from scale_process import scale_process

# Make sure that caffe is on the python path:
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')

import caffe


categories = ['road','sidewalk','building','wall','fence','pole','trafficlight','trafficsign',\
    'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

data_root = '../../data/Cityscapes/'
img_root = data_root + 'img/'

with open(data_root+'val.txt') as f:
    test_lst = f.readlines()
test_lst = [x.strip() for x in test_lst]

# remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
net = caffe.Net('edge_test_cityscapes.prototxt', 'pretrained_models/edge_cls_cityscapes_reweighted_loss_orig_data_iter_80000.caffemodel', caffe.TEST)

save_root = os.path.join(data_root, 'eval', 'res')
if not os.path.exists(save_root):
    os.mkdir(save_root)
for cls in categories:
    if not os.path.exists(os.path.join(save_root, cls)):
        os.mkdir(os.path.join(save_root, cls))


start_time = time.time()
for idx in range(len(test_lst)):
    # grid test
    im = Image.open(img_root+test_lst[idx]+'.png')
    label = scale_process(net, im, 500, 122.67891434, 116.66876762, 104.00698793)

    # save results
    for i, cls in enumerate(categories):
        edge = 255 * (1-label[i])
        cv2.imwrite(os.path.join(save_root, cls, test_lst[idx]+'.png'), edge)

diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))
