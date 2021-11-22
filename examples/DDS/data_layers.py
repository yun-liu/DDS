import caffe

import numpy as np
from PIL import Image
import cv2
import random

class SBDDEdgeDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic boundary
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a convolutional neural network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        - mirror: flip the images randomly
        - crop_size: crop size for images
        - resize: randomly resize images

        for SBDD semantic boundary.

        example

        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.mirror = params.get('mirror', True)
        self.crop_size = params.get('crop_size', 0)
        self.resize = params.get('resize', True)

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data, label \
                and label_thick.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.indices)

        if self.resize:
            self.scales = np.array([0.5, 0.75, 1.0, 1.25, 1.5])


    def reshape(self, bottom, top):
        # load image + label image pair
        flip = self.mirror and (np.random.randint(2) == 1)
        self.data, self.label, self.label_thick = \
            self.load_batch(self.indices[self.idx], flip)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.label_thick.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.label_thick

        # pick next input
        self.idx += 1
        if self.idx >= len(self.indices):
            if self.random:
                random.shuffle(self.indices)
            self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_batch(self, idx, flip):
        """
        Load input image and corresponding label map
        """
        im = cv2.imread('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        if self.resize:
            scale = self.scales[np.random.randint(5)]
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=
                    cv2.INTER_LINEAR)
        in_ = im.astype(dtype=np.float32)
        pad_height = max(self.crop_size - in_.shape[0], 0)
        pad_width = max(self.crop_size - in_.shape[1], 0)
        if pad_height > 0 or pad_width > 0:
            in_ = cv2.copyMakeBorder(in_, 0, pad_height, 0, pad_width,
                    cv2.BORDER_CONSTANT, self.mean)

        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        mat = mat['GTcls'][0]['Boundaries'][0]
        gt_ = np.empty((mat.shape[0], in_.shape[0], in_.shape[1]), dtype=np.uint8)
        for i in range(mat.shape[0]):
            tmp = mat[i, 0].toarray()
            if self.resize:
                tmp = cv2.resize(tmp, None, fx=scale, fy=scale, interpolation=
                        cv2.INTER_NEAREST)
            if pad_height > 0 or pad_width > 0:
                tmp = cv2.copyMakeBorder(tmp, 0, pad_height, 0, pad_width,
                        cv2.BORDER_CONSTANT, 255)
            gt_[i] = tmp

        mat = scipy.io.loadmat('{}/cls_thick/{}.mat'.format(self.sbdd_dir, idx))
        mat = mat['GTcls'][0]['Boundaries'][0]
        gt2_ = np.empty((mat.shape[0], in_.shape[0], in_.shape[1]), dtype=np.uint8)
        for i in range(mat.shape[0]):
            tmp = mat[i, 0].toarray()
            if self.resize:
                tmp = cv2.resize(tmp, None, fx=scale, fy=scale, interpolation=
                        cv2.INTER_NEAREST)
            if pad_height > 0 or pad_width > 0:
                tmp = cv2.copyMakeBorder(tmp, 0, pad_height, 0, pad_width,
                        cv2.BORDER_CONSTANT, 255)
            gt2_[i] = tmp

        if self.crop_size > 0:
            h_off = np.random.randint(in_.shape[0] - self.crop_size + 1)
            w_off = np.random.randint(in_.shape[1] - self.crop_size + 1)
            in_ = in_[h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size),:]
            gt_ = gt_[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
            gt2_ = gt2_[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
        if flip:
            in_ = in_[:,::-1,:]
            gt_ = gt_[:,:,::-1]
            gt2_ = gt2_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_, gt_, gt2_


class SBDDEdgeDataLayerPreLoad(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.mirror = params.get('mirror', True)
        self.crop_size = params.get('crop_size', 0)
        self.resize = params.get('resize', True)

        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data, label \
                and label_thick.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        self.order = np.array(range(len(self.indices)))

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.order)

        if self.resize:
            self.scales = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

        self.imgs = []
        for i in range(len(self.indices)):
            with open('{}/img/{}.jpg'.format(self.sbdd_dir, self.indices[i]), 'rb') as fid:
                self.imgs.append(np.fromstring(fid.read(), np.uint8))


    def reshape(self, bottom, top):
        # load image + label image pair
        flip = self.mirror and (np.random.randint(2) == 1)
        self.data, self.label, self.label_thick = self.load_batch(self.idx, flip)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.label_thick.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.label_thick

        # pick next input
        self.idx += 1
        if self.idx >= len(self.indices):
            if self.random:
                random.shuffle(self.order)
            self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_batch(self, idx, flip):
        """
        Load input image and corresponding label map
        """
        im = cv2.imdecode(self.imgs[self.order[idx]], cv2.IMREAD_COLOR)
        if self.resize:
            scale = self.scales[np.random.randint(5)]
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=
                    cv2.INTER_LINEAR)
        in_ = im.astype(dtype=np.float32)
        pad_height = max(self.crop_size - in_.shape[0], 0)
        pad_width = max(self.crop_size - in_.shape[1], 0)
        if pad_height > 0 or pad_width > 0:
            in_ = cv2.copyMakeBorder(in_, 0, pad_height, 0, pad_width,
                    cv2.BORDER_CONSTANT, self.mean)

        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, self.indices[self.order[idx]]))
        mat = mat['GTcls'][0]['Boundaries'][0]
        gt_ = np.empty((mat.shape[0], in_.shape[0], in_.shape[1]), dtype=np.uint8)
        for i in range(mat.shape[0]):
            tmp = mat[i, 0].toarray()
            if self.resize:
                tmp = cv2.resize(tmp, None, fx=scale, fy=scale, interpolation=
                        cv2.INTER_NEAREST)
            if pad_height > 0 or pad_width > 0:
                tmp = cv2.copyMakeBorder(tmp, 0, pad_height, 0, pad_width,
                        cv2.BORDER_CONSTANT, 255)
            gt_[i] = tmp

        mat = scipy.io.loadmat('{}/cls_thick/{}.mat'.format(self.sbdd_dir, self.indices[self.order[idx]]))
        mat = mat['GTcls'][0]['Boundaries'][0]
        gt2_ = np.empty((mat.shape[0], in_.shape[0], in_.shape[1]), dtype=np.uint8)
        for i in range(mat.shape[0]):
            tmp = mat[i, 0].toarray()
            if self.resize:
                tmp = cv2.resize(tmp, None, fx=scale, fy=scale, interpolation=
                        cv2.INTER_NEAREST)
            if pad_height > 0 or pad_width > 0:
                tmp = cv2.copyMakeBorder(tmp, 0, pad_height, 0, pad_width,
                        cv2.BORDER_CONSTANT, 255)
            gt2_[i] = tmp

        if self.crop_size > 0:
            h_off = np.random.randint(in_.shape[0] - self.crop_size + 1)
            w_off = np.random.randint(in_.shape[1] - self.crop_size + 1)
            in_ = in_[h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size),:]
            gt_ = gt_[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
            gt2_ = gt2_[:,h_off:(h_off+self.crop_size),w_off:(w_off+self.crop_size)]
        if flip:
            in_ = in_[:,::-1,:]
            gt_ = gt_[:,:,::-1]
            gt2_ = gt2_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_, gt_, gt2_
