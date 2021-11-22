import cv2
import numpy as np

def caffe_process(net, im):
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[0][...] = im
    net.forward()
    fore1 = net.blobs['sigmoid_fuse'].data[0].copy()

    im_ = im[:,:,::-1]
    net.blobs['data'].reshape(1, *im_.shape)
    net.blobs['data'].data[0][...] = im_
    net.forward()
    fore2 = net.blobs['sigmoid_fuse'].data[0].copy()

    fore = (fore1 + fore2[:,:,::-1]) / 2.
    return fore

def caffe_process2(net, im):
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    net.forward()
    score = net.blobs['sigmoid_fuse'].data[0]
    return score.copy()

def pre_img(im, crop_size, mean_r, mean_g, mean_b):
    row = im.shape[0]
    col = im.shape[1]
    im_pad = im.astype(np.float32, copy=True)
    if im_pad.shape[2] < 3:
        im_r = im_pad
        im_g = im_pad
        im_b = im_pad
        im_pad = np.dstack((im_r, img_g, im_b))
    if row < crop_size:
        patch = np.zeros((crop_size-row,col,3), dtype=np.float32)
        patch[:,:,0] = mean_r
        patch[:,:,1] = mean_g
        patch[:,:,2] = mean_b
        im_pad = np.vstack((im_pad,patch))
    if col < crop_size:
        patch = np.zeros((im_pad.shape[0],crop_size-col,3), dtype=np.float32)
        patch[:,:,0] = mean_r
        patch[:,:,1] = mean_g
        patch[:,:,2] = mean_b
        im_pad = np.hstack((im_pad,patch))
    im_mean = np.zeros((crop_size,crop_size,3), dtype=np.float32)
    im_mean[:,:,0] = mean_r
    im_mean[:,:,1] = mean_g
    im_mean[:,:,2] = mean_b
    im_pad = im_pad - im_mean
    im_pad = im_pad[:,:,::-1]
    im_pad = im_pad.transpose((2,0,1))
    return im_pad

def scale_process(net, image, crop_size, mean_r, mean_g, mean_b):
    image = np.array(image)
    img_rows = image.shape[0]
    img_cols = image.shape[1]
    long_size = max(img_rows, img_cols)
    short_size = min(img_rows, img_cols)
    if long_size <= crop_size:
        input_data = pre_img(image, crop_size, mean_r, mean_g, mean_b)
        score = caffe_process(net, input_data)
        score = score[:,0:img_rows,0:img_cols]
    else:
        stride_rate = 2. / 3.
        stride = np.int(np.ceil(crop_size*stride_rate))
        img_pad = image.copy()
        if short_size < crop_size:
            if img_rows < crop_size:
                patch = np.zeros((crop_size-img_rows,img_cols,3), dtype=np.float32)
                patch[:,:,0] = mean_r
                patch[:,:,1] = mean_g
                patch[:,:,2] = mean_b
                img_pad = np.vstack((img_pad,patch))
            if img_cols < crop_size:
                patch = np.zeros((img_pad.shape[0],crop_size-img_cols,3), dtype=np.float32)
                patch[:,:,0] = mean_r
                patch[:,:,1] = mean_g
                patch[:,:,2] = mean_b
                img_pad = np.hstack((img_pad,patch))
        pad_rows = img_pad.shape[0]
        pad_cols = img_pad.shape[1]
        h_grid = np.int(np.ceil((pad_rows-crop_size)*1.0/stride) + 1)
        w_grid = np.int(np.ceil((pad_cols-crop_size)*1.0/stride) + 1)
        data_scale = np.zeros((19,pad_rows,pad_cols),dtype=np.float32)
        count_scale = np.zeros((19,pad_rows,pad_cols),dtype=np.float32)
        for grid_yidx in range(h_grid):
            for grid_xidx in range(w_grid):
                s_x = grid_xidx * stride
                s_y = grid_yidx * stride
                e_x = np.minimum(s_x + crop_size, pad_cols)
                e_y = np.minimum(s_y + crop_size, pad_rows)
                s_x = e_x - crop_size
                s_y = e_y - crop_size
                img_sub = img_pad[s_y:e_y,s_x:e_x,:].copy()
                count_scale[:,s_y:e_y,s_x:e_x] = count_scale[:,s_y:e_y,s_x:e_x] + 1
                input_data = pre_img(img_sub, crop_size, mean_r, mean_g, mean_b)
                data_scale[:,s_y:e_y,s_x:e_x] = data_scale[:,s_y:e_y,s_x:e_x] \
                    + caffe_process(net, input_data)
        score = data_scale / count_scale
        score = score[:,0:img_rows,0:img_cols]
    return score

def scale_process2(net, image, mean_r, mean_g, mean_b):
    image = np.array(image)
    input_data = image.astype(dtype=np.float32, copy=True)
    input_data = input_data[:,:,::-1]
    input_data -= np.array((mean_b,mean_g,mean_r))
    input_data = input_data.transpose((2,0,1))

    input_data_ = input_data[:,:,::-1]
    score1 = caffe_process2(net, input_data)
    score2 = caffe_process2(net, input_data_)
    score = (score1 + score2[:,:,::-1]) / 2.
    return score
