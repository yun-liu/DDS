import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import gen_resnet as resnet

### Residual Convolution Uint
def RCU(bottom, num_output, kernel_size=3, pad=1, stride=1):
    relu1 = L.ReLU(bottom, in_place=True)
    conv1 = L.Convolution(relu1, num_output=num_output, kernel_size=kernel_size,
        pad=pad, stride=stride, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0),
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    relu2 = L.ReLU(conv1, in_place=False)
    conv2 = L.Convolution(relu2, num_output=num_output, kernel_size=kernel_size,
        pad=pad, stride=stride, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    joint = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    return relu1, conv1, relu2, conv2, joint


def DDS(phase='TRAIN', split='trainval'):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            batch_size=1, random=True, seed=1337, mirror=True, crop_size=352,
            resize=False)
    if phase == 'TRAIN':
        pydata_params['sbdd_dir'] = '../../data/SBD'
        pylayer = 'SBDDEdgeDataLayer'
    else:
        pydata_params['voc_dir'] = '../../data/VOC'
        pylayer = 'VOCSegDataLayer'
    n.data, n.label, n.label_thick = L.Python(module='data_layers', layer=pylayer, ntop=3,
        param_str=str(pydata_params))

    model = resnet.ResNet('', '')
    n = model.resnet_layers_proto(n=n)

    ########## loss5 ##########
    n.res5c_out_dimred = L.Convolution(n.res5c, num_output=512, kernel_size=3,
        pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    n.stage1_b1_prev_relu, n.stage1_b1_conv, n.stage1_b1_conv_relu, \
        n.stage1_b1_conv_relu_out_dimred, n.stage1_b1_joint = \
        RCU(n.res5c_out_dimred, 512)
    n.stage1_b2_prev_relu, n.stage1_b2_conv, n.stage1_b2_conv_relu, \
        n.stage1_b2_conv_relu_out_dimred, n.stage1_b2_joint = \
        RCU(n.stage1_b1_joint, 512)
    ########## loss_cls ##########
    n.pred_cls = L.Convolution(n.stage1_b2_joint, num_output=20, kernel_size=1,
        pad=0, stride=1, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=1,
        decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.pred_up_cls = L.Deconvolution(n.pred_cls, convolution_param=
        dict(num_output=20, kernel_size=16, stride=8, bias_term=False),
        param=[dict(lr_mult=0)])
    n.score_cls = L.Crop(n.pred_up_cls, n.data, crop_param=dict(axis=2, offset=4))
    if phase == 'TRAIN':
        n.loss_cls = L.MultiLabelLoss(n.score_cls, n.label_thick,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid_cls = L.Sigmoid(n.score_cls)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    ########## loss1 ##########
    n.conv1_dsn = L.Convolution(n.conv1, num_output=64, kernel_size=3,
        pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    n.conv1_b1_prev_relu, n.conv1_b1_conv, n.conv1_b1_conv_relu, \
        n.conv1_b1_conv_relu_out_dimred, n.conv1_b1_joint = \
        RCU(n.conv1_dsn, 64)
    n.conv1_b2_prev_relu, n.conv1_b2_conv, n.conv1_b2_conv_relu, \
        n.conv1_b2_conv_relu_out_dimred, n.conv1_b2_joint = \
        RCU(n.conv1_b1_joint, 64)
    n.score1 = L.Convolution(n.conv1_b2_joint, num_output=1, kernel_size=1,
        pad=0, stride=1, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=10,
        decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    if phase == 'TRAIN':
        n.loss1 = L.SigmoidCrossEntropyEdgeLoss(n.score1, n.label,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid1 = L.Sigmoid(n.score1)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    ########## loss2 ##########
    n.conv2_dsn = L.Convolution(n.res2c, num_output=128, kernel_size=3,
        pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    n.conv2_b1_prev_relu, n.conv2_b1_conv, n.conv2_b1_conv_relu, \
        n.conv2_b1_conv_relu_out_dimred, n.conv2_b1_joint = \
        RCU(n.conv2_dsn, 128)
    n.conv2_b2_prev_relu, n.conv2_b2_conv, n.conv2_b2_conv_relu, \
        n.conv2_b2_conv_relu_out_dimred, n.conv2_b2_joint = \
        RCU(n.conv2_b1_joint, 128)
    n.pred2 = L.Convolution(n.conv2_b2_joint, num_output=1, kernel_size=1,
        pad=0, stride=1, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=10,
        decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.pred_up2 = L.Deconvolution(n.pred2, convolution_param=
        dict(num_output=1, kernel_size=4, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])
    n.score2 = L.Crop(n.pred_up2, n.data, crop_param=dict(axis=2,offset=1))
    if phase == 'TRAIN':
        n.loss2 = L.SigmoidCrossEntropyEdgeLoss(n.score2, n.label,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid2 = L.Sigmoid(n.score2)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    ########## loss3 ##########
    n.conv3_dsn = L.Convolution(n.res3b3, num_output=128, kernel_size=3,
        pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    n.conv3_b1_prev_relu, n.conv3_b1_conv, n.conv3_b1_conv_relu, \
        n.conv3_b1_conv_relu_out_dimred, n.conv3_b1_joint = \
        RCU(n.conv3_dsn, 128)
    n.conv3_b2_prev_relu, n.conv3_b2_conv, n.conv3_b2_conv_relu, \
        n.conv3_b2_conv_relu_out_dimred, n.conv3_b2_joint = \
        RCU(n.conv3_b1_joint, 128)
    n.pred3 = L.Convolution(n.conv3_b2_joint, num_output=1, kernel_size=1,
        pad=0, stride=1, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=10,
        decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.pred_up3 = L.Deconvolution(n.pred3, convolution_param=
        dict(num_output=1, kernel_size=8, stride=4, bias_term=False),
        param=[dict(lr_mult=0)])
    n.score3 = L.Crop(n.pred_up3, n.data, crop_param=dict(axis=2,offset=2))
    if phase == 'TRAIN':
        n.loss3 = L.SigmoidCrossEntropyEdgeLoss(n.score3, n.label,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid3 = L.Sigmoid(n.score3)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    ########## loss4 ##########
    n.conv4_dsn = L.Convolution(n.res4b22, num_output=128, kernel_size=3,
        pad=1, stride=1, bias_term=False, weight_filler=dict(type='xavier'),
        param=dict(lr_mult=1, decay_mult=1))
    n.conv4_b1_prev_relu, n.conv4_b1_conv, n.conv4_b1_conv_relu, \
        n.conv4_b1_conv_relu_out_dimred, n.conv4_b1_joint = \
        RCU(n.conv4_dsn, 128)
    n.conv4_b2_prev_relu, n.conv4_b2_conv, n.conv4_b2_conv_relu, \
        n.conv4_b2_conv_relu_out_dimred, n.conv4_b2_joint = \
        RCU(n.conv4_b1_joint, 128)
    n.pred4 = L.Convolution(n.conv4_b2_joint, num_output=1, kernel_size=1,
        pad=0, stride=1, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=10,
        decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.pred_up4 = L.Deconvolution(n.pred4, convolution_param=
        dict(num_output=1, kernel_size=16, stride=8, bias_term=False),
        param=[dict(lr_mult=0)])
    n.score4 = L.Crop(n.pred_up4, n.data, crop_param=dict(axis=2,offset=4))
    if phase == 'TRAIN':
        n.loss4 = L.SigmoidCrossEntropyEdgeLoss(n.score4, n.label,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid4 = L.Sigmoid(n.score4)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    ########## concat ##########
    n.score_cls0, n.score_cls1, n.score_cls2, n.score_cls3, n.score_cls4, \
        n.score_cls5, n.score_cls6, n.score_cls7, n.score_cls8, n.score_cls9, \
        n.score_cls10, n.score_cls11, n.score_cls12, n.score_cls13, \
        n.score_cls14, n.score_cls15, n.score_cls16, n.score_cls17, \
        n.score_cls18, n.score_cls19 = L.Slice(n.score_cls,
        slice_param=dict(axis=1), ntop=20)
    n.concat = L.Concat(n.score_cls0, n.score4, n.score3, n.score2, n.score1,
            n.score_cls1, n.score4, n.score3, n.score2, n.score1,
            n.score_cls2, n.score4, n.score3, n.score2, n.score1,
            n.score_cls3, n.score4, n.score3, n.score2, n.score1,
            n.score_cls4, n.score4, n.score3, n.score2, n.score1,
            n.score_cls5, n.score4, n.score3, n.score2, n.score1,
            n.score_cls6, n.score4, n.score3, n.score2, n.score1,
            n.score_cls7, n.score4, n.score3, n.score2, n.score1,
            n.score_cls8, n.score4, n.score3, n.score2, n.score1,
            n.score_cls9, n.score4, n.score3, n.score2, n.score1,
            n.score_cls10, n.score4, n.score3, n.score2, n.score1,
            n.score_cls11, n.score4, n.score3, n.score2, n.score1,
            n.score_cls12, n.score4, n.score3, n.score2, n.score1,
            n.score_cls13, n.score4, n.score3, n.score2, n.score1,
            n.score_cls14, n.score4, n.score3, n.score2, n.score1,
            n.score_cls15, n.score4, n.score3, n.score2, n.score1,
            n.score_cls16, n.score4, n.score3, n.score2, n.score1,
            n.score_cls17, n.score4, n.score3, n.score2, n.score1,
            n.score_cls18, n.score4, n.score3, n.score2, n.score1,
            n.score_cls19, n.score4, n.score3, n.score2, n.score1,
            concat_param=dict(axis=1))
    n.score_fuse = L.Convolution(n.concat, num_output=20, kernel_size=1, pad=0,
        stride=1, group=20, bias_term=True, weight_filler=dict(type='xavier'),
        bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=10,
        decay_mult=1), dict(lr_mult=20, decay_mult=0)])

    if phase == 'TRAIN':
        n.loss_fuse = L.MultiLabelLoss(n.score_fuse, n.label_thick,
                loss_param=dict(ignore_label=255))
    elif phase == 'TEST':
        n.sigmoid_fuse = L.Sigmoid(n.score_fuse)
    else:
        raise Exception('phase should be either \'TRAIN\' or \'TEST\'')

    return n.to_proto()

if __name__ == '__main__':
    with open('ablation_studies/edge_train_sbd_DDS-R.prototxt', 'w') as f:
        f.write(str(DDS(phase='TRAIN', split='train')))
    with open('ablation_studies/edge_test_sbd_DDS-R.prototxt', 'w') as f:
        f.write(str(DDS(phase='TEST', split='val')))
