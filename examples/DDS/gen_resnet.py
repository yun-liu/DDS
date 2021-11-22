import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P

def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, pad=1, stride=1,
    dilation=1, conv_param=1, bn_param=[0,0,0], scale_param=[0,0]):

    if dilation == 1:
        conv = L.Convolution(bottom, num_output=num_output, bias_term=False,
                kernel_size=kernel_size, pad=pad, stride=stride,
                param=[dict(lr_mult=conv_param)])
    else:
        conv = L.Convolution(bottom, num_output=num_output, bias_term=False,
                kernel_size=kernel_size, pad=pad, stride=stride,
                dilation=dilation, param=[dict(lr_mult=conv_param)])
    conv_bn = L.BatchNorm(conv, in_place=True, use_global_stats=True,
            param=[dict(lr_mult=bn_param[0]), dict(lr_mult=bn_param[1]),
                dict(lr_mult=bn_param[2])])
    conv_scale = L.Scale(conv, in_place=True, bias_term=True,
            param=[dict(lr_mult=scale_param[0]), dict(lr_mult=scale_param[1])])
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu

def conv_bn_scale(bottom, num_output=64, kernel_size=1, pad=0, stride=1,
    conv_param=1, bn_param=[0,0,0], scale_param=[0,0]):

    conv = L.Convolution(bottom, num_output=num_output, bias_term=False,
            kernel_size=kernel_size, pad=pad, stride=stride,
            param=[dict(lr_mult=conv_param)])
    conv_bn = L.BatchNorm(conv, in_place=True, use_global_stats=True,
            param=[dict(lr_mult=bn_param[0]), dict(lr_mult=bn_param[1]),
                dict(lr_mult=bn_param[2])])
    conv_scale = L.Scale(conv, in_place=True, bias_term=True,
            param=[dict(lr_mult=scale_param[0]), dict(lr_mult=scale_param[1])])

    return conv, conv_bn, conv_scale

def eltwize_relu(bottom1, bottom2):

    residual_eltwise = L.Eltwise(bottom1, bottom2, operation=P.Eltwise.SUM)
    residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True)

    return residual_eltwise, residual_eltwise_relu

def residual_branch(bottom, base_output=64, dilation=1):

    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(
            bottom, num_output=base_output, kernel_size=1, pad=0, stride=1)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3,
            pad=dilation, stride=1, dilation=dilation)
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b,
            num_output=4*base_output)
    residual, residual_relu = eltwize_relu(bottom, branch2c)

    return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, branch2b, \
        branch2b_bn, branch2b_scale, branch2b_relu, branch2c, branch2c_bn, \
        branch2c_scale, residual, residual_relu

def residual_branch_shortcut(bottom, stride=2, base_output=64, dilation=1):

    branch1, branch1_bn, branch1_scale = conv_bn_scale(bottom,
            num_output=4*base_output, kernel_size=1, pad=0, stride=stride)
    branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(
            bottom, num_output=base_output, kernel_size=1, pad=0, stride=stride)
    branch2b, branch2b_bn, branch2b_scale, branch2b_relu = \
        conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3,
            pad=dilation, stride=1, dilation=dilation)
    branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b,
            num_output=4*base_output)
    residual, residual_relu = eltwize_relu(branch1, branch2c)

    return branch1, branch1_bn, branch1_scale, branch2a, branch2a_bn, \
        branch2a_scale, branch2a_relu, branch2b, branch2b_bn, branch2b_scale, \
        branch2b_relu, branch2c, branch2c_bn, branch2c_scale, residual, \
        residual_relu

branch_shortcut_string = 'n.res(stage)a_branch1, n.bn(stage)a_branch1, \
        n.scale(stage)a_branch1, n.res(stage)a_branch2a, n.bn(stage)a_branch2a, \
        n.scale(stage)a_branch2a, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.bn(stage)a_branch2b, n.scale(stage)a_branch2b, \
        n.res(stage)a_branch2b_relu, n.res(stage)a_branch2c, n.bn(stage)a_branch2c, \
        n.scale(stage)a_branch2c, n.res(stage)a, n.res(stage)a_relu = \
        residual_branch_shortcut((bottom), stride=(stride), base_output=(num), \
        dilation=(dilation))'

branch_string = 'n.res(stage)b(order)_branch2a, n.bn(stage)b(order)_branch2a, \
        n.scale(stage)b(order)_branch2a, n.res(stage)b(order)_branch2a_relu, \
        n.res(stage)b(order)_branch2b, n.bn(stage)b(order)_branch2b, \
        n.scale(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_relu, \
        n.res(stage)b(order)_branch2c, n.bn(stage)b(order)_branch2c, \
        n.scale(stage)b(order)_branch2c, n.res(stage)b(order), \
        n.res(stage)b(order)_relu = residual_branch((bottom), \
        base_output=(num), dilation=(dilation))'

branch_second_string = 'n.res(stage)b_branch2a, n.bn(stage)b_branch2a, \
        n.scale(stage)b_branch2a, n.res(stage)b_branch2a_relu, \
        n.res(stage)b_branch2b, n.bn(stage)b_branch2b, n.scale(stage)b_branch2b, \
        n.res(stage)b_branch2b_relu, n.res(stage)b_branch2c, n.bn(stage)b_branch2c, \
        n.scale(stage)b_branch2c, n.res(stage)b, n.res(stage)b_relu = \
        residual_branch((bottom), base_output=(num), dilation=(dilation))'

branch_third_string = 'n.res(stage)c_branch2a, n.bn(stage)c_branch2a, \
        n.scale(stage)c_branch2a, n.res(stage)c_branch2a_relu, \
        n.res(stage)c_branch2b, n.bn(stage)c_branch2b, n.scale(stage)c_branch2b, \
        n.res(stage)c_branch2b_relu, n.res(stage)c_branch2c, n.bn(stage)c_branch2c, \
        n.scale(stage)c_branch2c, n.res(stage)c, n.res(stage)c_relu = \
        residual_branch((bottom), base_output=(num), dilation=(dilation))'


class ResNet(object):
    def __init__(self, lmdb_train, lmdb_test):
        self.train_data = lmdb_train
        self.test_data = lmdb_test

    def resnet_layers_proto(self, n, batch_size=1, phase='TRAIN',
            stages=(3, 4, 23, 3)):
        """
            (3, 4, 6, 3) for 50 layers;
            (3, 4, 23, 3) for 101 layers;
            (3, 8, 36, 3) for 152 layers
        """
        #n = caffe.NetSpec()
        #if phase == 'TRAIN':
        #    n.data, n.label = L.Data(source=self.train_data, backend=P.Data.LMDB,
        #        batch_size=batch_size, ntop=2, include=dict(phase=0),
        #        transform_param=dict(crop_size=224, mean_value=[104, 117, 123],
        #        mirror=True))
        #else:
        #    n.data, n.label = L.Data(source=self.test_data, backend=P.Data.LMDB,
        #        batch_size=batch_size, ntop=2, include=dict(phase=1),
        #        transform_param=dict(crop_size=224, mean_value=[104, 117, 123],
        #        mirror=False))

        n.conv1, n.bn_conv1, n.scale_conv1, n.conv1_relu = conv_bn_scale_relu(
            n.data, num_output=64, kernel_size=7, stride=1, pad=3)
        n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)

        for num in xrange(len(stages)):
            for i in xrange(stages[num]):
                if i == 0:
                    stage_string = branch_shortcut_string
                    bottom_string = ['n.pool1', 'n.res2c', 'n.res3b%s' %
                        str(stages[1] - 1), 'n.res4b%s' % str(stages[2] - 1)][num]
                else:
                    stage_string = branch_string
                    if i == 1:
                        if num == 0 or num == 3:
                            stage_string = branch_second_string
                        bottom_string = 'n.res%sa' % str(num + 2)
                    else:
                        if num == 0 or num == 3:
                            stage_string = branch_third_string
                            bottom_string = 'n.res%sb' % str(num + 2)
                        else:
                            bottom_string = 'n.res%sb%s' % (str(num + 2),
                                str(i - 1))

                exec (stage_string.replace('(stage)', str(num + 2)).
                    replace('(bottom)', bottom_string).
                    replace('(num)', str(2 ** num * 64)).
                    replace('(order)', str(i)).
                    replace('(stride)', str(int(num > 0 and num < 3) + 1)).
                    replace('(dilation)', str(int(num == 3) * 2 + 2)))

        #exec 'n.pool5 = L.Pooling((bottom), pool=P.Pooling.AVE, global_pooling=True)'.
        #    replace('(bottom)', 'n.res5c')
        #n.fc1000 = L.InnerProduct(n.pool5, num_output=self.classifier_num)
        #n.prob = L.Softmax(n.fc1000)
        #if phase == 'TEST':
        #    n.accuracy_top1 = L.Accuracy(n.fc1000, n.label, include=dict(phase=1))
        #    n.accuracy_top5 = L.Accuracy(n.fc1000, n.label, include=dict(phase=1),
        #        accuracy_param=dict(top_k=5))

        return n

#def save_proto(proto, prototxt):
#    with open(prototxt, 'w') as f:
#        f.write(str(proto))

#if __name__ == '__main__':
#    model = ResNet('/data/train_lmbd', 'data/test_lmbd', 1000)
#    train_proto = model.resnet_layers_proto(64)
#    test_proto = model.resnet_layers_proto(64, phase='TEST')
#    save_proto(train_proto, 'proto/train.prototxt')
#    save_proto(test_proto, 'proto/test.prototxt')
