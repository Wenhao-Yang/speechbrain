#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ARET.py
@Time: 2021/10/20 16:27
@Overview:
"""
import torch  # noqa: F401
import torch.nn as nn
from torch import Tensor

from speechbrain.lobes.models.filterlayer import TimeFreqMaskLayer
from speechbrain.lobes.models.pooling import StatisticPooling, SelfAttentionPooling, SelfAttentionPooling_v2, \
    AttentionStatisticPooling, \
    AttentionStatisticPooling_v2
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear


def get_activation(activation):
    if activation == 'relu':
        nonlinearity = nn.ReLU
    elif activation in ['leakyrelu', 'leaky_relu']:
        nonlinearity = nn.LeakyReLU
    elif activation == 'prelu':
        nonlinearity = nn.PReLU

    return nonlinearity


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, time_len = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, time_len)

    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, time_len)

    return x


class TimeDelayLayer_v5(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu'):
        super(TimeDelayLayer_v5, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        self.kernel = nn.Conv1d(self.input_dim, self.output_dim, self.context_size, stride=self.stride,
                                padding=self.padding, dilation=self.dilation, groups=self.groups)

        if activation == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif activation in ['leakyrelu', 'leaky_relu']:
            self.nonlinearity = nn.LeakyReLU()
        elif activation == 'prelu':
            self.nonlinearity = nn.PReLU()

        self.bn = nn.BatchNorm1d(output_dim)

        # self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        # _, _, d = x.shape
        # assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
        #     self.input_dim, d)
        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.bn(x)

        return x.transpose(1, 2)


class TDNNBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='leakyrelu', **kwargs):
        super(TDNNBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inter_connect, context_size=3,
                                       stride=1, dilation=dilation, padding=1, activation=activation)
        # self.relu = nn.ReLU(inplace=True)
        self.tdnn2 = TimeDelayLayer_v5(input_dim=inter_connect, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1, activation=activation)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

        out += identity
        # out = self.relu(out)

        return out


class TDNNBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out).transpose(1, 2)

        out += identity
        out = self.act(out)
        # out = self.relu(out)

        return out


class TDCBAM(nn.Module):
    # input should be like [Batch, time, frequency]
    def __init__(self, inplanes, planes, time_freq='time', pooling='avg'):
        super(TDCBAM, self).__init__()
        self.time_freq = time_freq
        self.activation = nn.Sigmoid()
        self.pooling = pooling

        self.cov_t = nn.Conv2d(1, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.avg_t = nn.AdaptiveAvgPool2d((None, 1))
        self.max_t = nn.AdaptiveMaxPool2d((None, 1))

        self.cov_f = nn.Conv2d(1, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.avg_f = nn.AdaptiveAvgPool2d((1, None))
        self.max_f = nn.AdaptiveMaxPool2d((1, None))

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(1)

        t_output = self.avg_t(input)
        if self.pooling == 'both':
            t_output += self.max_t(input)

        t_output = self.cov_t(t_output)
        t_output = self.activation(t_output)
        t_output = input * t_output

        f_output = self.avg_f(input)
        if self.pooling == 'both':
            f_output += self.max_f(input)

        f_output = self.cov_f(f_output)
        f_output = self.activation(f_output)
        f_output = input * f_output

        output = (t_output + f_output) / 2

        if len(input.shape) == 4:
            output = output.squeeze(1)

        return output


class TDCBAM_v2(nn.Module):
    # input should be like [Batch, time, frequency]
    def __init__(self, inplanes, planes, time_freq='time', pooling='avg'):
        super(TDCBAM_v2, self).__init__()
        self.time_freq = time_freq
        self.activation = nn.Sigmoid()
        self.pooling = pooling

        self.cov_t = nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=3)
        self.cov_f = nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, input):
        t_output = input.mean(dim=2)

        t_output = self.cov_t(t_output.unsqueeze(1))
        t_output = self.activation(t_output)
        t_output = input * t_output.transpose(1, 2)

        f_output = input.mean(dim=1)
        f_output = self.cov_f(f_output.unsqueeze(1))
        f_output = self.activation(f_output)
        f_output = input * f_output

        output = (t_output + f_output) / 2

        return output


class TDNNCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, **kwargs):
        super(TDNNCBAMBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inter_connect, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=inter_connect, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)

        self.CBAM_layer = TDCBAM(planes, planes)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

        out = self.CBAM_layer(out)

        out += identity
        # out = self.relu(out)

        return out


class TDNNCBAMBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNCBAMBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.CBAM_layer = TDCBAM(planes, planes)

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)

        out = self.CBAM_layer(out).transpose(1, 2)
        out += identity
        out = self.act(out)

        return out


class TDNNCBAMBlock_v3(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNCBAMBlock_v3, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.CBAM_layer = TDCBAM_v2(planes, planes)

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)

        out = self.CBAM_layer(out).transpose(1, 2)
        out += identity
        out = self.act(out)

        return out


class SqueezeExcitation(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, reduction_ratio=2):
        super(SqueezeExcitation, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(inplanes, max(int(inplanes / self.reduction_ratio), 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(int(inplanes / self.reduction_ratio), 1), inplanes)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        scale = input.mean(dim=2)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale).unsqueeze(2)

        output = input * scale

        return output

    def __repr__(self):
        return "SqueezeExcitation(reduction_ratio=%f)" % self.reduction_ratio


class TDNNSEBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1,
                 activation='relu', reduction_ratio=2, **kwargs):
        super(TDNNSEBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()
        self.reduction_ratio = reduction_ratio

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.SE_layer = SqueezeExcitation(inplanes=planes, reduction_ratio=reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)

        out = self.SE_layer(out).transpose(1, 2)

        out += identity
        out = self.act(out)

        return out


class TDNNBottleBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1,
                 groups=32, activation='leayrelu', **kwargs):
        super(TDNNBottleBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm1d
        # width = int(planes * (base_width / 32.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inplanes, context_size=1,
                                       stride=1, dilation=dilation, activation=activation)
        # self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inplanes * 2, context_size=3,
                                       stride=1, dilation=dilation, padding=1, groups=groups, activation=activation)

        self.tdnn3 = TimeDelayLayer_v5(input_dim=inplanes * 2, output_dim=planes, context_size=1,
                                       stride=1, dilation=dilation, activation=activation)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out = self.tdnn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class TDNNBottleBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNBottleBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 1, stride=1,
                                      padding=0, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.tdnn3_kernel = nn.Conv1d(inter_connect, planes, 1, stride=1,
                                      padding=0, dilation=dilation, bias=False)
        self.tdnn3_bn = nn.BatchNorm1d(planes)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)  # .transpose(1, 2)
        out = self.act(out)

        out = self.tdnn3_kernel(out)
        out = self.tdnn3_bn(out).transpose(1, 2)

        out += identity
        out = self.act(out)
        # out = self.relu(out)

        return out


class ShuffleTDNNBlock(nn.Module):

    def __init__(self, inplanes=512, planes=512, context_size=3, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu', **kwargs) -> None:
        super(ShuffleTDNNBlock, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = inplanes
        self.output_dim = planes
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups
        self.activation = activation
        nonlinearity = get_activation(activation)

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = self.output_dim // 2
        assert (self.stride != 1) or (self.input_dim == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(self.input_dim, self.input_dim, kernel_size=3, stride=self.stride, padding=1,
                                    dilation=dilation),
                nn.BatchNorm1d(self.input_dim),
                nn.Conv1d(self.input_dim, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(branch_features),
                nonlinearity(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv1d(self.input_dim if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nonlinearity(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=self.context_size,
                                dilation=self.dilation, stride=self.stride,
                                padding=int((self.context_size - 1) * dilation / 2)),
            nn.BatchNorm1d(branch_features),
            nn.Conv1d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nonlinearity(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0,
                       dilation: int = 1, bias: bool = False) -> nn.Conv1d:
        return nn.Conv1d(i, o, kernel_size, stride, padding, dilation=dilation, bias=bias, groups=i)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out.transpose(1, 2)


class RET(nn.Module):
    def __init__(self, embedding_size, input_dim, alpha=0., input_norm='',
                 channels=[512, 512, 512, 512, 512, 1536], context=[5, 3, 3, 5], activation='leakyrelu',
                 downsample=None, resnet_size=17, dilation=[1, 1, 1, 1], stride=[1],
                 red_ratio=2, dropout_p=0.0, dropout_layer=False, encoder_type='SASP2', block_type='agg',
                 mask='None', mask_len=[5, 10], **kwargs):
        super(RET, self).__init__()
        # self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask
        self.channels = channels
        self.context = context
        self.activation = activation
        self.red_ratio = red_ratio

        tdnn_type = {14: [1, 1, 1, 0],
                     17: [1, 1, 1, 1],
                     21: [1, 1, 1, 1],
                     18: [2, 2, 2, 2],
                     34: [3, 4, 6, 3], }
        self.layers = tdnn_type[resnet_size] if resnet_size in tdnn_type else tdnn_type[17]
        self.stride = stride
        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])

        # if input_norm == 'Inst':
        #     self.inst_layer = nn.InstanceNorm1d(input_dim)
        # elif input_norm == 'Mean':
        #     self.inst_layer = Mean_Norm()
        # else:
        #     self.inst_layer = None

        if self.mask == "both":
            self.mask_layer = TimeFreqMaskLayer(mask_len=mask_len)
        else:
            self.mask_layer = None

        TDNN_layer = TimeDelayLayer_v5
        if block_type.lower() == 'basic':
            Blocks = TDNNBlock
        elif block_type.lower() == 'basic_v2':
            Blocks = TDNNBlock_v2
        elif block_type.lower() == 'shublock':
            Blocks = ShuffleTDNNBlock
        elif block_type.lower() == 'agg':
            Blocks = TDNNBottleBlock
        elif block_type.lower() == 'agg_v2':
            Blocks = TDNNBottleBlock_v2
        elif block_type.lower() == 'cbam':
            Blocks = TDNNCBAMBlock
        elif block_type.lower() == 'cbam_v2':
            TDNN_layer = TimeDelayLayer_v5
            Blocks = TDNNCBAMBlock_v2
        elif block_type.lower() == 'seblock_v2':
            Blocks = TDNNSEBlock_v2
        else:
            raise ValueError(block_type)

        self.frame1 = TDNN_layer(input_dim=self.input_dim, output_dim=self.channels[0],
                                 context_size=self.context[0], dilation=dilation[0], stride=self.stride[0],
                                 activation=self.activation)

        self.frame2 = Blocks(inplanes=self.channels[0], planes=self.channels[0],
                             downsample=downsample, dilation=1, activation=self.activation)

        self.frame4 = TDNN_layer(input_dim=self.channels[0], output_dim=self.channels[1],
                                 context_size=self.context[1], dilation=dilation[1], stride=self.stride[1],
                                 activation=self.activation)
        self.frame5 = Blocks(inplanes=self.channels[1], planes=self.channels[1],
                             downsample=downsample, dilation=1, activation=self.activation)

        self.frame7 = TDNN_layer(input_dim=self.channels[1], output_dim=self.channels[2],
                                 context_size=self.context[2], dilation=dilation[2], stride=self.stride[2],
                                 activation=self.activation)
        self.frame8 = Blocks(inplanes=self.channels[2], planes=self.channels[2],
                             downsample=downsample, dilation=1, activation=self.activation)

        if self.layers[3] != 0:
            self.frame10 = TDNN_layer(input_dim=self.channels[2], output_dim=self.channels[3],
                                      context_size=self.context[3], dilation=dilation[3], stride=self.stride[3],
                                      activation=self.activation)
            self.frame11 = Blocks(inplanes=self.channels[3], planes=self.channels[3],
                                  downsample=downsample, dilation=1, activation=self.activation)

        self.frame13 = TDNN_layer(input_dim=self.channels[3], output_dim=self.channels[4],
                                  context_size=1, dilation=1, activation=self.activation)
        self.frame14 = TDNN_layer(input_dim=self.channels[4], output_dim=self.channels[5],
                                  context_size=1, dilation=1, activation=self.activation)

        # self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[5])
        elif encoder_type in ['SASP', 'ASTP']:
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[5], hidden_dim=int(embedding_size / 2))
        elif encoder_type in ['SASP2', 'ASTP2']:
            self.encoder = AttentionStatisticPooling_v2(input_dim=self.channels[5], hidden_dim=int(embedding_size / 2))
        else:
            raise ValueError(encoder_type)

        # nonlinearity = get_activation(activation)

        self.segment1 = nn.Sequential(
            nn.BatchNorm1d(self.channels[5] * 2),
            nn.Linear(self.channels[5] * 2, embedding_size),
        )

        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v5):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nonlinear = 'leaky_relu' if self.activation == 'leakyrelu' else self.activation
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity=nonlinear)

    def forward(self, x):
        # pdb.set_trace()
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.mask_layer != None:
            x = self.mask_layer(x)

        # x = x.transpose(1, 2)
        x = self.frame1(x)
        x = self.frame2(x)

        x = self.frame4(x)
        x = self.frame5(x)

        x = self.frame7(x)
        x = self.frame8(x)

        if self.layers[3] != 0:
            x = self.frame10(x)
            x = self.frame11(x)

        x = self.frame13(x)
        x = self.frame14(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding = self.segment1(x)

        return embedding
