#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: filterlayer.py
@Time: 2022/3/30 21:07
@Overview:
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# https://github.com/mravanelli/SincNet
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        # 反变换到线性频率域
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        # the out_channels is 80 in the paper
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        # kernel_size will be 251 in the paper
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # 计算mel尺度下的滤波器参数，反变换回线性域
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        # 滤波器起始频率
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        # 滤波器宽度
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        # 时域滤波器
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1).abs()


# https://github.com/mravanelli/SincNet
class Sinc2Conv(nn.Module):
    def __init__(self, input_dim, out_dim=60, fs=16000):
        super(Sinc2Conv, self).__init__()
        self.fs = fs
        self.current_input = input_dim
        self.out_dim = out_dim

        # conv_layers = [(80, 251, 1), (60, 5, 1), (out_dim, 5, 1)]
        self.conv_layers = nn.ModuleList()
        self.sinc_conv = nn.Sequential(
            SincConv_fast(80, 251, self.fs, stride=6),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(80),  # nn.LayerNorm([80, int((self.current_input - 251 + 1) / 6 / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 251 + 1) / 6 / 3)
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(60),  # nn.LayerNorm([60, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 5 + 1) / 3)
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=self.out_dim, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.InstanceNorm1d(self.out_dim),  # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        self.current_output = int((self.current_input - 5 + 1) / 3)

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        x = self.sinc_conv(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        # x = self.conv_layer4(x)

        return x.transpose(1, 2)


class Sinc2Down(nn.Module):
    def __init__(self, input_dim, out_dim=60, fs=16000):
        super(Sinc2Down, self).__init__()
        self.fs = fs
        self.current_input = input_dim
        self.out_dim = out_dim

        # conv_layers = [(80, 251, 1), (60, 5, 1), (out_dim, 5, 1)]
        self.conv_layers = nn.ModuleList()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=80, kernel_size=251, stride=6),
            # SincConv_fast(80, 251, self.fs, stride=6),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(80),  # nn.LayerNorm([80, int((self.current_input - 251 + 1) / 6 / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 251 + 1) / 6 / 3)
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),  # nn.AvgPool1d(kernel_size=3),
            nn.InstanceNorm1d(60),  # nn.LayerNorm([60, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        self.current_input = int((self.current_input - 5 + 1) / 3)
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=self.out_dim, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=3),
            nn.InstanceNorm1d(self.out_dim),  # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
            nn.LeakyReLU(),
        )

        # self.conv_layer4 = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=self.out_dim, kernel_size=5, stride=2),
        #     nn.AvgPool1d(kernel_size=3),  # nn.MaxPool1d(kernel_size=3),
        #     nn.InstanceNorm1d(self.out_dim),  # nn.LayerNorm([self.out_dim, int((self.current_input - 5 + 1) / 3)]),
        #     nn.LeakyReLU(),
        # )

        self.current_output = int((self.current_input - 5 + 1) / 3)

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        # x = self.conv_layer4(x)

        return x.transpose(1, 2)


# https://github.com/pytorch/fairseq/blob/c47a9b2eef0f41b0564c8daf52cb82ea97fc6548/fairseq/models/wav2vec/wav2vec.py#L367
class Wav2Conv(nn.Module):
    def __init__(self, out_dim=512, log_compression=True):
        super(Wav2Conv, self).__init__()

        in_d = 1
        conv_layers = [(40, 10, 5), (200, 5, 4), (300, 3, 2), (512, 3, 2), (out_dim, 3, 2)]
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(self.block(in_d, dim, k, stride))
            in_d = dim
        self.tmp_gate = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        self.log_compression = log_compression
        # self.skip_connections = skip_connections
        # self.residual_scale = math.sqrt(residual_scale)

    def block(self, n_in, n_out, k, stride):
        return nn.Sequential(
            nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
            nn.InstanceNorm1d(n_out),  # nn.GroupNorm(1, n_out), in wav2spk replace group by instance normalization
            nn.ReLU(),
        )

    def forward(self, x):
        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:
            x = x.squeeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        tmp_gate = self.tmp_gate(x.transpose(1, 2)).transpose(1, 2)
        x = x * tmp_gate
        return x.transpose(1, 2)
