#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: pooling.py
@Time: 2022/4/11 11:48
@Overview:
"""

import torch
import torch.nn as nn


class StatisticPooling(nn.Module):

    def __init__(self, input_dim):
        super(StatisticPooling, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        if len(x.shape) != 3:
            x = x.reshape(x_shape[0], x_shape[-2], -1)

        assert x.shape[-1] == self.input_dim, print(x.shape[-1])

        mean_x = x.mean(dim=1)
        std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.attention_soft = nn.Tanh()

    def forward(self, x):
        """
        :param x:   [batch, length, feat_dim] vector
        :return:   [batch, feat_dim] vector
        """
        x_shape = x.shape
        if len(x.shape) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x_shape[0], x_shape[2], -1)
        assert x.shape[-1] == self.input_dim

        fx = self.attention_activation(self.attention_linear(x))
        vf = fx.matmul(self.attention_vector)
        alpha = self.attention_soft(vf)

        alpha_ht = x.mul(alpha)
        mean = torch.sum(alpha_ht, dim=-2)

        return mean


class SelfAttentionPooling_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionPooling_v2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.Tanh = nn.Tanh()
        self.attention_vector = nn.Linear(self.hidden_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x:   [batch, length, feat_dim] vector
        :return:   [batch, feat_dim] vector
        """
        x_shape = x.shape
        if len(x.shape) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x_shape[0], x_shape[2], -1)

        assert x.shape[-1] == self.input_dim, print(x.shape[-1], self.input_dim)

        alpha = self.Tanh(self.attention_linear(x))
        alpha = self.softmax(self.attention_vector(alpha))

        mean = torch.sum(alpha * x, dim=1)

        return mean


class AttentionStatisticPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionStatisticPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.attention_soft = nn.Tanh()

    def forward(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        if len(x_shape) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x_shape[0], x_shape[2], -1)

        assert x.shape[-1] == self.input_dim

        fx = self.attention_activation(self.attention_linear(x))
        vf = fx.matmul(self.attention_vector)
        alpha = self.attention_soft(vf)

        alpha_ht = x.mul(alpha)
        mean = torch.sum(alpha_ht, dim=-2)

        # pdb.set_trace()
        sigma_power = torch.sum(torch.pow(x, 2).mul(alpha), dim=-2) - torch.pow(mean, 2)
        # alpha_ht_ht = x*x.mul(alpha)
        sigma = torch.sqrt(sigma_power.clamp(min=1e-12))

        mean_sigma = torch.cat((mean, sigma), 1)

        return mean_sigma


class AttentionStatisticPooling_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionStatisticPooling_v2, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.attention_vector = nn.Linear(self.hidden_dim, input_dim)
        self.Tanh = nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x:   [batch, channels, length, feat_dim] or [batch, length, feat_dim]
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        if len(x_shape) == 4:
            x = x.transpose(1, 2)
            x = x.reshape(x_shape[0], x_shape[2], -1)

        assert x.shape[-1] == self.input_dim

        alpha = self.Tanh(self.attention_linear(x))
        alpha = self.softmax(self.attention_vector(alpha))

        mean = torch.sum(alpha * x, dim=1)

        # pdb.set_trace()
        residuals = torch.sum(alpha * x ** 2, dim=1) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))

        mean_sigma = torch.cat((mean, std), 1)

        return mean_sigma
