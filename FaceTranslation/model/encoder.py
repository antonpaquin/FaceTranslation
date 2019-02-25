import torch
import torch.nn as nn


class KeyedBnReluConvConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(KeyedBnReluConvConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels + 1,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels + 1,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(stride, stride)
        )

    def forward(self, x, enc_key):
        shaped_enc_key = enc_key.view([enc_key.shape[0], 1, 1, 1]).expand([-1, -1, x.shape[2], x.shape[3]])
        x = self.bn(x)
        x = self.relu(x)
        x = torch.cat((x, shaped_enc_key), 1)
        x = self.conv1(x)
        x = torch.cat((x, shaped_enc_key), 1)
        x = self.conv2(x)
        return x


class EncodingStep(nn.Module):
    def __init__(self, depth_in, scalebias_per_step=1, terminating=False):
        super(EncodingStep, self).__init__()
        self.terminating = terminating
        self.brcc_scale = KeyedBnReluConvConv(depth_in, scalebias_per_step * depth_in)
        self.brcc_bias = KeyedBnReluConvConv(depth_in, scalebias_per_step * depth_in)
        if not terminating:
            self.brcc_cont = KeyedBnReluConvConv(depth_in, 2*depth_in, 2)

    def forward(self, x, enc_key):
        scale = self.brcc_scale(x, enc_key)
        bias = self.brcc_bias(x, enc_key)
        if self.terminating:
            return (scale, bias)
        cont = self.brcc_cont(x, enc_key)
        return (scale, bias), cont


class Encoder(nn.Module):
    def __init__(self, scalebias_per_step):
        super(Encoder, self).__init__()
        self.steps = []
        dim = 256
        depth = 1
        while dim >= 8:
            step = EncodingStep(depth, scalebias_per_step)
            self.steps.append(step)
            self.add_module('step_{}'.format(dim), step)
            depth *= 2
            dim //= 2
        step = EncodingStep(depth, scalebias_per_step, terminating=True)
        self.steps.append(step)
        self.add_module('step_{}'.format(dim), step)

    def forward(self, x, enc_key):
        encoded = []
        for step in self.steps[:-1]:
            e, x = step(x, enc_key)
            encoded.append(e)
        encoded.append(self.steps[-1](x, enc_key))

        return tuple(encoded)
