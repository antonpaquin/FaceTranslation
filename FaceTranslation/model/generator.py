import torch
import torch.nn as nn


class BnReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(BnReluConv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.relu = nn.LeakyReLU(0.1)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class AdaIN(nn.Module):
    def __init__(self, input_dim):
        super(AdaIN, self).__init__()
        self.bn = nn.BatchNorm2d(input_dim, affine=False)

    def forward(self, x, y_scale, y_bias):
        x = self.bn(x)
        x = x * y_scale
        x = x + y_bias
        return x


class GeneratorUnit(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(GeneratorUnit, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channels_in + 1,
            out_channels=channels_out,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.ada = AdaIN(channels_out)

    def forward(self, x, enc_key, y_scale, y_bias):
        shaped_enc_key = enc_key.view([enc_key.shape[0], 1, 1, 1]).expand([-1, -1, x.shape[2], x.shape[3]])
        x = torch.cat((x, shaped_enc_key), 1)
        x = self.conv(x)
        x = self.ada(x, y_scale, y_bias)
        return x


class GeneratorRemapping(nn.Module):
    def __init__(self, channels, scalebias_per_step):
        super(GeneratorRemapping, self).__init__()
        self.channels = channels
        self.scalebias_per_step = scalebias_per_step
        self.conv_1 = BnReluConv(
            in_channels=(channels * scalebias_per_step * 2) + 1,
            out_channels=channels * scalebias_per_step * 2,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.conv_2 = BnReluConv(
            in_channels=(channels * scalebias_per_step * 2) + 1,
            out_channels=channels * scalebias_per_step * 2,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv_3 = BnReluConv(
            in_channels=(channels * scalebias_per_step * 2) + 1,
            out_channels=channels * scalebias_per_step * 2,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

    def forward(self, y_scale, y_bias, enc_key):
        shaped_enc_key = enc_key.view([enc_key.shape[0], 1, 1, 1]).expand([-1, -1, y_scale.shape[2], y_scale.shape[3]])
        y = torch.cat((y_scale, y_bias), 1)

        y = torch.cat((y, shaped_enc_key), 1)
        y = self.conv_1(y)
        y = torch.cat((y, shaped_enc_key), 1)
        y = self.conv_2(y)
        y = torch.cat((y, shaped_enc_key), 1)
        y = self.conv_3(y)

        y_scales = []
        y_biases = []
        for idx in range(0, self.channels * self.scalebias_per_step, self.channels):
            idx_b = idx + self.channels
            y_scales.append(y[:, idx:idx + self.channels, :, :])
            y_biases.append(y[:, idx_b:idx_b + self.channels, :, :])

        return y_scales, y_biases


class GeneratorStep(nn.Module):
    def __init__(self, channels_in, channels_out, scalebias_per_step=1, initialize=False, initial_shape=None, terminate=False):
        super(GeneratorStep, self).__init__()
        self.initialize = initialize
        self.terminate = terminate
        if initialize:
            self.base = nn.Parameter(torch.randn((channels_in, initial_shape[0], initial_shape[1])))
        else:
            self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)

        self.remap = GeneratorRemapping(channels_in, scalebias_per_step)

        self.units = []
        for idx in range(scalebias_per_step):
            unit = GeneratorUnit(channels_in, channels_in)
            self.units.append(unit)
            self.add_module('gen_{}'.format(idx), unit)

        if not terminate:
            self.conv_out = nn.Conv2d(
                in_channels=channels_in + 1,
                out_channels=channels_out,
                kernel_size=(3, 3),
                padding=(1, 1),
            )

    def forward(self, y_scale, y_bias, enc_key, x=None):
        if self.initialize:
            batch_size = y_scale.shape[0]
            x = self.base.view([1, *self.base.shape]).expand([batch_size, -1, -1, -1])
        else:
            x = self.upscale(x)

        y_scales, y_biases = self.remap(y_scale, y_bias, enc_key)

        for unit, y_scale, y_bias in zip(self.units, y_scales, y_biases):
            x = unit(x, enc_key, y_scale, y_bias)

        if not self.terminate:
            shaped_enc_key = enc_key.view([enc_key.shape[0], 1, 1, 1]).expand([-1, -1, x.shape[2], x.shape[3]])
            x = torch.cat((x, shaped_enc_key), 1)
            x = self.conv_out(x)

        return x


class Generator(nn.Module):
    def __init__(self, scalebias_per_step):
        super(Generator, self).__init__()
        self.steps = []
        depth = 256 // 4  # max dim / min dim
        initialize = True
        initial_shape = (4, 4)  # Min dim
        while depth >= 2:
            step = GeneratorStep(depth, depth // 2, scalebias_per_step, initialize, initial_shape)
            self.steps.append(step)
            self.add_module('step_{}'.format(depth // 2), step)
            depth //= 2
            initialize = False
        step_out = GeneratorStep(1, 1, scalebias_per_step, terminate=True)
        self.steps.append(step_out)
        self.add_module('step_out', step_out)

        self.tanh = nn.Tanh()

    def forward(self, y_encoded, enc_key):
        y_encoded = list(reversed(y_encoded))
        x = None
        for step, scalebias in zip(self.steps, y_encoded):
            y_scale, y_bias = scalebias
            x = step(y_scale, y_bias, enc_key, x)
        x = self.tanh(x)
        return (x + 1) / 2
