import torch
import torch.nn as nn


class BnReluConvConv(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(BnReluConvConv, self).__init__()
        self.bn = nn.BatchNorm2d(input_depth, affine=False)
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(
            in_channels=input_depth,
            out_channels=output_depth,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_depth,
            out_channels=output_depth,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BnReluFc(nn.Module):
    def __init__(self, in_features, out_features):
        super(BnReluFc, self).__init__()
        self.bn = nn.BatchNorm1d(in_features, affine=False)
        self.relu = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x


class DiscriminatorStep(nn.Module):
    def __init__(self, input_depth, initialize=False):
        super(DiscriminatorStep, self).__init__()
        self.initialize = initialize
        if initialize:
            in_depth = input_depth
        else:
            in_depth = input_depth * 2
        self.brcc = BnReluConvConv(in_depth, input_depth * 2)

    def forward(self, *inputs):
        if self.initialize:
            scale, bias = inputs
            x = torch.cat((scale, bias), 1)
        else:
            x, enc = inputs
            scale, bias = enc
            x = torch.cat((x, scale, bias), 1)
        x = self.brcc(x)
        return x


class DiscriminatorConv(nn.Module):
    def __init__(self, scalebias_per_step=1):
        super(DiscriminatorConv, self).__init__()
        dim = 256
        depth = 2 * scalebias_per_step

        self.steps = []
        step = DiscriminatorStep(depth, initialize=True)
        self.add_module('step_{}'.format(dim), step)
        self.steps.append(step)
        dim //= 2
        depth *= 2
        while dim >= 8:
            step = DiscriminatorStep(depth)
            self.add_module('step_{}'.format(dim), step)
            self.steps.append(step)
            dim //= 2
            depth *= 2

    def forward(self, encoded_layers):
        x = self.steps[0](*encoded_layers[0])
        for step, enc in zip(self.steps[1:], encoded_layers[1:-1]):
            x = step(x, enc)
        out_scale, out_bias = encoded_layers[-1]
        x = torch.cat((x, out_scale, out_bias), 1)
        return x


class DiscriminatorHead(nn.Module):
    def __init__(self, features):
        super(DiscriminatorHead, self).__init__()
        self.layers = []
        for in_features, out_features in features:
            fc = BnReluFc(in_features, out_features)
            self.add_module('fc_{}-{}'.format(in_features, out_features), fc)
            self.layers.append(fc)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        for layer in self.layers:
            x = layer(x)
        x = self.softmax(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, scalebias_per_step):
        super(Discriminator, self).__init__()

        self.conv_step = DiscriminatorConv(scalebias_per_step)

        # Smooth reduce from ? channels to 64
        # I'm not sure if this max depth calculation actually reflects the underlying math, but it works out for
        # the 256x256 image case
        intermediate_channels = int((scalebias_per_step * 256 * 64) ** 0.5)
        self.conv_reduction_1 = nn.Conv2d(
            in_channels=256 * scalebias_per_step,
            out_channels=intermediate_channels,
            kernel_size=(1, 1)
        )
        self.conv_reduction_2 = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=64,
            kernel_size=(1, 1)
        )

        self.head = DiscriminatorHead([(4 * 4 * 64, 128), (128, 16), (16, 2)])

    def forward(self, encoded_layers):
        x = self.conv_step(encoded_layers)
        x = self.conv_reduction_1(x)
        x = self.conv_reduction_2(x)
        x = self.head(x)
        return x

