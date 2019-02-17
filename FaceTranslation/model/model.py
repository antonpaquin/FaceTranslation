import torch.nn as nn


class DeepConvStride(nn.Module):
    def __init__(self, n_conv=1, in_channels=3, out_channels=3, transpose=False, activation_out=True):
        super(DeepConvStride, self).__init__()
        self.dynamic_conv_layers = []
        self.dynamic_activations = []
        if transpose:
            conv_layer = nn.ConvTranspose2d
        else:
            conv_layer = nn.Conv2d

        c = in_channels
        for idx in range(n_conv):
            conv = conv_layer(
                in_channels=c,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            c = out_channels
            self.add_module('conv_{}'.format(idx), conv)
            self.dynamic_conv_layers.append(conv)
            elu = nn.ELU()
            self.add_module('elu_{}'.format(idx), elu)
            self.dynamic_activations.append(elu)

        self.conv_out = conv_layer(
            in_channels=c,
            out_channels=out_channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.apply_activation_out = activation_out
        if activation_out:
            self.activation_out = nn.ELU()

    def forward(self, x):
        for conv, activation in zip(self.dynamic_conv_layers, self.dynamic_activations):
            x = conv(x)
            x = activation(x)
        x = self.conv_out(x)
        if self.apply_activation_out:
            x = self.activation_out(x)
        return x


class FaceTranslationModel(nn.Module):
    def __init__(self):
        super(FaceTranslationModel, self).__init__()
        self.conv1 = DeepConvStride(1, in_channels=3, out_channels=5)
        self.conv2 = DeepConvStride(1, in_channels=5, out_channels=7)
        self.deconv1 = DeepConvStride(1, in_channels=7, out_channels=5, transpose=True)
        self.deconv2 = DeepConvStride(1, in_channels=5, out_channels=3, transpose=True, activation_out=False)
        self.activation_out = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.activation_out(x)
        return x
