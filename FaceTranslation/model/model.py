import torch
import torch.nn as nn


class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        in_depth = 1
        depth_step = 6
        n_steps = 5

        self.layers = []
        for idx in range(n_steps):
            conv = nn.Conv2d(
                in_channels=(idx * depth_step) + in_depth + 1,
                out_channels=((idx + 1) * depth_step) + in_depth,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            self.add_module('conv_{}'.format(idx), conv)
            self.layers.append(conv)
            actv = nn.LeakyReLU(0.1)
            self.add_module('relu_{}'.format(idx), actv)
            self.layers.append(actv)

        self.activation_out = nn.Tanh()

    def forward(self, x, label):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x = torch.cat((x, label), 1)
            x = layer(x)
        return x


class DecoderModel(nn.Module):
    def __init__(self):
        super(DecoderModel, self).__init__()
        out_depth = 1
        depth_step = 6
        n_steps = 5

        self.layers = []
        for idx in range(n_steps, 0, -1):
            conv = nn.ConvTranspose2d(
                in_channels=(idx * depth_step) + out_depth + 1,
                out_channels=((idx - 1) * depth_step) + out_depth,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            self.add_module('deconv_{}'.format(idx), conv)
            self.layers.append(conv)
            if idx == 1:
                actv = nn.Tanh()
            else:
                actv = nn.LeakyReLU(0.1)

            self.add_module('relu_{}'.format(idx), actv)
            self.layers.append(actv)

    def forward(self, x, label):
        for layer in self.layers:
            if isinstance(layer, nn.ConvTranspose2d):
                x = torch.cat((x, label), 1)
            x = layer(x)
        return x


class AdversaryModel(nn.Module):
    def __init__(self):
        super(AdversaryModel, self).__init__()
        encoder_depth = 31
        depth_step = 5
        conv_conv_pool = 5
        image_dim = 256

        ccp_out_depth = encoder_depth + (depth_step * conv_conv_pool)
        deep_image_dim = int(image_dim / (2**conv_conv_pool))
        dense_dimensions = [(deep_image_dim ** 2) * ccp_out_depth, 100, 50, 10, 2]

        self.ccp_layers = []
        for idx in range(conv_conv_pool):
            conv_a = nn.Conv2d(
                in_channels=(idx * depth_step) + encoder_depth,
                out_channels=((idx + 1) * depth_step) + encoder_depth,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            self.add_module('conv_{}a'.format(idx), conv_a)
            self.ccp_layers.append(conv_a)
            relu_a = nn.LeakyReLU(0.1)
            self.add_module('relu_{}a'.format(idx), relu_a)
            self.ccp_layers.append(relu_a)

            conv_b = nn.Conv2d(
                in_channels=((idx + 1) * depth_step) + encoder_depth,
                out_channels=((idx + 1) * depth_step) + encoder_depth,
                kernel_size=(3, 3),
                padding=(1, 1),
            )
            self.add_module('conv_{}b'.format(idx), conv_b)
            self.ccp_layers.append(conv_b)
            relu_b = nn.LeakyReLU(0.1)
            self.add_module('relu_{}b'.format(idx), relu_b)
            self.ccp_layers.append(relu_b)

            pool = nn.AvgPool2d((2, 2))
            self.add_module('pool_{}'.format(idx), pool)
            self.ccp_layers.append(pool)

        self.dense_layers = []
        for idx, dims in enumerate(zip(dense_dimensions[:-1], dense_dimensions[1:])):
            in_dim, out_dim = dims
            dense = nn.Linear(in_dim, out_dim)
            self.add_module('dense_{}'.format(idx), dense)
            self.dense_layers.append(dense)
            if idx != len(dense_dimensions) - 2:
                relu = nn.LeakyReLU(0.1)
                self.add_module('dense_relu_{}'.format(idx), relu)
                self.dense_layers.append(relu)

        self.activation_out = nn.Softmax(1)

    def forward(self, x):
        for layer in self.ccp_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.activation_out(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self.adversary = AdversaryModel()

    def encoder_params(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def adversary_params(self):
        yield from self.adversary.parameters()

    def forward(self, x, label):
        encoded = self.encoder(x, label)
        decoded = self.decoder(encoded, label)
        adversary_pred = self.adversary(encoded)
        return decoded, adversary_pred
