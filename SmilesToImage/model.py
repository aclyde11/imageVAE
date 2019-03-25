import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from ResNet import ResNet, BasicBlock, Bottleneck
import torch.utils.model_zoo as model_zoo
import torchvision
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class Repeat(nn.Module):

    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)

class Flatten(nn.Module):

    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)

# class SmilesDecoder(nn.Module):
#     def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
#         super(SmilesDecoder, self).__init__()
#         self.rep_size = rep_size
#         self.embeder = embedder
#         self.vocab_size = vocab_size
#         self.max_length_sequence = max_length_sequence
#         self.repeat_vector = Repeat(self.max_length_sequence)
#         self.gru1 = nn.GRU(input_size = rep_size, num_layers=3, hidden_size=501, batch_first=True)
#         self.dense = nn.Sequential(nn.Linear(501, vocab_size), nn.Softmax())
#         self.timedib = TimeDistributed(self.dense, batch_first=True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#
#     def forward(self, x):
#         x = self.repeat_vector(x)
#         x, _ = self.gru1(x)
#         x = self.timedib(x)
#         return x
#
#
# class SmilesEncoder(nn.Module):
#
#     def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
#         super(SmilesEncoder, self).__init__()
#         self.rep_size = rep_size
#         self.embeder = embedder
#         self.vocab_size = vocab_size
#         self.max_length_sequence = max_length_sequence
#
#         ##layers
#
#         self.conv1 = nn.Conv1d(in_channels=self.max_length_sequence, out_channels=90, kernel_size=9, stride=1)
#         self.conv2 = nn.Conv1d(in_channels=90, out_channels=300, kernel_size=10, stride=1)
#         self.conv3 = nn.Conv1d(in_channels=300, out_channels=900, kernel_size=10, stride=1)
#
#         self.relu = nn.ReLU()
#
#         # Latent vectors mu and sigma
#         self.fc22 = nn.Linear(900, rep_size)
#         self.fc21 = nn.Linear(900, rep_size)
#
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = Flatten()(x)
#
#         return self.fc21(x), self.fc22(x)

class PictureEncoder(nn.Module):
    def __init__(self, rep_size=500):
        super(PictureEncoder, self).__init__()
        self.rep_size = rep_size
        resnet = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(8192, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, 512)
        self.log_var = nn.Linear(512, 512)


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return self.fc_mu(x), self.log_var(x)

def conv3x3T(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv4x4T(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                     padding=1, bias=False)

def conv1x1T(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upscale=None):
        super(TransposeBlock, self).__init__()

        self.conv1 = conv1x1T(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride > 1:
            self.conv2 = conv4x4T(out_channels, out_channels, stride)
        else:
            self.conv2 = conv3x3T(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1T(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels )
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.Upsample(scale_factor=(2, 2))
        self.upconv = conv1x1T(in_channels, out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.stride > 1:
            print(identity.shape)
            identity = self.unpool(identity)
            print(identity.shape)
            identity = self.upconv(identity)

        print('x  ', x.shape, 'id', identity.shape)
        x = x + identity
        return self.relu(x)




class PictureDecoder(nn.Module):
    def __init__(self, rep_size=512):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size
        self.in_planes = 8
        self.fc = nn.Sequential(nn.Linear(rep_size, 512), nn.ReLU())
        # Decoder
        layers = []
        sizes =   [2, 1, 1, 2, 2, 2, 1]
        strides = [2, 2, 2, 2, 1, 2, 1]
        planes =  [8, 7, 6, 5, 4, 3, 3]

        for size, stride, plane in zip(sizes, strides, planes):
            for i in range(size):
                if i == 0 and stride > 1:
                    print('going from ', self.in_planes, ' to ', self.in_planes / 2)
                    layers.append(TransposeBlock(self.in_planes, plane, stride=2))
                else:
                    layers.append(TransposeBlock(self.in_planes, plane, stride=1))
                self.in_planes = plane

        self.model = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def decode(self, z):
        z = self.fc(z)
        z = z.view(-1, 8, 8, 8)
        z = self.model(z)
        print(z.shape)
        return z


    def forward(self, z):
        return self.decode(z)


class GeneralVae(nn.Module):
    def __init__(self, encoder_model, decoder_model, rep_size=512):
        super(GeneralVae, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Lambda(nn.Module):

    def __init__(self, i=1000, o=500, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale


    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class ComboVAE(nn.Module):
    def __init__(self, encoder_model_1, encoder_model_2, decoder_model_1, decoder_model_2, rep_size=500):
        super(ComboVAE, self).__init__()
        self.rep_size = rep_size

        self.scale = 1E-2
        self.encoder1 = encoder_model_1
        self.encoder2 = encoder_model_2
        self.decoder1 = decoder_model_1
        self.decoder2 = decoder_model_2


        self.z_mean = nn.Linear(rep_size * 2, rep_size)
        self.z_log_var = nn.Linear(rep_size * 2, rep_size)


    def encode(self, x1, x2):  ##returns single values encoded
        return self.encoder1(x1), self.encoder2(x2)


    def reparam(self, logvar, mu):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode(self, z):
        return self.decoder1(z), self.decoder2(z)

    def encode_latent_(self, x1, x2):
        x1, x2 = self.encode(x1, x2)
        x = torch.cat([x1,x2], dim=1)
        mu, logvar = (self.z_mean(x), self.z_log_var(x))
        z = self.reparam(logvar, mu)

        return z, mu, logvar

    def forward(self, x1, x2):
        z, mu, logvar= self.encode_latent_(x1, x2)

        y_1, y_2 = self.decode(z)
        return y_1, y_2, mu, logvar


import torch
import torch.nn as nn
from torch.autograd import Variable




# class SELU(nn.Module):
#
#     def __init__(self, alpha=1.6732632423543772848170429916717,
#                  scale=1.0507009873554804934193349852946, inplace=False):
#         super(SELU, self).__init__()
#
#         self.scale = scale
#         self.elu = nn.ELU(alpha=alpha, inplace=inplace)
#
#     def forward(self, x):
#         return self.scale * self.elu(x)
#
#
# def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
#     model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
#              SELU(inplace=True)
#              ]
#     if p > 0.:
#         model += [nn.Dropout(p)]
#     return nn.Sequential(*model)
#
#
#
#
# class MolEncoder(nn.Module):
#
#     def __init__(self, i=60, o=500, c=27):
#         super(MolEncoder, self).__init__()
#
#         self.i = i
#
#         self.conv_1 = ConvSELU(i, 9, kernel_size=9)
#         self.conv_2 = ConvSELU(9, 9, kernel_size=9)
#         self.conv_3 = ConvSELU(9, 10, kernel_size=11)
#         self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
#                                      SELU(inplace=True))
#
#         self.z_mean = nn.Linear(435, o)
#         self.z_log_var = nn.Linear(435, o)
#
#
#     def forward(self, x):
#         out = self.conv_1(x)
#         out = self.conv_2(out)
#         out = self.conv_3(out)
#         out = Flatten()(out)
#         out = self.dense_1(out)
#
#         return self.z_mean(out), self.z_log_var(out)
#
#
#
# class MolDecoder(nn.Module):
#
#     def __init__(self, i=500, o=60, c=27):
#         super(MolDecoder, self).__init__()
#
#         self.latent_input = nn.Sequential(nn.Linear(i, i),
#                                           SELU(inplace=True))
#         self.repeat_vector = Repeat(o)
#         self.gru = nn.GRU(i, 501, 3, batch_first=True)
#         self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
#                                                           nn.Softmax())
#                                             )
#
#     def forward(self, x):
#         out = self.latent_input(x)
#         out = self.repeat_vector(out)
#         out, h = self.gru(out)
#         return self.decoded_mean(out
#       )

class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             SELU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def __init__(self, i=435, o=292, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)

        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.mu)

    # def forward(self, x):
    #     self.mu = self.z_mean(x)
    #     self.log_v = self.z_log_var(x)
    #     eps = self.scale * Variable(torch.randn(*self.log_v.size())
    #                                 ).type_as(self.log_v)
    #     return self.mu + torch.exp(self.log_v / 2.) * eps


class MolEncoder(nn.Module):

    def __init__(self, i=60, o=500, c=27):
        super(MolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
                                     SELU(inplace=True))

        #self.lmbd = Lambda(435, o)
        self.z_mean = nn.Linear(435, o)
        self.z_log_var = nn.Linear(435, o)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        return out

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(size_average=True)
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class DenseMolEncoder(nn.Module):

    def __init__(self, i=60, o=500, c=27):
        super(DenseMolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)

        self.dense_0 = nn.Sequential(Flatten(),
                                     nn.Linear(60 * 27, 500),
                                     SELU(inplace=True),
                                     nn.Linear(500, 500),
                                     SELU(inplace=True),
                                     nn.Linear(500, 500),
                                     SELU(inplace=True))
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 500),
                                     SELU(inplace=True))

        self.z_mean = nn.Linear(500, o)
        self.z_log_var = nn.Linear(500, o)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out) + self.dense_0(x)

        return out

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(size_average=True)
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=500, o=60, c=27):
        super(MolDecoder, self).__init__()

        self.latent_input = nn.Sequential(nn.Linear(i, i),
                                          SELU(inplace=True))
        self.repeat_vector = Repeat(o)
        self.gru = nn.GRU(i, 501, 3, batch_first=True)
        self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
                                                          nn.Softmax())
                                            )

    def forward(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        return self.decoded_mean(out)

class ZSpaceTransform(nn.Module):
    def __init__(self, i=500, o=60, ):
        super(ZSpaceTransform, self).__init__()

        self.mu = nn.Sequential(nn.Linear(i, i),
                                  SELU(inplace=True),
                                nn.Linear(i, i), SELU(inplace=True),
                                nn.Linear(i,i), SELU(inplace=True), nn.Linear(i,i))

        self.logvar = nn.Sequential(nn.Linear(i, i),
                                  SELU(inplace=True),
                                nn.Linear(i, i), SELU(inplace=True),
                                nn.Linear(i,i), SELU(inplace=True), nn.Linear(i,i))

    def forward(self, mu, log):
        mu = self.mu(mu)
        log = self.logvar(log)
        return mu, log



class TestVAE(nn.Module):

    def __init__(self, encoder, transformer, decoder):
        super(TestVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def encode(self, x):
        self.mu, self.log_v = self.encoder(x)
        self.mu, self.log_v = self.transformer(self.mu, self.log_v)

        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        y =  eps.mul(std).add_(self.mu)
        return y

    def decode(self,x):
        return self.decoder(x)


    def forward(self, x, return_y = False):
        self.mu, self.log_v = self.encoder(x)
        self.mu, self.log_v = self.transformer(self.mu, self.log_v)
        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        y =  eps.mul(std).add_(self.mu)
        if return_y:
            return y, self.decoder(y)
        return self.decoder(y)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.mu, self.log_v

        #bce = nn.BCELoss(size_average=True)
        bce = nn.MSELoss(reduction="sum")

        xent_loss =  bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss



class AutoModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoModel, self).__init__()
        self.encoder = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=500)
        self.attention = nn.Linear(500, 500)
        self.reduce = nn.Linear(500, 292)
        self.decoder = decoder


    def forward(self, x):
        x = self.encoder(x)
        atten = nn.Softmax()(self.attention(x))
        x = nn.ReLU()(self.reduce(atten * x))
        x = self.decoder(x)
        return x

