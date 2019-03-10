import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from ResNet import ResNet, BasicBlock


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

class SmilesDecoder(nn.Module):
    def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
        super(SmilesDecoder, self).__init__()
        self.rep_size = rep_size
        self.embeder = embedder
        self.vocab_size = vocab_size
        self.max_length_sequence = max_length_sequence
        self.repeat_vector = Repeat(self.max_length_sequence)
        self.gru1 = nn.GRU(input_size = rep_size, num_layers=3, hidden_size=501, batch_first=True)
        self.dense = nn.Sequential(nn.Linear(501, vocab_size), nn.Softmax())
        self.timedib = TimeDistributed(self.dense, batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.repeat_vector(x)
        x, _ = self.gru1(x)
        x = self.timedib(x)
        return x


class SmilesEncoder(nn.Module):

    def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
        super(SmilesEncoder, self).__init__()
        self.rep_size = rep_size
        self.embeder = embedder
        self.vocab_size = vocab_size
        self.max_length_sequence = max_length_sequence

        ##layers

        self.conv1 = nn.Conv1d(in_channels=self.max_length_sequence, out_channels=90, kernel_size=9, stride=1)
        self.conv2 = nn.Conv1d(in_channels=90, out_channels=300, kernel_size=10, stride=1)
        self.conv3 = nn.Conv1d(in_channels=300, out_channels=900, kernel_size=10, stride=1)

        self.relu = nn.ReLU()

        # Latent vectors mu and sigma
        self.fc22 = nn.Linear(900, rep_size)
        self.fc21 = nn.Linear(900, rep_size)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = Flatten()(x)

        return self.fc21(x), self.fc22(x)

class PictureEncoder(nn.Module):
    def __init__(self, rep_size=500):
        super(PictureEncoder, self).__init__()
        self.rep_size = rep_size
        self.encoder = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=rep_size)
        self.mu = nn.Linear(rep_size, rep_size)
        self.logvar = nn.Linear(rep_size, rep_size)

    def forward(self, x):
        x = self.encoder(x)

        return self.mu(x), self.logvar(x)


class PictureDecoder(nn.Module):
    def __init__(self, rep_size=500):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size
        # Sampling vector
        self.fc3 = nn.Linear(rep_size, rep_size)
        self.fc_bn3 = nn.BatchNorm1d(rep_size)
        self.fc4 = nn.Linear(rep_size, rep_size)
        self.fc_bn4 = nn.BatchNorm1d(rep_size)

        # Decoder
        self.preconv = nn.ConvTranspose2d(125, 125, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv15 = nn.ConvTranspose2d(125, 128, kernel_size=2, stride=2, padding=0,  bias=False)
        self.conv15_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv16_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv20 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv20_ = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(64)
        self.conv17 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv17_ = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(32)
        self.conv18 = nn.ConvTranspose2d(32, 16, kernel_size=40, stride=2, padding=0, bias=False)
        self.conv18_ = nn.ConvTranspose2d(16, 3, kernel_size=40, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(3)
        self.conv19 = nn.ConvTranspose2d(3, 3, kernel_size=40, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def decode(self, z):
        out = self.fc_bn3(self.fc3(z))
        out = self.relu(out)
        out = self.fc_bn4(self.fc4(out))
        out = self.relu(out).view(-1, 125, 2, 2)
        out = self.relu(self.preconv(out))
        print(out.shape)
        out = self.relu(self.conv15(out))
        out = self.relu(self.conv15_(out))
        out = self.bn15(out)
        out = self.relu(self.conv16(out))
        out = self.relu(self.conv16_(out))
        out = self.bn16(out)

        out = self.relu(self.conv20(out))
        out = self.relu(self.conv20_(out))
        out = self.bn20(out)
        out = self.relu(self.conv17(out))
        out = self.relu(self.conv17_(out))
        out = self.bn21(out)

        out = self.relu(self.conv18(out))
        out = self.relu(self.conv18_(out))
        out = self.bn22(out)
        out = self.conv19(out)
        print(out.shape)
        return out


    def forward(self, z):
        return self.decode(z)


class GeneralVae(nn.Module):
    def __init__(self, encoder_model, decoder_model, rep_size=2000):
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

import torch
import torch.nn as nn
from torch.autograd import Variable




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




class MolEncoder(nn.Module):

    def __init__(self, i=60, o=500, c=27):
        super(MolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
                                     SELU(inplace=True))

        self.z_mean = nn.Linear(435, o)
        self.z_log_var = nn.Linear(435, o)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        return self.z_mean(out), self.z_log_var(out)



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