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

class SmilesDecoder(nn.Module):
    def __init__(self,  vocab_size, max_length_sequence, rep_size = 2000 , embedder = None):
        super(SmilesDecoder, self).__init__()
        self.rep_size = rep_size
        self.embeder = embedder
        self.vocab_size = vocab_size
        self.max_length_sequence = max_length_sequence

        self.repeat_vector = lambda x : x.unsqueeze(2).expand(-1, max_length_sequence)
        self.gru1 = nn.GRU(input_size = 1, num_layers=1, hidden_size=501, batch_first=True)
        self.gru2 = nn.GRU(input_size = 1, num_layers=1, hidden_size=501, batch_first=True)
        self.gru3 = nn.GRU(input_size = 1, num_layers=1, hidden_size=501, batch_first=True)
        self.dense = nn.Linear(501, vocab_size)
        self.timedib = TimeDistributed(self.dense, batch_first=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        print(x.shape)
        x = self.repeat_vector(x)
        print(x.shape)
        x, b = self.tanh(self.gru1(x))
        print(x.shape, b.shape)
        x, b = self.tanh(self.gru2(x, b))
        print(x.shape, b.shape)
        x, _ = self.tanh(self.gru3(x, b))
        print(x.shape)
        x = self.relu(self.timedib(x))
        print(x.shape)
        return x


class SmilesEncoder(nn.Module):

    def __init__(self,  vocab_size, max_length_sequence, rep_size = 2000 , embedder = None):
        super(SmilesEncoder, self).__init__()
        self.rep_size = rep_size
        self.embeder = embedder
        self.vocab_size = vocab_size
        self.max_length_sequence = max_length_sequence

        ##layers

        self.conv1 = nn.Conv1d(in_channels=self.vocab_size, out_channels=9, kernel_size=9, stride=1)
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9, stride=1)
        self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11, stride=1)

        self.relu = nn.ReLU()

        # Latent vectors mu and sigma
        self.fc22 = nn.Linear(256 * 2, rep_size)
        self.fc21 = nn.Linear(256 * 2, rep_size)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        print(x.shape)
        x = x.view(-1, 256 * 2)

        return self.fc21(x), self.fc22(x)

class PictureEncoder(nn.Module):
    def __init__(self, rep_size=2000):
        super(PictureEncoder, self).__init__()
        self.rep_size = rep_size
        self.encoder = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=rep_size * 2)

    def forward(self, x):
        x = torch.split(self.encoder(x), self.rep_size, 1)

        return x[0], x[1]


class PictureDecoder(nn.Module):
    def __init__(self, rep_size=2000):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size
        # Sampling vector
        self.fc3 = nn.Linear(rep_size, rep_size)
        self.fc_bn3 = nn.BatchNorm1d(rep_size)
        self.fc4 = nn.Linear(rep_size, rep_size)
        self.fc_bn4 = nn.BatchNorm1d(rep_size)

        # Decoder
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
        out = self.relu(out).view(-1, 125, 4, 4)
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