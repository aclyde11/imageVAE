import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F

class SmilesEncoder(nn.Module):

    def __init__(self,  vocab_size, max_length_sequence, rep_size = 500 * 2, embedder = None):
        super(SmilesEncoder, self).__init__()
        self.rep_size = rep_size
        self.embeder = embedder
        self.vocab_size = vocab_size
        self.max_length_sequence = max_length_sequence



        ##layers

        self.conv1 = nn.Conv1d(self.vocab_size, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.covn3 = nn.Conv1d(9, 10, 11)
        self.relu = nn.ReLU()
        self.dense = nn.Linear(2000, rep_size)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 9 * 10 * 11)
        x = self.relu(self.dense(x))
        print(x.shape)
        return

class PictureDecoder(nn.Module):
    def __init__(self, rep_size=500):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size

        # Decoder
        self.conv15 = nn.ConvTranspose2d(300, 256, kernel_size=2, stride=2, padding=1, output_padding=1, bias=False)
        self.conv15_ = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv16_ = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn16 = nn.BatchNorm2d(32)
        self.conv20 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=2, output_padding=1, bias=False)
        self.conv20_ = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=2, bias=False)
        self.bn20 = nn.BatchNorm2d(32)
        self.conv17 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv17_ = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn21 = nn.BatchNorm2d(16)
        self.conv18 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv18_ = nn.ConvTranspose2d(16, 8, kernel_size=30, stride=1, padding=2, bias=False)
        self.bn22 = nn.BatchNorm2d(8)
        self.conv19 = nn.ConvTranspose2d(8, 3, kernel_size=31, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

    def decode(self, z):
        out = self.fc_bn3(self.fc3(z))
        out = self.relu(out)
        out = self.fc_bn4(self.fc4(out))
        out = self.relu(out).view(-1, 300, 3, 3)
        out = self.relu(self.conv15(out))
        out = self.relu(self.conv15_(out))
        out = self.bn15(out)
        out = self.relu(self.conv16(out))
        ouu = self.relu(self.conv16_(out))
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
        return out.view(-1, 3, 256, 256)


    def forward(self, z):
        return self.decode(z)


class SmilesToImageModle(nn.Module):
    def __init__(self, encoder_model, decoder_model, rep_size=500):
        super(SmilesToImageModle, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(rep_size, rep_size)
        self.fc_bn1 = nn.BatchNorm1d(rep_size)
        self.fc21 = nn.Linear(rep_size, rep_size)
        self.fc22 = nn.Linear(rep_size, rep_size)

        # Sampling vector
        self.fc3 = nn.Linear(rep_size, rep_size)
        self.fc_bn3 = nn.BatchNorm1d(rep_size)
        self.fc4 = nn.Linear(rep_size, rep_size)
        self.fc_bn4 = nn.BatchNorm1d(rep_size)


    def encode(self, x):
        x = torch.split(self.encoder(x), self.rep_size, 1)

        return x[0], x[1]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar