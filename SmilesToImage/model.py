import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from ResNet import ResNet, BasicBlock, Bottleneck
import math
import numpy as np
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

# class PictureEncoder(nn.Module):
#     def __init__(self, rep_size=500):
#         super(PictureEncoder, self).__init__()
#         self.rep_size = rep_size
#         self.encoder = ResNet(BasicBlock, [3, 2, 2, 3], num_classes=rep_size)
#         self.mu = nn.Linear(rep_size, rep_size)
#         self.logvar = nn.Linear(rep_size, rep_size)
#
#     def forward(self, x):
#         x = self.encoder(x)
#
#         return self.mu(x), self.logvar(x)
#

def unit_prefix(x, n=1):
    for i in range(n): x = x.unsqueeze(0)
    return x


def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd:
        y = unit_prefix(y, xd - yd)
    elif yd > xd:
        x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd - i - 1
        if ys[td] == 1:
            ys[td] = xs[td]
        elif xs[td] == 1:
            xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)


def matmul(X, Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i], Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)


def xrecons_grid(X, B, A):
    """
    plots canvas for single time step
    X is x_recons, (batch_size x img_size)
    assumes features = BxA images
    batch is assumed to be a square number
    """
    padsize = 1
    padval = .5
    ph = B + 2 * padsize
    pw = A + 2 * padsize
    batch_size = X.shape[0]
    N = int(np.sqrt(batch_size))
    X = X.reshape((N, N, B, A))
    img = np.ones((N * ph, N * pw)) * padval
    for i in range(N):
        for j in range(N):
            startr = i * ph + padsize
            endr = startr + B
            startc = j * pw + padsize
            endc = startc + A
            img[startr:endr, startc:endc] = X[i, j, :, :]
    return img


class DrawModel(nn.Module):
    def __init__(self, T, A, B, z_size, N, dec_size, enc_size):
        super(DrawModel, self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.logsigmas, self.sigmas, self.mus = [0] * T, [0] * T, [0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size, dec_size)
        self.decoder_gru = nn.GRUCell(z_size, dec_size)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.dec_w_linear = nn.Linear(dec_size, N * N)

        self.sigmoid = nn.Sigmoid()

    def normalSample(self):
        return Variable(torch.randn(self.batch_size, self.z_size))

    # correct
    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = align(tmp, g)
        mu = tmp_t + g_t
        return mu

    # correct
    def filterbank(self, gx, gy, sigma2, delta):
        rng = Variable(torch.arange(0, self.N).view(1, -1))
        mu_x = self.compute_mu(gx, rng, delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = Variable(torch.arange(0, self.A).view(1, 1, -1))
        b = Variable(torch.arange(0, self.B).view(1, 1, -1))

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = self.filterbank_matrices(a, mu_x, sigma2)
        Fy = self.filterbank_matrices(b, mu_y, sigma2)

        return Fx, Fy

    def forward(self, x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            x_hat = x - self.sigmoid(c_prev)  # 3
            r_t = self.read(x, x_hat, h_dec_prev)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state))
            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        return self.sigmoid(self.cs[-1])

    def get_output(self):
        return self.sigmoid(self.cs[-1])

    def loss(self, x):
        x = x.view(x.shape[0], -1)
        self.forward(x)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons, x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * self.T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)  ####################################################
        loss = Lz + Lx  # 12
        return loss

    # correct
    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a, t_mu_x = align(a, mu_x)
        temp = t_a - t_mu_x
        temp, t_sigma = align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp, 2))
        F = F / (F.sum(2, True).expand_as(F) + epsilon)
        return F

    # correct
    def attn_window(self, h_dec):
        params = self.dec_linear(h_dec)
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # 21

        # gx_ = Variable(torch.ones(4,1))
        # gy_ = Variable(torch.ones(4, 1) * 2)
        # log_sigma_2 = Variable(torch.ones(4, 1) * 3)
        # log_delta = Variable(torch.ones(4, 1) * 4)
        # log_gamma = Variable(torch.ones(4, 1) * 5)

        gx = (self.A + 1) / 2 * (gx_ + 1)  # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)  # 23
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    # correct
    def read(self, x, x_hat, h_dec_prev):
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

        def filter_img(img, Fx, Fy, gamma, A, B, N):
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            # img = img.transpose(2,1)
            # glimpse = matmul(Fy,matmul(img,Fxt))
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N * N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)
        return torch.cat((x, x_hat), 1)

    # correct
    def write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size, self.N, self.N)
        # w = Variable(torch.ones(4,5,5) * 3)
        # self.batch_size = 4
        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A * self.B)
        return wr / gamma.view(-1, 1).expand_as(wr)

    def sampleQ(self, h_enc):
        e = self.normalSample()
        # mu_sigma = self.mu_sigma_linear(h_enc)
        # mu = mu_sigma[:, :self.z_size]
        # log_sigma = mu_sigma[:, self.z_size:]
        mu = self.mu_linear(h_enc)  # 1
        log_sigma = self.sigma_linear(h_enc)  # 2
        sigma = torch.exp(log_sigma)

        return mu + sigma * e, mu, log_sigma, sigma

    def generate(self, batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True)
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True)

        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs


class SeparableConv3(nn.Module):
    def __init__(self, kernel_size, stride=3, padding=1, bias=False):
        super(SeparableConv3, self).__init__()
        self.channels = 3

        self.ch1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())
        self.ch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())
        self.ch3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())

    def forward(self, x):
        return torch.cat((self.ch1(x[:, 0, ...]),
                          self.ch2(x[:, 1, ...]),
                          self.ch3(x[:, 2, ...])),
                         dim=1)


class PictureEncoder(nn.Module):
    def __init__(self, rep_size=256):
        super(PictureEncoder, self).__init__()
        self.rep_size = rep_size
        self.encoder = ResNet(BasicBlock, [3, 2, 2, 2], num_classes=rep_size, in_classes=1)
        self.encoder_color = ResNet(BasicBlock, [2, 1, 1, 2], num_classes=rep_size, in_classes=3)

    def forward(self, x):
        color_enc = self.encoder_color(x).view(-1, 256)
        x = torch.sum(x, dim=1, keepdim=True)
        black_enc = self.encoder(x).view(-1, 256)

        return black_enc, color_enc


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
        self.inc = in_channels
        self.ouc = out_channels
        self.conv1 = conv1x1T(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride > 1:
            self.conv2 = conv4x4T(out_channels, out_channels, stride)
        else:
            self.conv2 = conv3x3T(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1T(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
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
            identity = self.unpool(identity)
            identity = self.upconv(identity)
        elif self.inc != self.ouc:
            identity = self.upconv(identity)

        x = x + identity
        return self.relu(x)


## Dense Net

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


# class BindingAffModel(nn.Module):
#     def __init__(self, rep_size=512, dropout=None):
#         super(BindingAffModel, self).__init__()
#         self.rep_size = rep_size
#
#         sizes = [64, 32, 16, 8]
#         self.layers = []
#
#         curr_size = rep_size
#         for i in sizes:
#             if dropout is not None:
#                 self.layers.append(nn.Sequential(
#                     nn.Linear(curr_size, i),
#                     nn.ReLU(),
#                     nn.Dropout(dropout)
#                 ))
#             else:
#                 self.layers.append(nn.Sequential(
#                     nn.Linear(curr_size, i),
#                     nn.ReLU()
#                 ))
#             curr_size += i
#         self.layers = ListModule(*self.layers)
#         self.final_layer = nn.Sequential(
#             nn.Linear(curr_size, 1),
#             nn.ReLU()
#         )
#
#
#
#     def forward(self, x):
#         concats = x
#         for layer in self.layers:
#             x = layer(concats)
#             concats = torch.cat((concats, x), dim=1)
#         x = self.final_layer(concats)
#         return x
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class BindingAffModel(nn.Module):
    def __init__(self, rep_size=500, dropout=None):
        super(BindingAffModel, self).__init__()
        self.rep_size = rep_size

        self.attention = MultiHeadAttention(2, 500)
        self.model = nn.Sequential(
            nn.Linear(500, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.attention(x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size))
        return self.model(out.view(batch_size, -1))


class TranposeConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, padding=(0, 0), stride=(0, 0), kernel_size=(0, 0), dropout=None):
        super(TranposeConvBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_plane, out_plane, kernel_size=kernel_size[0], padding=padding[0],
                                        stride=stride[0], bias=False)
        self.conv2 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1],
                                        stride=stride[1], bias=False)
        # self.conv3 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1], stride=stride[1], bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PictureDecoder(nn.Module):
    def __init__(self, rep_size=256):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size
        # Sampling vector
        self.fc3 = nn.Linear(rep_size, rep_size)
        self.fc_bn3 = nn.BatchNorm1d(rep_size)
        self.fc4 = nn.Linear(rep_size, 500)
        self.fc_bn4 = nn.BatchNorm1d(500)

        # Decoder
        self.preconv = nn.ConvTranspose2d(125, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv15 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.conv15_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv16_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(128)
        self.conv20 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv20_ = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn20 = nn.BatchNorm2d(64)
        self.conv17 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv17_ = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv18 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv18_ = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn22 = nn.BatchNorm2d(16)
        self.conv19 = nn.ConvTranspose2d(16, 3, kernel_size=5, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, out):
        out = self.fc_bn3(self.fc3(out))
        out = self.relu(out)
        out = self.fc_bn4(self.fc4(out))
        out = self.relu(out).view(-1, 125, 2, 2)
        out = self.relu(self.preconv(out))
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

        out = self.sigmoid(out)
        return out


# class PictureDecoder(nn.Module):
#     def __init__(self, rep_size=500):
#         super(PictureDecoder, self).__init__()
#         self.rep_size = rep_size
#         # Sampling vector
#         self.fc3 = nn.Linear(rep_size, rep_size)
#         self.fc_bn3 = nn.BatchNorm1d(rep_size)
#         self.fc4 = nn.Linear(rep_size, rep_size)
#         self.fc_bn4 = nn.BatchNorm1d(rep_size)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         # Decoder
#         conv1 = TranposeConvBlock(125, 128, kernel_size=[5, 5], stride=[1, 1], padding=[1,1])
#         conv2 = TranposeConvBlock(128, 128, kernel_size=[5, 5], stride=[1, 1], padding=[1, 1])
#         conv3 = TranposeConvBlock(128, 128, kernel_size=[5, 5],  stride=[1, 1], padding=[1, 1])
#         conv4 = TranposeConvBlock(128, 128, kernel_size=[5, 5], stride=[1, 1], padding=[1, 1])
#         conv5 = TranposeConvBlock(128,  64, kernel_size=[5, 5], stride=[1, 1], padding=[1, 1])
#         conv6 = TranposeConvBlock(64, 64, kernel_size=[30, 40], stride=[1, 1], padding=[1,1])
#         conv7 = TranposeConvBlock(64, 64, kernel_size=[36, 36], stride=[1, 1], padding=[0,0])
#         conv8 = TranposeConvBlock(64, 64, kernel_size=[40, 30], stride=[1, 1], padding=[0,0])
#         conv9 = TranposeConvBlock(64, 32, kernel_size=[20, 5], stride=[1, 1,], padding=[1,1])
#         conv10 = TranposeConvBlock(32, 3, kernel_size=[5, 5], stride=[1, 1,], padding=[0,0])
#         conv11 = TranposeConvBlock(3, 3, kernel_size=[5, 2], stride=[1,1], padding=[0,0])
#         relu = nn.ReLU()
#
#
#         self.model = nn.Sequential(
#             conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11)
#
#
#     def decode(self, z):
#         out = self.fc_bn3(self.fc3(z))
#         out = self.tanh(out)
#         out = out.view(-1, 125, 2, 2)
#         out = self.model(out)
#         return out
#
#
#     def forward(self, z):
#         return self.decode(z)

#
# class PictureDecoder(nn.Module):
#     def __init__(self, rep_size=512):
#         super(PictureDecoder, self).__init__()
#         self.rep_size = rep_size
#         self.in_planes = 128
#         self.fc = nn.Sequential(nn.Linear(rep_size, 512), nn.ReLU())
#         # Decoder
#         layers = []
#         sizes =   [2,    2,  2, 2, 2, 1, 1, 1]
#         strides = [2,    2,  2, 2, 2, 2, 2, 1]
#         planes =  [64, 32, 16, 8, 4, 3, 3, 3]
#
#         for size, stride, plane in zip(sizes, strides, planes):
#             for i in range(size):
#                 if i == 0 and stride > 1:
#                     layers.append(TransposeBlock(self.in_planes, plane, stride=2))
#                 else:
#                     layers.append(TransposeBlock(self.in_planes, plane, stride=1))
#                 self.in_planes = plane
#
#         self.model = nn.Sequential(*layers)
#         self.relu = nn.ReLU()
#
#     def decode(self, z):
#         z = self.fc(z)
#         z = z.view(-1, 128, 2, 2)
#         z = self.model(z)
#         return z
#
#
#     def forward(self, z):
#         return self.decode(z)


# class PictureDecoder(nn.Module):
#     def __init__(self, rep_size=500):
#         super(PictureDecoder, self).__init__()
#         self.rep_size = rep_size
#         # Sampling vector
#         self.fc3 = nn.Linear(rep_size, rep_size)
#         self.fc_bn3 = nn.BatchNorm1d(rep_size)
#         self.fc4 = nn.Linear(rep_size, rep_size)
#         self.fc_bn4 = nn.BatchNorm1d(rep_size)
#
#         # Decoder
#         self.preconv = nn.ConvTranspose2d(125, 125, kernel_size=3, stride=1, padding=0, bias=False)
#         self.conv15 = nn.ConvTranspose2d(125, 128, kernel_size=2, stride=2, padding=0,  bias=False)
#         self.conv15_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn15 = nn.BatchNorm2d(128)
#         self.conv16 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv16_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn16 = nn.BatchNorm2d(128)
#         self.conv20 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv20_ = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1, bias=False)
#         self.bn20 = nn.BatchNorm2d(64)
#         self.conv17 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv17_ = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0, bias=False)
#         self.bn21 = nn.BatchNorm2d(32)
#         self.conv18 = nn.ConvTranspose2d(32, 16, kernel_size=40, stride=2, padding=0, bias=False)
#         self.conv18_ = nn.ConvTranspose2d(16, 3, kernel_size=40, stride=1, padding=0, bias=False)
#         self.bn22 = nn.BatchNorm2d(3)
#         self.conv19 = nn.ConvTranspose2d(3, 3, kernel_size=40, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU()
#
#     def decode(self, z):
#         out = self.fc_bn3(self.fc3(z))
#         out = self.relu(out)
#         out = self.fc_bn4(self.fc4(out))
#         out = self.relu(out).view(-1, 125, 2, 2)
#         out = self.relu(self.preconv(out))
#         out = self.relu(self.conv15(out))
#         out = self.relu(self.conv15_(out))
#         out = self.bn15(out)
#         out = self.relu(self.conv16(out))
#         out = self.relu(self.conv16_(out))
#         out = self.bn16(out)
#
#         out = self.relu(self.conv20(out))
#         out = self.relu(self.conv20_(out))
#         out = self.bn20(out)
#         out = self.relu(self.conv17(out))
#         out = self.relu(self.conv17_(out))
#         out = self.bn21(out)
#
#         out = self.relu(self.conv18(out))
#         out = self.relu(self.conv18_(out))
#         out = self.bn22(out)
#         out = self.conv19(out)
#         return out
#
#
#     def forward(self, z):
#         return self.decode(z)


class GeneralVae(nn.Module):
    def __init__(self, encoder_model, decoder_model, rep_size=500):
        super(GeneralVae, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar.mul(0.5))
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


class GeneralVaeBinding(nn.Module):
    def __init__(self, encoder_model, decoder_model, binding_model, rep_size=500):
        super(GeneralVaeBinding, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model
        self.binding = binding_model

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def calc_binding(self, z):
        return self.binding(z)

    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.calc_binding(z)


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
        x = torch.cat([x1, x2], dim=1)
        mu, logvar = (self.z_mean(x), self.z_log_var(x))
        z = self.reparam(logvar, mu)

        return z, mu, logvar

    def forward(self, x1, x2):
        z, mu, logvar = self.encode_latent_(x1, x2)

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

        # self.lmbd = Lambda(435, o)
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
                                nn.Linear(i, i), SELU(inplace=True), nn.Linear(i, i))

        self.logvar = nn.Sequential(nn.Linear(i, i),
                                    SELU(inplace=True),
                                    nn.Linear(i, i), SELU(inplace=True),
                                    nn.Linear(i, i), SELU(inplace=True), nn.Linear(i, i))

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
        y = eps.mul(std).add_(self.mu)
        return y

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, return_y=False):
        self.mu, self.log_v = self.encoder(x)
        self.mu, self.log_v = self.transformer(self.mu, self.log_v)
        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        y = eps.mul(std).add_(self.mu)
        if return_y:
            return y, self.decoder(y)
        return self.decoder(y)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.mu, self.log_v

        # bce = nn.BCELoss(size_average=True)
        bce = nn.MSELoss(reduction="sum")

        xent_loss = bce(x_decoded_mean, x.detach())
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
