import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from ResNet import ResNet, BasicBlock, Bottleneck
import math
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
        self.encoder = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=rep_size)
        self.mu = nn.Linear(rep_size, rep_size)
        self.logvar = nn.Linear(rep_size, rep_size)

    def forward(self, x):
        x = self.encoder(x)

        return self.mu(x), self.logvar(x)


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
        batch_size=x.shape[0]
        out = self.attention(x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size))
        return self.model(out.view(batch_size, -1))



class TranposeConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, padding=(0,0), stride=(0,0), kernel_size=(0,0), dropout=None):
        super(TranposeConvBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_plane, out_plane, kernel_size=kernel_size[0], padding=padding[0], stride=stride[0], bias=False)
        self.conv2 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1], stride=stride[1], bias=False)
        #self.conv3 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1], stride=stride[1], bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

#
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
#         self.preconv = nn.ConvTranspose2d(125, 128, kernel_size=3, stride=1, padding=0, bias=False)
#         self.conv15 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0,  bias=False)
#         self.conv15_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn15 = nn.BatchNorm2d(128)
#         self.conv16 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv16_ = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn16 = nn.BatchNorm2d(128)
#         self.conv20 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv20_ = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=1, bias=False)
#         self.bn20 = nn.BatchNorm2d(64)
#         self.conv17 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
#         self.conv17_ = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=0, bias=False)
#         self.bn21 = nn.BatchNorm2d(64)
#         self.conv18 = nn.ConvTranspose2d(64, 32, kernel_size=40, stride=2, padding=0, bias=False)
#         self.conv18_ = nn.ConvTranspose2d(32, 16, kernel_size=40, stride=1, padding=0, bias=False)
#         self.bn22 = nn.BatchNorm2d(16)
#         self.conv19 = nn.ConvTranspose2d(16, 3, kernel_size=40, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU()
#
#     def forward(self, z):
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
#
#
#         out = self.relu(self.conv18(out))
#         out = self.relu(self.conv18_(out))
#         out = self.bn22(out)
#         out = self.conv19(out)
#         return out

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
        return self.decode(z), mu, logvar, z

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

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class GridLSTMCell2d(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(GridLSTMCell2d, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        print(embed_dim + encoder_dim)
        print(decoder_dim)
        self.lstm1 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.lstm2 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)


    def forward(self, x, hs, ms): # x : (batch, embedded), H : (batch, 2, h), M : (batch, 2, m)
        H = hs[0] +hs[1]
        m1 = ms[0]
        m2 = ms[1]

        h1, m1 = self.lstm1(x, (H, m1))
        h2, m2 = self.lstm2(x, (H, m2))

        return h1, h2, m1, m2


class StackGridLSTMCell2d(nn.Module):
    """
    Decoder.
    """

    def __init__(self,  attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(StackGridLSTMCell2d, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.lstm1 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
        self.lstm2 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
        self.lstm3 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)



    def forward(self, x, hs, ms): # x : (batch, embedded), H : (batch, 2, h), M : (batch, 2, m)
        hsA1, hsB1, hsB2, hsB3 = hs
        msA1, msB1, msB2, msB3 = ms

        hsA2, hsB1prime, msA2, msB1prime = self.lstm1(x, [hsA1, hsB1], [msA1, msB1])
        hsA3, hsB2prime, msA3, msB2prime = self.lstm2(x,  [hsA2, hsB2], [msA2, msB2])
        hsA4, hsB3prime, msA4, msB3prime = self.lstm3(x, [hsA3, hsB3], [msA3, msB3])

        #need
        return hsA4, msA4, [hsB1prime, hsB2prime, hsB3prime], [msB1prime, msB2prime, msB3prime]




class GridLSTMDecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(GridLSTMDecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step1 = StackGridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths, teacher_forcing=True, use_first_state=False):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, m = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        newhs = [h, h, h]
        newms = [m, m, m]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).cuda()
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).cuda()

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])


            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = None
            #print("embedding size", embeddings.shape, attention_weighted_encoding.shape)

            if teacher_forcing and t > 0:
                lstm_input = torch.cat([self.embedding(torch.max(predictions[:batch_size_t, t, :], dim=1)[1].long()), attention_weighted_encoding], dim=1)
            else:
                lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)

            if use_first_state and t == 0:
                lstm_input = torch.cat([embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t]], dim=1)

            newhs.insert(0, h)
            newms.insert(0, m)
            newhs = map(lambda x : x[:batch_size_t, ...], newhs)
            newms = map(lambda x : x[:batch_size_t, ...], newms)

            #print("input shape", lstm_input.shape)
            #print('running grid ', t)
            h, m, newhs, newms = self.decode_step1(lstm_input, newhs, newms)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind