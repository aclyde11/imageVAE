import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2048 * 2, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class VAE_CNN(nn.Module):



    def __init__(self, rep_size = 2700):
        super(VAE_CNN, self).__init__()
        self.rep_size = rep_size

        # Encoder
        self.encoder = ResNet(BasicBlock, [3,4,6,3], num_classes=rep_size *2)

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

    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
