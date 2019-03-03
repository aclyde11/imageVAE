import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import torch
import os
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from model import VAE_CNN

batch_size = 32 * 8
epochs = 50
no_cuda = False
seed = 5
data_para = True
log_interval = 50
LR = 0.01
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}

train_root = '/homes/aclyde11/imageVAE/draw2dPNG/train/'
val_root = '/homes/aclyde11/imageVAE/draw2dPNG/test/'

train_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(train_root, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

val_loader_food = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_root, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


model = VAE_CNN()
if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)
loss_mse = customLoss()

val_losses = []
train_losses = []


def train(epoch):
    # model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader_food):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item() / len(data), datetime.datetime.now()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_food.dataset)))
    train_losses.append(train_loss / len(train_loader_food.dataset))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(val_loader_food):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_mse(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 3, 256, 256)[:n]])
                save_image(comparison.cpu(),
                           '/homes/aclyde11/imageVAE/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)


for epoch in range(epochs + 1, 2 * epochs + 1):
    sched.step()
    train(epoch)
    test(epoch)
    torch.save(model.module.state_dict(), 'epoch_' + str(epoch) + '.pt')
    with torch.no_grad():
        sample = torch.randn(64, 2096).to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   '/homes/aclyde11/imageVAE/results/sample_' + str(epoch) + '.png')
