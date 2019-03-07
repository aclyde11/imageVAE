import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import SmilesToImageModle, SmilesEncoder, PictureDecoder
import pickle
import numpy as np
import pandas as pd
starting_epoch=1
epochs = 50
no_cuda = False
seed = 42
data_para = True
log_interval = 50
LR = 0.001           ##adam rate
rampDataSize = 0.23 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
KLD_annealing = 0.1  ##set to 1 if not wanted.
load_state = None
model_load = None
cuda = not no_cuda and torch.cuda.is_available()
data_size = 1000000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/SmilesToImage/results/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}


train_root = '/homes/aclyde11/imageVAE/draw2dPNG/train/'
val_root = '/homes/aclyde11/imageVAE/draw2dPNG/test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab")

def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

class ImageFolderWithFile(datasets.ImageFolder):
    def __getitem__(self, index):
        return  super(ImageFolderWithFile, self).__getitem__(index), self.imgs[index]

def generate_data_loader(root, batch_size, data_size):
    return torch.utils.data.DataLoader(
        ImageFolderWithFile(root, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return 1.25 * loss_MSE + min(1.0, float(round(epochs / 2 + 0.75)) * KLD_annealing) * loss_KLD

model = None
if model_load is None:
    model = SmilesToImageModle(SmilesEncoder(embedding_width, embedding_size, ), PictureDecoder())
else:
    model = torch.load(model_load)
if load_state is not None:
    model.load_state_dict(torch.load(load_state))

if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)
loss_mse = customLoss()

val_losses = []
train_losses = []

def get_batch_size(epoch):
    #return min(32 * epoch, 256 * 7)
    return 256 * 13


one_hot_encoded_fn = lambda row: np.array(map(lambda x: one_hot_array(x, len(vocab)),
                                     one_hot_index(row, vocab)))

def apply_one_hot(ch):
    return np.array(map(lambda x : np.pad(one_hot_encoded_fn(x), pad_width=[(0,60 - len(x)), (0,0)], mode='constant', constant_values=0), ch))
#apply_one_hot =  lambda ch: np.array(map(one_hot_encoded_fn, ch))


train_loader_food = generate_data_loader(train_root, get_batch_size(  3), int(rampDataSize * data_size))

def train(epoch):
    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    train_loss = 0
    for batch_idx, (data, file) in enumerate(train_loader_food):
        data = data[0]
        index = map(lambda x : int(x.split('/')[-1].split('.')[0]), file[0])
        index = list(smiles_lookup.iloc[index,1])
        embed = apply_one_hot(index)
        embed = torch.from_numpy(embed).float().cuda()
        data = data.cuda()

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(embed)
        loss = loss_mse(recon_batch, data, mu, logvar, epoch)
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



def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

def test(epoch):
    val_loader_food = generate_data_loader(val_root, get_batch_size(epoch), int(rampDataSize * data_size))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, file) in enumerate(val_loader_food):
            data = data[0]
            index = map(lambda x: x.split('/')[-1].split('.')[0], file[0])
            print(index)

            recon_batch, mu, logvar = model(data)
            test_loss += loss_mse(recon_batch, data, mu, logvar, epoch).item()
            if i == 0:
                n_image_gen = 8
                images = []
                n_samples_linspace = 16
                for i in range(n_image_gen):
                    data_latent = model.module.encode_latent_(data)
                    pt_1 = data_latent[i * 2, ...].cpu().numpy()
                    pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                    sample_vec = interpolate_points(pt_1, pt_2, np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                    sample_vec = torch.from_numpy(sample_vec).to(device)
                    images.append(model.module.decode(sample_vec).cpu())
                save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png', nrow=n_samples_linspace)

                n_image_gen = 8
                images = []
                n_samples_linspace = 16
                for i in range(n_image_gen):
                    data_latent = model.module.encode_latent_(data)
                    pt_1 = data_latent[i, ...].cpu().numpy()
                    pt_2 = data_latent[i + 1, ...].cpu().numpy()
                    sample_vec = interpolate_points(pt_1, pt_2,
                                                    np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                    sample_vec = torch.from_numpy(sample_vec).to(device)
                    images.append(model.module.decode(sample_vec).cpu())
                save_image(torch.cat(images), output_dir + 'linspace_path_' + str(epoch) + '.png', nrow=n_samples_linspace)

                ##
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(get_batch_size(epoch), 3, 256, 256)[:n]])
                save_image(comparison.cpu(),
                           output_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
    train(epoch)
    test(epoch)
    torch.save(model.module, 'epoch_' + str(epoch) + '.pt')
    with torch.no_grad():
        sample = torch.randn(64, 2700).sort()[0].to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   output_dir + 'sample_' + str(epoch) + '.png')
