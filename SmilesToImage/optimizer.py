import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from model import GeneralVae,  PictureDecoder, PictureEncoder, BindingAffPredictor
import pickle
from PIL import  ImageOps
from utils import MS_SSIM
from invert import Invert

import numpy as np
import pandas as pd
starting_epoch=129
epochs = 200
no_cuda = False
seed = 42
data_para = False
log_interval = 50
LR = 0.0005          ##adam rate
rampDataSize = 0.2 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
#load_state = None
model_load = {'decoder' : '/homes/aclyde11/imageVAE/im_im_small/model/decoder_epoch_170.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im_small/model/encoder_epoch_170.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/im_opt/results/'
save_files = '/homes/aclyde11/imageVAE/im_opt/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
binding_affinity = pd.read_table("vina_results_norm.tab")

train_root = '/homes/aclyde11/moldata/moses/train/'
val_root =   '/homes/aclyde11/moldata/moses/test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab")

def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)
one_hot_encoded_fn = lambda row: np.array(map(lambda x: one_hot_array(x, len(vocab)),
                                     one_hot_index(row, vocab)))
def apply_one_hot(ch):
    return np.array(map(lambda x : np.pad(one_hot_encoded_fn(x), pad_width=[(0,60 - len(x)), (0,0)], mode='constant', constant_values=0), ch))

class ImageFolderWithFile(datasets.ImageFolder):
    def __getitem__(self, index):
        t = self.imgs[index][0]
        t = int(t.split('/')[-1].split('.')[0])
        i = t
        try:
            t = list(smiles_lookup.iloc[t, 1])
            bidningaff = list(binding_affinity[binding_affinity['id'] == i]['bindingaffinity'])
        except:
            print(t)
            exit()
        embed = apply_one_hot([t])[0].astype(np.float32)
        im = super(ImageFolderWithFile, self).__getitem__(index)

        return  im, embed, bidningaff

def generate_data_loader(root, batch_size, data_size):
    invert = transforms.Compose([
        Invert(),
        transforms.ToTensor()
    ])
    return torch.utils.data.DataLoader(
        ImageFolderWithFile(root, transform=invert),
        batch_size=batch_size, shuffle=False, drop_last=True, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, data_size))),  **kwargs)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.crispyLoss = MS_SSIM()

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cripsy = self.crispyLoss(x_recon, x)

        return loss_MSE + min(1.0, float(round(epochs / 2 + 0.75)) * KLD_annealing) * loss_KLD +  loss_cripsy

model = None
encoder = None
decoder = None
if model_load is None:
    encoder = PictureEncoder()
    decoder = BindingAffPredictor()
else:
    encoder = torch.load(model_load['encoder'])
    decoder = torch.load(model_load['decoder'])
model = GeneralVae(encoder, decoder, rep_size=500)


if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)
loss_mse = customLoss()


def get_batch_size(epoch):
    return min(64  + 2 * epoch, 322 )
train_loader_food = generate_data_loader(train_root, get_batch_size(200), int(5000))
val_loader_food = generate_data_loader(val_root, get_batch_size(200), int(5000))

val_losses = []
train_losses = []

tensors = []
def train(epoch):

    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    train_loss = 0
    for batch_idx, (data, _, ind) in enumerate(train_loader_food):
        data = data[0].cuda()
        optimizer.zero_grad()
        x = model(data)
        loss = nn.MSELoss()(x, ind)
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
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, ind) in enumerate(val_loader_food):
            data = data[0].cuda()
            x = model(data)
            test_loss += nn.MSELoss()(x, ind).item()


    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
    train(epoch)
    test(epoch)

    torch.save(model.module.encoder, save_files + 'encoder_epoch_' + str(epoch) + '.pt')
    torch.save(model.module.decoder, save_files + 'decoder_epoch_' + str(epoch) + '.pt')
    # with torch.no_grad():
    #     sample = torch.randn(64, 500).to(device)
    #     sample = model.module.decode(sample).cpu()
    #     save_image(sample.view(64, 3, 256, 256),
    #                output_dir + 'sample_' + str(epoch) + '.png')