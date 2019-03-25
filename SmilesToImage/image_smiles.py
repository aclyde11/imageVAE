from comet_ml import Experiment

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4,5,6,7'
from itertools import chain

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from model import GeneralVae,  PictureDecoder, PictureEncoder, TestVAE, DenseMolEncoder, ZSpaceTransform, ComboVAE, MolDecoder, AutoModel
import pickle
from PIL import  ImageOps
from utils import MS_SSIM
from invert import Invert

import numpy as np
import pandas as pd

torch.cuda.set_device(5)

hyper_params = {
    "num_epochs": 1000,
    "train_batch_size": 28,
    "val_batch_size": 128,
    'seed' : 42,
    "learning_rate": 0.001
}


experiment = Experiment(project_name="pytorch")
experiment.log_parameters(hyper_params)

starting_epoch=1
epochs = hyper_params['num_epochs']
no_cuda = False
seed = hyper_params['seed']
data_para = True
log_interval = 7
LR = hyper_params['learning_rate']       ##adam rate
rampDataSize = 0.2 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
vocab.insert(0,' ')
print(vocab)
embedding_width = 60
embedding_size = len(vocab)
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
#load_state = None
model_load1 = {'decoder' : '/homes/aclyde11/imageVAE/combo/model/decoder1_epoch_111.pt', 'encoder':'/homes/aclyde11/imageVAE/smi_smi/model/encoder_epoch_100.pt'}
#model_load = None
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/combo/results/'
save_files = '/homes/aclyde11/imageVAE/combo/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}


train_root = '/homes/aclyde11/moldata/moses/train/'
val_root =   '/homes/aclyde11/moldata/moses/test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab", header=None)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)
one_hot_encoded_fn = lambda row: np.array(map(lambda x: one_hot_array(x, len(vocab)),
                                     one_hot_index(row, vocab)))
def apply_t(x):
    x = x + list((''.join([char*(embedding_width - len(x)) for char in [' ']])))
    smi = one_hot_encoded_fn(x)
    return smi

def apply_one_hot(ch):
    return np.array(map(apply_t, ch))

class ImageFolderWithFile(datasets.ImageFolder):
    def __getitem__(self, index):
        t = self.imgs[index][0]
        t = int(t.split('/')[-1].split('.')[0])
        t = list(smiles_lookup.iloc[t, 1])
        embed = apply_one_hot([t])[0].astype(np.float32)
        return  super(ImageFolderWithFile, self).__getitem__(index), embed, t

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
        #self.crispyLoss = MS_SSIM()

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #loss_cripsy = self.crispyLoss(x_recon, x)
        loss_cripsy = 0
        return loss_MSE + min(1.0, float(round(epochs / 2 + 0.75)) * KLD_annealing) * loss_KLD +  loss_cripsy


encoder = PictureEncoder()
decoder = PictureDecoder()
model = GeneralVae(encoder, decoder).cuda()


# if data_para and torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)


optimizer = optim.Adam(model.parameters(), lr=LR)


train_loader = generate_data_loader(train_root, 4, int(50000))
val_loader = generate_data_loader(val_root, 4, int(800))
val_losses = []
train_losses = []
lossf = customLoss()

def get_batch_size(epoch):
    return 100

def train(epoch):
    with experiment.train():
        experiment.log_current_epoch(epoch)

        print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
        model.train()
        train_loss = 0
        for batch_idx, (data, embed, _) in enumerate(train_loader):
            data = data[0].float().cuda()
            #embed = embed.float().cuda()
            recon_batch, mu, logvar = model(data)

            loss = lossf(recon_batch, data, mu, logvar, epoch)

            experiment.log_metric("loss", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % log_interval == 0:


                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data), datetime.datetime.now()))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        train_losses.append(train_loss / len(train_loader.dataset))




def test(epoch):
    with experiment.test():
        experiment.log_current_epoch(epoch)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, embed, smiles) in enumerate(val_loader):
                data = data[0].float().cuda()
                embed = embed.float().cuda()
                recon_batch = model(data)

                loss = lossf(recon_batch.float(), embed)

                test_loss += loss.item()

                if i == 0:
                    for i in range(4):
                        sampled = recon_batch.cpu().detach().numpy()[i, ...].argmax(axis=1)
                        mol = embed.cpu().numpy()[i, ...].argmax(axis=1)
                        mol = decode_smiles_from_indexes(mol, vocab)
                        sampled = decode_smiles_from_indexes(sampled, vocab)
                        print("Orig: ", mol, " Sample: ", sampled, ' BCE: ')

    experiment.log_metric("loss", test_loss)
    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):


    if epoch > 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
    train(epoch)
    test(epoch)

