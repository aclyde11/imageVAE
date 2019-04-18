import os
from comet_ml import Experiment

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from model import GeneralVae,  PictureDecoder, PictureEncoder, PixelCNN
import pickle
from PIL import  ImageOps
from utils import MS_SSIM, AverageMeter
from invert import Invert
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import pandas as pd
from PIL import Image
import io
import cairosvg
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-w', '--workers', default=16, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-g', '--grad-clip', default=2.0, type=float,
                    metavar='N', help='mini-batch size per process (default: 256)')

args = parser.parse_args()


from DataLoader import MoleLoader
experiment = Experiment(project_name="pytorch")

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

starting_epoch=1
epochs = 500
no_cuda = False
seed = 42
data_para = True
log_interval = 25
LR = 8.0e-4         ##adam rate
rampDataSize = 0.3 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
#load_state = None
model_load = None #{'decoder' : '/homes/aclyde11/imageVAE/im_im_small/model/decoder_epoch_128.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im_small/model/encoder_epoch_128.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/im_im_bw/results/'
save_files = '/homes/aclyde11/imageVAE/im_im_bw/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}

binding_aff = pd.read_csv("/homes/aclyde11/moldata/moses/norm_binding_aff.csv")
binding_aff_orig = binding_aff
binding_aff['id'] = binding_aff['id'].astype('int64')
binding_aff = binding_aff.set_index('id')
print(binding_aff.head())

smiles_lookup = pd.read_csv("/homes/aclyde11//moses/data/train.csv")
print(smiles_lookup.head())
def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)
one_hot_encoded_fn = lambda row: np.array(map(lambda x: one_hot_array(x, len(vocab)),
                                     one_hot_index(row, vocab)))
def apply_one_hot(ch):
    return np.array(map(lambda x : np.pad(one_hot_encoded_fn(x), pad_width=[(0,60 - len(x)), (0,0)], mode='constant', constant_values=0), ch))

smiles_lookup_train = pd.read_csv("/homes/aclyde11/moses/data/train.csv")
print(smiles_lookup_train.head())
smiles_lookup_test = pd.read_csv("/homes/aclyde11/moses/data/test.csv")
print(smiles_lookup_test.head())



val_loader_food = torch.utils.data.DataLoader(
        MoleLoader(smiles_lookup_test),
        batch_size=args.batch_size, shuffle=False, drop_last=True,  **kwargs)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.crispyLoss = MS_SSIM()

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cripsy = self.crispyLoss(x_recon, x)

        return loss_MSE + loss_KLD + 10000.0 * loss_cripsy

model = None
encoder = None
decoder = None
encoder = PictureEncoder(rep_size=256)
decoder = PictureDecoder()
#checkpoint = torch.load( save_files + 'epoch_' + str(8) + '.pt', map_location='cpu')
#encoder.load_state_dict(checkpoint['encoder_state_dict'])
#decoder.load_state_dict(checkpoint['decoder_state_dict'])

model = GeneralVae(encoder, decoder, rep_size=256).cuda()


print("LR: {}".format(LR))
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.8, nesterov=True)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for param_group in optimizer.param_groups:
   param_group['lr'] = LR

if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

loss_picture = customLoss()

if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    loss_picture = nn.DataParallel(loss_picture)
loss_picture.cuda()
val_losses = []
train_losses = []

def get_batch_size(epoch):
    return args.batch_size

def clip_gradient(optimizer, grad_clip=1.0):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

train_data = MoleLoader(smiles_lookup_train)
# train_loader_food = torch.utils.data.DataLoader(
#     train_data,
#     batch_size=args.batch_size, shuffle=True, drop_last=True,
#     **kwargs)


train_loader_food = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=True, drop_last=True, #sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(train_data), size=1250000))))),
    **kwargs)

val_data = MoleLoader(smiles_lookup_test)

val_loader_food = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True, drop_last=True, #sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(val_data), size=10000))))),
        **kwargs)

def train(epoch, size=100000):
    # train_loader_food = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(train_data), size=size))))),
    #     **kwargs)

    with experiment.train():
        experiment.log_current_epoch(epoch)
        print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (_, data, _) in enumerate(train_loader_food):
            data = data.float().cuda()

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)

            loss = loss_picture(recon_batch, data, mu, logvar, epoch)
            loss = torch.sum(loss)
            loss_meter.update(loss.item(), int(recon_batch.shape[0]))
            experiment.log_metric('loss', loss_meter.avg)

            loss.backward()

            clip_gradient(optimizer, grad_clip=args.grad_clip)
            optimizer.step()


            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                    epoch, batch_idx * len(data), len(train_loader_food.dataset),
                           100. * batch_idx / len(train_loader_food),
                           loss_meter.avg, datetime.datetime.now()))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, loss_meter.avg))
    return loss_meter.avg



def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

def test(epoch):
    with experiment.test():

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (_, data, _) in enumerate(val_loader_food):
                data = data.float().cuda()
                #aff = aff.float().cuda(4)


                recon_batch, mu, logvar = model(data)


                loss = loss_picture(recon_batch, data, mu, logvar, epoch)
                loss = torch.sum(loss)

                experiment.log_metric('loss', loss.item())
                test_loss += loss.item()
                if i == 0:
                    ##
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(get_batch_size(epoch), 3, 256, 256)[:n]])
                    save_image(comparison.cpu(),
                               output_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)

                    del recon_batch
                    del mu
                    del logvar

                    n_image_gen = 8
                    images = []
                    n_samples_linspace = 16
                    data_latent = model.module.encode_latent_(data[:25, ...])

                    for i in range(n_image_gen):
                        pt_1 = data_latent[i * 2, ...].cpu().numpy()
                        pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                        sample_vec = interpolate_points(pt_1, pt_2, np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                        sample_vec = torch.from_numpy(sample_vec).to(device)
                        images.append(model.module.decode(sample_vec).cpu())
                    save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png', nrow=n_samples_linspace)





        test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    val_losses.append(test_loss)


for epoch in range(starting_epoch, epochs):


    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
        experiment.log_metric('lr', param_group['lr'])

    loss = train(epoch)
    test(epoch)

    torch.save({
        'epoch': epoch,
        'encoder_state_dict': model.module.encoder.state_dict(),
        'decoder_state_dict' : model.module.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
         }, save_files + 'epoch_' + str(epoch) + '.pt')
    with torch.no_grad():
        sample = torch.randn(64, 256).to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   output_dir + 'sample_' + str(epoch) + '.png')
