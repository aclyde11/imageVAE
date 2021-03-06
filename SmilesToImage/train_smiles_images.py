import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
from itertools import chain

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from model import GeneralVae,  PictureDecoder, PictureEncoder, TestVAE, DenseMolEncoder, ZSpaceTransform
import pickle
from PIL import  ImageOps
from utils import MS_SSIM
from invert import Invert

import numpy as np
import pandas as pd
starting_epoch=1
epochs = 200
no_cuda = False
seed = 42
data_para = False
log_interval = 10
LR = 0.001        ##adam rate
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
model_load = {'decoder' : '/homes/aclyde11/imageVAE/im_im_small/model/decoder_epoch_156.pt', 'encoder':'/homes/aclyde11/imageVAE/smi_smi/model/encoder_epoch_100.pt'}
#model_load = None
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/smi_im/results/'
save_files = '/homes/aclyde11/imageVAE/smi_im/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 12, 'pin_memory': True} if cuda else {}


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
        self.crispyLoss = MS_SSIM()

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cripsy = self.crispyLoss(x_recon, x)

        return loss_MSE + min(1.0, float(round(epochs / 2 + 0.75)) * KLD_annealing) * loss_KLD +  loss_cripsy


encoder = DenseMolEncoder()
decoder = torch.load(model_load['decoder'])
encoder_good =  torch.load('/homes/aclyde11/imageVAE/im_im_small/model/encoder_epoch_156.pt')


for param in encoder_good.parameters():
    param.requires_grad = False
for param in decoder.parameters():
    param.requires_grad = False

deocder = decoder.cuda().eval()
encoder = encoder.cuda()
encoder_good = encoder_good.cuda().eval()
transformer = ZSpaceTransform().cuda()
#decoder = torch.load(model_load['decoder'])
#model = TestVAE(encoder, decoder).cuda()

#
#
# if data_para and torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)


optimizer = optim.Adam(chain(encoder.parameters(), transformer.parameters()), lr=LR)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)

train_loader = generate_data_loader(train_root, 800, int(2000))
val_loader = generate_data_loader(val_root, 800, int(2000))
mse = customLoss()

val_losses = []
train_losses = []

def get_batch_size(epoch):
    return 800

def train(epoch):

    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    encoder.train()
    transformer.train()
    train_loss = 0
    for batch_idx, (data, embed, _) in enumerate(train_loader):
        data = data[0].cuda()
        embed = embed.cuda()
        z, logvar = encoder(embed)
        z, logvar = transformer(z, logvar)
        z_h, logvar_h = encoder_good(data)

        loss = 500 * (nn.MSELoss()(z, z_h) + nn.L1Loss()(logvar, logvar_h))
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data), datetime.datetime.now()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    train_losses.append(train_loss / len(train_loader.dataset))



def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

def test(epoch):
    encoder.eval()
    transformer.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, embed, smiles) in enumerate(val_loader):
            data = data[0].cuda()
            embed = embed.cuda()
            z, logvar = encoder(embed)
            z, logvar = encoder(embed)
            z, logvar = transformer(z, logvar)
            z_h, logvar_h = encoder_good(data)

            test_loss += (500 * (nn.MSELoss()(z, z_h) + nn.L1Loss()(logvar, logvar_h))).item()

            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            y = eps.mul(std).add_(z)
            recon_batch = decoder(y)



            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(get_batch_size(epoch), 3, 256, 256)[:n]])
                save_image(comparison.cpu(),
                           output_dir + 'reconstruction_' + str(epoch) + '.png', nrow=n)
                # n_image_gen = 8
                # images = []
                # n_samples_linspace = 16
                # for i in range(n_image_gen):
                #     data_latent = model.encode(embed)
                #     pt_1 = data_latent[i * 2, ...].cpu().numpy()
                #     pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
                #     sample_vec = interpolate_points(pt_1, pt_2, np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                #     sample_vec = torch.from_numpy(sample_vec).to(device)
                #     images.append(model.decode(sample_vec).cpu())
                # save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png', nrow=n_samples_linspace)
                #
                # n_image_gen = 8
                # images = []
                # n_samples_linspace = 16
                # for i in range(n_image_gen):
                #     data_latent = model.encode(embed)
                #     pt_1 = data_latent[i, ...].cpu().numpy()
                #     pt_2 = data_latent[i + 1, ...].cpu().numpy()
                #     sample_vec = interpolate_points(pt_1, pt_2,
                #                                     np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
                #     sample_vec = torch.from_numpy(sample_vec).to(device)
                #     images.append(model.decode(sample_vec).cpu())
                # save_image(torch.cat(images), output_dir + 'linspace_path_' + str(epoch) + '.png', nrow=n_samples_linspace)

    ##


    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))

    train(epoch)
    test(epoch)
    torch.save(encoder, save_files + 'encoder_epoch_' + str(epoch) + '.pt')
    torch.save(transformer, save_files + 'transformer_epoch_' + str(epoch) + '.pt')

    #torch.save(model.decoder, save_files + 'decoder_epoch_' + str(epoch) + '.pt')
    with torch.no_grad():
        sample = torch.randn(64, 500).to(device)
        sample = decoder(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   output_dir + 'sample_' + str(epoch-1) + '.png')

