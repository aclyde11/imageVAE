import os
import sys
import cv2
import math
import random, string

import numpy as np
from scipy.stats import norm
from sklearn import manifold
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import GeneralVae, SmilesEncoder, PictureDecoder, PictureEncoder
import pickle
from utils import MS_SSIM
import numpy as np
import pandas as pd
starting_epoch=52
epochs = 200
no_cuda = False
seed = 42
data_para = True
log_interval = 50
LR = 0.001          ##adam rate
rampDataSize = 0.33 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
load_state = None
model_load = {'decoder' : '/homes/aclyde11/imageVAE/im_im/model/decoder_epoch_51.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im/model/encoder_epoch_51.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/im_im/video/'
save_files = '/homes/aclyde11/imageVAE/im_im/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}


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
        try:
            t = list(smiles_lookup.iloc[t, 1])
        except:
            print(t)
            exit()
        embed = apply_one_hot([t])[0].astype(np.float32)
        return  super(ImageFolderWithFile, self).__getitem__(index), embed

def generate_data_loader(root, batch_size, data_size):
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(root, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, data_size))),  **kwargs)


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



#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, nesterov=True)
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.000001, last_epoch=-1)
loss_mse = customLoss()

val_losses = []
train_losses = []

def get_batch_size(epoch):
    return min(16 * epoch, 512 )

def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

def sample(epoch, model, data):
    model.eval()
    with torch.no_grad():
        n_image_gen = 8
        images = []
        n_samples_linspace = 16
        for i in range(n_image_gen):
            data_latent = model.encode_latent_(data)
            pt_1 = data_latent[i * 2, ...].cpu().numpy()
            pt_2 = data_latent[i * 2 + 1, ...].cpu().numpy()
            sample_vec = interpolate_points(pt_1, pt_2, np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
            sample_vec = torch.from_numpy(sample_vec).to(device)
            images.append(model.decode(sample_vec).cpu())
        save_image(torch.cat(images), output_dir + 'linspace_' + str(epoch) + '.png', nrow=n_samples_linspace)

        n_image_gen = 8
        images = []
        n_samples_linspace = 16
        for i in range(n_image_gen):
            data_latent = model.encode_latent_(data)
            pt_1 = data_latent[i, ...].cpu().numpy()
            pt_2 = data_latent[i + 1, ...].cpu().numpy()
            sample_vec = interpolate_points(pt_1, pt_2,
                                            np.linspace(0, 1, num=n_samples_linspace, endpoint=True))
            sample_vec = torch.from_numpy(sample_vec).to(device)
            images.append(model.decode(sample_vec).cpu())
        save_image(torch.cat(images), output_dir + 'linspace_path_' + str(epoch) + '.png', nrow=n_samples_linspace)


def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i] * 255.
        img = img.astype(np.uint8).reshape([3, 256, 256])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

def sample_plot(epoch, model, data):
    model.eval()
    data = data.cuda()
    # Compute latent space representation
    print("Computing latent space projection...")
    mu, logvar = model.encoder(data)
    X_encoded = model.reparameterize(mu, logvar).cpu().detach().numpy()
    data = data.cpu().numpy()
    print("gt latent")
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    print("Plotting t-SNE visualization...")
    fig, ax = plt.subplots()
    imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=data, ax=ax, zoom=0.6)
    plt.savefig('books_read.png')


for epoch in range(starting_epoch, epochs):
    loader = generate_data_loader(val_root, 100, int(20000))
    #train(epoch)
    data = None
    for d, _ in loader:
        data = d
        break
    data = data.cuda()
    for i in range(50, epoch):

        encoder = torch.load('/homes/aclyde11/imageVAE/im_im/model/encoder_epoch_' + str(i)+ '.pt')
        decoder = torch.load('/homes/aclyde11/imageVAE/im_im/model/decoder_epoch_' + str(i)+ '.pt')
        model = GeneralVae(encoder, decoder).cuda()
        #sample(i, model, data)
        sample_plot(i, model, data)
        del model





