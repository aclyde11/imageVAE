import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import GeneralVae, MolEncoder, MolDecoder
import pickle
from torch.nn import init
torch.set_printoptions(profile="full")

from utils import MS_SSIM
import numpy as np
import pandas as pd
starting_epoch=21
epochs = 200
no_cuda = False
seed = 42
data_para = True
log_interval = 25
LR = 0.001          ##adam rate
rampDataSize = 0.1 ## data set size to use
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
vocab.insert(0,' ')
print(vocab)
embedding_width = 60
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
load_state = None
#model_load = None
model_load = {'decoder' : '/homes/aclyde11/imageVAE/smi_smi/model/decoder_epoch_20.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im/model/encoder_epoch_20.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/smi_smi/results/'
save_files = '/homes/aclyde11/imageVAE/smi_smi/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}


train_root = '/homes/aclyde11/moldata/moses/train/'
val_root =   '/homes/aclyde11/moldata/moses/test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab")

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
        t = list(smiles_lookup.iloc[t-1, 1])
        embed = apply_one_hot([t])[0].astype(np.float32)
        return  super(ImageFolderWithFile, self).__getitem__(index), embed

def generate_data_loader(root, batch_size, data_size):
    return torch.utils.data.DataLoader(
        ImageFolderWithFile(root, transform=transforms.ToTensor()),
        batch_size=batch_size, drop_last=True, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, data_size))),  **kwargs)


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        #self.mse_loss = nn.MSELoss(reduction="sum")
        self.bce_loss = nn.BCELoss(size_average=True)

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = embedding_width * self.bce_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD


model = None
encoder = None
decoder = None
if model_load is None:
    encoder = MolEncoder()
    decoder = MolDecoder()
else:
    encoder = torch.load(model_load['encoder'])
    decoder = torch.load(model_load['decoder'])
model = GeneralVae(encoder, decoder, rep_size=500)

if starting_epoch ==1 :
    def initialize_weights(m):
        if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)):
            init.xavier_uniform(m.weight.data)
        elif isinstance(m, nn.GRU):
            for weights in m.all_weights:
                for weight in weights:
                    if len(weight.size()) > 1:
                        init.xavier_uniform(weight.data)

    model.apply(initialize_weights)

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
    #return min(16 * epoch, 512)
    return min(1024 + epoch * 64, 8069)


def train(epoch):
    train_loader_food = generate_data_loader(train_root, get_batch_size(epoch), int(rampDataSize * data_size))

    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    train_loss = 0
    for batch_idx, (_, embed) in enumerate(train_loader_food):
        embed = embed.float().cuda()
        recon_batch, mu, logvar = model(embed)
        loss = loss_mse(recon_batch, embed, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(embed), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item(), datetime.datetime.now()))
            for i in range(3):
                sampled = recon_batch.cpu().detach().numpy()[i, ...].argmax(axis=1)
                mol = embed.cpu().numpy()[i, ...].argmax(axis=1)
                mol = decode_smiles_from_indexes(mol, vocab)
                sampled = decode_smiles_from_indexes(sampled, vocab)
                print("Orig: ", mol, " Sample: ", sampled, ' BCE: ', loss_mse(recon_batch, embed, mu, logvar).item())


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
    val_loader_food = generate_data_loader(val_root, get_batch_size(epoch), int(5000))
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, (_, embed) in enumerate(val_loader_food):
            embed = embed.cuda()

            recon_batch, mu, logvar = model(embed)
            # for i in range(recon_batch.shape[0]):
            #     sampled = recon_batch.cpu().numpy()[i, ...].argmax(axis=1)
            #     mol = embed.cpu().numpy()[i, ...].argmax(axis=1)
            #     mol = decode_smiles_from_indexes(mol, vocab)
            #     sampled = decode_smiles_from_indexes(sampled, vocab)
            #     print("Orig: ", mol, " Sample: ", sampled)
            test_loss += loss_mse(recon_batch, embed, mu, logvar).item()


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
    #     sample = torch.randn(64, 2000).to(device)
    #     sample = model.module.decode(sample).cpu()
    #     save_image(sample.view(64, 3, 256, 256),
    #                output_dir + 'sample_' + str(epoch) + '.png')
