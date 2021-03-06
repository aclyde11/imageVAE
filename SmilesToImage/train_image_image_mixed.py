import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from model import GeneralVae,  PictureDecoder, PictureEncoder, BindingAffModel, GeneralVaeBinding
import pickle
from PIL import  ImageOps
from utils import MS_SSIM
from invert import Invert

import numpy as np
import pandas as pd


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


starting_epoch=1
epochs = 200
no_cuda = False
seed = 42
data_para = True
log_interval = 50
LR = 7e-5          ##adam rate
rampDataSize = 0.15 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
KLD_annealing = 0.05  ##set to 1 if not wanted.
#load_state = None
model_load = None #{'decoder' : '/homes/aclyde11/imageVAE/im_im_small/model/decoder_epoch_128.pt', 'encoder':'/homes/aclyde11/imageVAE/im_im_small/model/encoder_epoch_128.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/mixed_im_im_small/results/'
save_files = '/homes/aclyde11/imageVAE/mixed_im_im_small/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}

binding_aff = pd.read_csv("/homes/aclyde11/moldata/moses/norm_binding_aff.csv")
binding_aff_orig = binding_aff
binding_aff['id'] = binding_aff['id'].astype('int64')
binding_aff = binding_aff.set_index('id')
print(binding_aff.head())

train_root = '/homes/aclyde11/moldata/moses/binding_train/'
val_root =   '/homes/aclyde11/moldata/moses/binding_test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab", names=['id', 'smiles'])
smiles_lookup = smiles_lookup.set_index('id')
print(smiles_lookup.head())

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
        f = self.imgs[index][0]
        f = int(f.split('/')[-1].split('.')[0])
        t=None
        aff=None
        try:
            aff = float(binding_aff.loc[f, 'norm_aff'])
            t = list(smiles_lookup.loc[f, 'smiles'])
        except:
            print(f)
            print('aff: ', aff)
            print('t', t)
            exit()
        #embed = apply_one_hot([t])[0].astype(np.float32)
        im = super(ImageFolderWithFile, self).__getitem__(index)

        return  im, 0, aff


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
encoder = PictureEncoder()
decoder = PictureDecoder()
binding_model = BindingAffModel()

checkpoint = torch.load('/homes/aclyde11/imageVAE/mixed_im_im_small/model/' + 'mixed_epoch_' + str(120) + '.pt', map_location="cuda:0")
#encoder.load_state_dict(checkpoint['encoder_state_dict'])
#decoder.load_state_dict(checkpoint['decoder_state_dict'])

model = GeneralVaeBinding(encoder, decoder, binding_model, rep_size=500).cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
if data_para and torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)



#binding_optimizer = optim.SGD(binding_model.parameters(), lr=5e-5, momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, nesterov=True)
#sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-6, last_epoch=-1)
#binding_sched = torch.optim.lr_scheduler.CosineAnnealingLR(binding_optimizer, 10, eta_min=5e-6, last_epoch=-1)
loss_picture = customLoss()
loss_mse = nn.MSELoss()
loss_mae = nn.L1Loss()
val_losses = []
train_losses = []

def get_batch_size(epoch):
    return min(64  + 8 * epoch, 450 )

def clip_gradient(optimizer, grad_clip=5.0):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train(epoch):
    train_loader_food = generate_data_loader(train_root, get_batch_size(epoch), int(rampDataSize * data_size))
    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    #binding_model.train()
    train_loss = 0
    loss = None
    for batch_idx, (data, _, aff) in enumerate(train_loader_food):
        data = data[0].cuda(0)
        aff = aff.float().cuda(0)

        optimizer.zero_grad()
        #binding_optimizer.zero_grad()

        recon_batch, mu, logvar, binding_pred = model(data)

        #binding_pred = binding_model(z.cuda(4))
        binding_loss = loss_mse(aff, binding_pred)
        binding_mae = loss_mae(aff, binding_pred)

        picture_loss = loss_picture(recon_batch, data, mu, logvar, epoch)
        loss = 1000.0 * binding_loss + picture_loss

        train_loss += loss.item()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
           scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)


        # binding_loss.backward()
        # binding_optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader_food.dataset),
                       100. * batch_idx / len(train_loader_food),
                       loss.item() / len(data), datetime.datetime.now()))
            print("BINDING LOSS: mse {}, mae {}, pictureLoss: {}".format(binding_loss.item(), binding_mae.item(), picture_loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader_food.dataset)))
    train_losses.append(train_loss / len(train_loader_food.dataset))
    return loss



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
    #binding_model.eval()
    test_loss = 0
    #binding_loss = 0
    #binding_mae = 0
    with torch.no_grad():
        for i, (data, _, aff) in enumerate(val_loader_food):
            data = data[0].cuda(0)
            aff = aff.float().cuda(0)

            optimizer.zero_grad()
            # binding_optimizer.zero_grad()

            recon_batch, mu, logvar, binding_pred = model(data)

            # binding_pred = binding_model(z.cuda(4))
            binding_loss = loss_mse(aff, binding_pred)
            binding_mae = loss_mae(aff, binding_pred)

            picture_loss = loss_picture(recon_batch, data, mu, logvar, epoch)
            loss = 1000.0 * binding_loss + picture_loss

            test_loss += loss.item()
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
    print("BINDING LOSS: mse {}, mae {}, pictureLoss: {}".format(binding_loss.item(), binding_mae.item(),
                                                                 picture_loss.item()))

    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))

    #binding_sched.step()
    #sched.step()

    loss = train(epoch)
    test(epoch)

    torch.save({
        'epoch': epoch,
        'encoder_state_dict': model.module.encoder.state_dict(),
        'decoder_state_dict' : model.module.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'binding_state_dict' : binding_model.state_dict(),
        #'binding_optimizer_state_dict' : binding_optimizer.state_dict(),
        'loss': loss}, save_files + 'mixed_epoch_' + str(epoch) + '.pt')
    with torch.no_grad():
        sample = torch.randn(64, 500).to(device)
        sample = model.module.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   output_dir + 'sample_' + str(epoch) + '.png')