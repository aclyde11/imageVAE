import os
from comet_ml import Experiment, ExistingExperiment

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
import torch.distributed as dist

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
parser.add_argument('-t', '--train_file', default='/homes/aclyde11/moses/data/train.csv', type=str)
parser.add_argument('-v', '--val_file', default='/homes/aclyde11/moses/data/test.csv', type=str)
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('-o', '--output_file', default=None, type=str)
args = parser.parse_args()
print("MY RANK IS: ", args.local_rank)
if args.local_rank == 0:
    experiment = Experiment(project_name='pytorch', auto_metric_logging=False)
from DataLoader import MoleLoader

starting_epoch=1
epochs = 500
seed = 42
log_interval = 25
LR = 8.0e-4         ##adam rate
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
embedding_size = len(vocab)
cuda = True
data_size = 1400000
torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}

smiles_lookup_train = pd.read_csv(args.train_file)
smiles_lookup_test = pd.read_csv(args.val_file)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        #self.crispyLoss = MS_SSIM()

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #loss_cripsy = self.crispyLoss(x_recon, x)

        return loss_MSE + loss_KLD #+ loss_cripsy

model = None
encoder = None
decoder = None
encoder = PictureEncoder(rep_size=256)
decoder = PictureDecoder()
#checkpoint = torch.load( save_files + 'epoch_' + str(48) + '.pt', map_location='cpu')
#encoder.load_state_dict(checkpoint['encoder_state_dict'])
#decoder.load_state_dict(checkpoint['decoder_state_dict'])

model = GeneralVae(encoder, decoder, rep_size=256).cuda()

model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)


print("LR: {}".format(LR))
optimizer = optim.Adam(model.parameters(), lr=LR)


for param_group in optimizer.param_groups:
   param_group['lr'] = LR

sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=8.0e-5, last_epoch=-1)


loss_picture = customLoss()
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
# val_data = MoleLoader(smiles_lookup_test)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
# val_sampler =   torch.utils.data.distributed.DistributedSampler(val_data)
train_loader_food = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, sampler=train_sampler, drop_last=True, #sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(train_data), size=1250000))))),
    **kwargs)
# val_loader_food = torch.utils.data.DataLoader(
#         val_data,
#         batch_size=args.batch_size, sampler=val_sampler, drop_last=True, #sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(val_data), size=10000))))),
#         **kwargs)

def train(epoch, size=100000):
    # train_loader_food = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=torch.utils.data.SubsetRandomSampler(indices=list(set(list(np.random.randint(0, len(train_data), size=size))))),
    #     **kwargs)

    experiment.log_current_epoch(epoch)
    print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
    model.train()
    for batch_idx, (_, data, _) in enumerate(train_loader_food):
        data = data.float().cuda()

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_picture(recon_batch, data, mu, logvar, epoch)
        loss.backward()

        clip_gradient(optimizer, grad_clip=args.grad_clip)
        optimizer.step()



        if args.local_rank == 0 and batch_idx % log_interval == 0:
            with experiment.train():
                reduced_loss = reduce_tensor(loss.data)
                reduced_loss=float(reduced_loss)
                torch.cuda.synchronize()

                experiment.log("lossred", reduced_loss)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                    epoch, batch_idx * len(data), len(train_loader_food.dataset),
                           100. * batch_idx / len(train_loader_food), datetime.datetime.now()))


    return None


def open_ball(i, x, eps):
    x = x.view(1, 256)

    x = x.repeat(9, 1)
    print(x.shape)

    for j in range(4):
        x[0 + j, i] = x[j, i]  + (4-j) * 5

    # 4 stays same
    for j in range(5, 9):
        x[0 + j, i] = x[j, i] +  (j-4) * 5
    return x

def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)

def test(epoch):

    model.eval()
    losses = AverageMeter()
    test_loss = 0
    with torch.no_grad():
        for i, (_, data, _) in enumerate(val_loader_food):
            data = data.float().cuda()
            recon_batch, mu, logvar = model(data)


            loss2 = loss_picture(recon_batch, data, mu, logvar, epoch)
            losses.update(loss2.item(), int(data.shape[0]))
            test_loss += loss2.item()


    test_loss /= len(val_loader_food.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    experiment.log_metric('loss', losses.avg)

    val_losses.append(test_loss)



torch.cuda.set_device(args.gpu)
args.world_size = torch.distributed.get_world_size()




for epoch in range(starting_epoch, epochs):
    if epoch == starting_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR

    sched.step()

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = LR
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))
        experiment.log_metric('lr', param_group['lr'])

    loss = train(epoch)
    #test(epoch)

    torch.save({
        'epoch': epoch,
        'ex_id' : experiment.get_key(),
        'encoder_state_dict': model.module.encoder.state_dict(),
        'decoder_state_dict' : model.module.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
         }, args.output_file + 'epoch.pt')
