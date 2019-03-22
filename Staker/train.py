from comet_ml import Experiment
from torch.nn.utils.rnn import pack_padded_sequence

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4,5,6,7'
from itertools import chain

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import pickle
from PIL import  ImageOps
from models import DecoderWithAttention, Encoder
from invert import Invert

import numpy as np
import pandas as pd


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
log_interval = 1
LR = hyper_params['learning_rate']       ##adam rate
rampDataSize = 0.2 ## data set size to use
embedding_width = 60
vocab = pickle.load( open( "/homes/aclyde11/moldata/charset.p", "rb" ) )
vocab.insert(0,' ')
vocab.insert(0, '!')
vocab.insert(0, '?')
vocab = {k: v for v, k in enumerate(vocab)}
charset = {k: v for v ,k in vocab.iteritems()}
embedding_width = 70
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
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


train_root = '/homes/aclyde11/moldata/moses/train/'
val_root =   '/homes/aclyde11/moldata/moses/test/'
smiles_lookup = pd.read_table("/homes/aclyde11/moldata/moses_cleaned.tab", header=None)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec):
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
        #embed = apply_one_hot([t])[0].astype(np.float32)
        t.insert(0, '!')
        t.append('?')
        caplen = len(t)
        while len(t) < 70:
            t.append(' ')
        embed = [vocab[i] for i in t]
        embed = torch.LongTensor(embed)
        return  super(ImageFolderWithFile, self).__getitem__(index), embed, caplen

def generate_data_loader(root, batch_size, data_size):
    invert = transforms.Compose([
        Invert(),
        transforms.ToTensor()
    ])
    return torch.utils.data.DataLoader(
        ImageFolderWithFile(root, transform=invert),
        batch_size=batch_size, shuffle=False, drop_last=True, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, data_size))),  **kwargs)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
fine_tune_encoder = True  # fine-tune encoder?
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=len(vocab),
                               dropout=dropout)

decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=decoder_lr)
encoder = Encoder()
encoder.fine_tune(fine_tune_encoder)
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=encoder_lr) if fine_tune_encoder else None

encoder = encoder.cuda()
decoder = decoder.cuda()

train_loader = generate_data_loader(train_root, 64, int(50000))
val_loader = generate_data_loader(val_root, 64, int(800))
criterion = nn.CrossEntropyLoss().to(device)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_batch_size(epoch):
    return 100

def train(epoch):
    with experiment.train():
        experiment.log_current_epoch(epoch)

        print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()
        losses = AverageMeter()  # loss (per word decoded)
        for batch_idx, (data, embed, embedlen) in enumerate(train_loader):
            imgs = data[0].float().cuda()
            caps = embed.cuda()
            caplens = embedlen.cuda().view(-1, 1)

            # Forward prop.
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            scores_copy = scores.clone()
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))

            if batch_idx % log_interval == 0:



                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(data), datetime.datetime.now()))

                print(scores_copy.shape)
                _, preds = torch.max(scores_copy, dim=2)

                #
                #
                # sampled = scores.cpu().detach()
                #
                # _, preds = torch.max(sampled, dim=2)
                # preds = preds.tolist()
                # temp_preds = list()
                # for j, p in enumerate(preds):
                #     temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                # preds = temp_preds
                # print(preds)

                # print(sampled.shape)
                # print(sampled)
                # mol = targets.cpu().numpy()
                # mol = decode_smiles_from_indexes(mol)
                # sampled = decode_smiles_from_indexes(sampled)
                # print("Orig: ", mol, " Sample: ", sampled, ' BCE: ')





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
                        mol = decode_smiles_from_indexes(mol)
                        sampled = decode_smiles_from_indexes(sampled)
                        print("Orig: ", mol, " Sample: ", sampled, ' BCE: ')

    experiment.log_metric("loss", test_loss)
    test_loss /= len(val_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)

for epoch in range(starting_epoch, epochs):
    train(epoch)
    #test(epoch)

