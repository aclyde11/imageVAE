from comet_ml import Experiment
from torch.nn.utils.rnn import pack_padded_sequence
from DataLoader import MoleLoader
import os

from itertools import chain

import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
import pickle
from PIL import ImageOps
from models import GridLSTMDecoderWithAttention, Encoder
from model import GeneralVae, PictureEncoder, PictureDecoder
from invert import Invert

import numpy as np
import pandas as pd

import sys

gpu1, gpu2 = sys.argv[1], sys.argv[2]

torch.cuda.set_device(2)
hyper_params = {
    "num_epochs": 1000,
    "train_batch_size": 28,
    "val_batch_size": 128,
    'seed': 42,
    "learning_rate": 0.001
}

experiment = Experiment(project_name="grid-lstm", disabled=False)
experiment.log_parameters(hyper_params)
batch_size = 256
starting_epoch = 1
epochs = hyper_params['num_epochs']
no_cuda = False
seed = hyper_params['seed']
data_para = True
log_interval = 5
LR = hyper_params['learning_rate']  ##adam rate
rampDataSize = 0.2  ## data set size to use
embedding_width = 70
vocab = pickle.load(open("/homes/aclyde11/zinc/vocab_cleaned.pkl", "rb"))
vocab = {k: v for v, k in enumerate(vocab)}
charset = {k: v for v, k in vocab.items()}

# model_load1 = {'decoder' : '/homes/aclyde11/imageVAE/combo/model/decoder1_epoch_111.pt', 'encoder':'/homes/aclyde11/imageVAE/smi_smi/model/encoder_epoch_100.pt'}
cuda = True
data_size = 1400000
torch.manual_seed(seed)
output_dir = '/homes/aclyde11/imageVAE/combo/results/'
save_files = '/homes/aclyde11/imageVAE/combo/model/'
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 32, 'pin_memory': True} if cuda else {}

train_data = MoleLoader(
    pd.read_csv("/homes/aclyde11/zinc/zinc_cleaned.smi", sep=' ', header=None, engine='c', low_memory=False), vocab,
    max_len=70)
val_data = MoleLoader(pd.read_csv("/homes/aclyde11/zinc/zinc_cleaned.smi", sep=' ', header=None, engine='c', low_memory=False),
                      vocab, max_len=70)

train_loader_food = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
val_loader_food = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

vocab = train_data.vocab
charset = train_data.charset
embedding_width = 150
embedding_size = len(vocab)


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


emb_dim = 48  # dimension of word embeddings
attention_dim = 256  # dimension of attention linear layers
decoder_dim = 256  # dimension of decoder RNN
dropout = 0.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
encoder_lr = 5e-4  # learning rate for encoder if fine-tuning
decoder_lr = 5e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
fine_tune_encoder = True  # fine-tune encoder?

decoder = GridLSTMDecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(vocab),
                                       encoder_dim=512,
                                       dropout=dropout)

decoder.fine_tune_embeddings(True)
decoder = decoder.cuda(gpu2)

decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=decoder_lr)
encoder = Encoder()
encoder.fine_tune(fine_tune_encoder)
encoder = encoder.cuda(gpu1)

encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=encoder_lr) if fine_tune_encoder else None

decoder_sched = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, 8, eta_min=5e-6, last_epoch=-1)
encoder_sched = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, 8, eta_min=5e-6, last_epoch=-1)

criterion = nn.CrossEntropyLoss().cuda(gpu2)


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
    return batch_size


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def add_text_to_image(ten, text, which="orig", dis=None):
    from PIL import ImageFont
    from PIL import ImageDraw
    img = transforms.ToPILImage(mode='RGB')(ten)
    img = Invert()(img)
    draw = ImageDraw.Draw(img)
    # ont = ImageFont.truetype(<font-file>, <font-size>)
    sfont = ImageFont.truetype("Vera.ttf", 9)
    font = ImageFont.truetype("Vera.ttf", 11)

    draw.text((0, 0), text, (0, 0, 0), font=sfont)
    if which is not None:
        draw.text((225, 225), which, (0, 0, 0), font=font)
    if dis is not None:
        draw.text((0, 225), "edit: " + dis, (0, 0, 0), font=font)
    img.convert('RGB')
    return transforms.ToTensor()(img).float().view(1, 3, 256, 256)


def train(epoch):
    with experiment.train():
        experiment.log_current_epoch(epoch)

        print("Epoch {}: batch_size {}".format(epoch, get_batch_size(epoch)))
        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()
        losses = AverageMeter()  # loss (per word decoded)
        corrects = []
        wrongs = []
        for batch_idx, (embed, data, embedlen) in enumerate(train_loader_food):
            rangeobj = range(1, 2)

            for which_image in rangeobj:

                imgs = data.float()
                imgs_orig = imgs
                caps = embed.cuda(gpu2)
                caplens = embedlen.cuda(gpu2).view(-1, 1)
                imgs = imgs.cuda(gpu1)

                # Forward prop.
                imgs = encoder(imgs).cuda(gpu2)
                print(imgs.shape)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens,
                                                                                teacher_forcing=bool(epoch < 3))

                scores_copy = scores.clone()
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                imgs_vae = imgs_vae[sort_ind]
                imgs_orig = imgs_orig[sort_ind]
                targets = caps_sorted[:, 1:]
                targets_copy = targets.clone()

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

                experiment.log_metric('loss', loss.item())
                if which_image == 0:
                    experiment.log_metric("orig_loss", loss.item())
                else:
                    experiment.log_metric("vae_loss", loss.item())

                acc = torch.max(scores, dim=1)[1].eq(targets).sum().item() / float(targets.shape[0])
                experiment.log_metric("acc_per_char", acc)

                acc_per_string = 0
                if batch_idx % log_interval == 0:
                    print("wrongs len: {}, correct len: {}".format(len(wrongs), len(corrects)))
                    print('Train Epoch {}: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                        "orig" if which_image == 0 else "vaes",
                        epoch, batch_idx * len(data), len(train_loader_food.dataset),
                        100. * batch_idx / len(train_loader_food),
                        loss.item() / len(data), datetime.datetime.now()))

                    _, preds = torch.max(scores_copy, dim=2)
                    preds = preds.cpu().numpy()
                    targets_copy = targets_copy.cpu().numpy()

                    imgs_orig = imgs_orig.detach().cpu()
                    imgs_vae = imgs_vae.detach().cpu()
                    for i in range(preds.shape[0]):
                        sample = preds[i, ...]
                        target = targets_copy[i, ...]
                        s1 = "".join([charset[chars] for chars in target]).strip()
                        s2 = "".join([charset[chars] for chars in sample]).strip()
                        if i < 4:
                            print("ORIG: {}\nNEW : {}\n".format(s1, s2))
                        acc_per_string += 1 if s1 == s2 else 0

                        if len(corrects) < 50 and s1 == s2:
                            a = add_text_to_image(imgs_orig[i, ...], s1, "orig")
                            corrects.append(a)
                            a = add_text_to_image(imgs_vae[i, ...], s2, "vae", str(0))
                            corrects.append(a)

                        if len(wrongs) < 50 and s1 != s2:
                            dist = levenshteinDistance(s1, s2)
                            s2 = s2
                            a = add_text_to_image(imgs_orig[i, ...], s1, "orig")
                            wrongs.append(a)
                            a = add_text_to_image(imgs_vae[i, ...], s2, "vae", str(dist))
                            wrongs.append(a)

                    if which_image == 0:
                        experiment.log_metric('orig_acc_per_string', float(acc_per_string) / float(preds.shape[0]))
                    else:
                        experiment.log_metric('vaes_acc_per_string', float(acc_per_string) / float(preds.shape[0]))

        if len(corrects) == 50:
            save_image(torch.cat(corrects), "corrects_" + str(epoch) + ".png", nrow=10)
        if len(wrongs) == 50:
            save_image(torch.cat(wrongs), "wrongs_" + str(epoch) + ".png", nrow=10)

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
        decoder.eval()
        encoder.eval()
        losses = AverageMeter()  # loss (per word decoded)
        with torch.no_grad():
            for batch_idx, (embed, data, embedlen) in enumerate(val_loader_food):
                for which_image in range(2):

                    imgs = data.float()
                    caps = embed.cuda(gpu2)
                    caplens = embedlen.cuda(gpu2).view(-1, 1)

                    imgs = imgs.cuda(gpu1)

                    imgs = encoder(imgs).cuda(gpu2)

                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens,
                                                                                    teacher_forcing=bool(epoch > 1))

                    scores_copy = scores.clone()
                    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                    targets = caps_sorted[:, 1:]
                    targets_copy = targets.clone()
                    # Remove timesteps that we didn't decode at, or are pads
                    # pack_padded_sequence is an easy trick to do this
                    # print(caplens)
                    # for i in range(4):
                    #     print(scores_copy[i, ...].shape)
                    #     print(targets_copy[i, ...].shape)
                    #     print(decode_lengths[i])

                    scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                    targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                    # Calculate loss
                    loss = criterion(scores, targets)

                    # Add doubly stochastic attention regularization
                    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                    # Keep track of metrics
                    losses.update(loss.item(), sum(decode_lengths))

                    acc = torch.max(scores, dim=1)[1].eq(targets).sum().item() / float(targets.shape[0])
                    experiment.log_metric("acc_per_char", acc)
                    if batch_idx % log_interval == 0:

                        print('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                            epoch, batch_idx * len(data), len(val_loader_food.dataset),
                                   100. * batch_idx / len(val_loader_food),
                                   loss.item() / len(data), datetime.datetime.now()))

                        _, preds = torch.max(scores_copy, dim=2)
                        preds = preds.cpu().numpy()
                        targets_copy = targets_copy.cpu().numpy()
                        for i in range(4):
                            sample = preds[i, ...]
                            target = targets_copy[i, ...]
                            print("ORIG: {}\nNEW : {}".format(
                                "".join([charset[chars] for chars in target]),
                                "".join([charset[chars] for chars in sample])
                            ))

                    if which_image == 0:
                        experiment.log_metric("orig_loss", loss.item())
                    else:
                        experiment.log_metric("vae_loss", loss.item())

            experiment.log_metric("loss", losses.avg)
    return losses.avg


for epoch in range(starting_epoch, epochs):
    decoder_sched.step()
    encoder_sched.step()
    train(epoch)
    val = test(epoch)

    torch.save({
        'epoch': epoch,
        'decoder_state_dict': decoder.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'encoder_state_dict': encoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict()}, 'state_' + str(epoch) + ".pt")
