import torch
from torch.autograd import Variable
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet34(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune



# def conv5x5(in_planes, out_planes, stride=1):
#     """5x5 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
#                      padding=2, bias=False)
#
#
# class BasicBlock(nn.Module):
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv5x5(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv5x5(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.stride = stride
#         if self.stride > 1:
#             self.maxpooling = nn.MaxPool2d(kernel_size=self.stride, stride=1, padding=0, dilation=1, ceil_mode=False)
#
#     def forward(self, x):
#         print("INPUT SHAPE: ", x.shape)
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.stride > 1:
#             out = self.maxpooling(out)
#             identity   = self.maxpooling(identity)
#         print("IDEN SHAPE: ", identity.shape)
#         print("out SHAPE: ", out.shape)
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#     def get_output_size(self, input_dim):
#         if self.stride == 1:
#             return input_dim
#         else:
#             return (input_dim + 2 * 0 - 1 * (2 - 1) - 1) / 1 + 1
#
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         params = {
#             'block1_layers' : 2,
#             'block2_layers' : 4,
#             'block4_layers' : 3
#         }
#
#         self.params = params
#
#         self.block1 = self.__make__blocks(params['block1_layers'], pooling=1, inplanes=3, outplanes=3)
#         self.block2 = self.__make__blocks(params['block2_layers'], pooling=2, inplanes=3, outplanes=3)
#         self.block3 = self.__make__blocks(1, pooling=1, inplanes=3, outplanes=3)
#         self.block4 = self.__make__blocks(params['block4_layers'], pooling=2, inplanes=3, outplanes=3)
#         self.statefc = nn.Sequential(nn.Linear(250, 250), nn.ReLU())
#
#         self.gridLSTM = None
#         self.attention = None
#
#
#     def __make__blocks(self, number_blocks, pooling=1, inplanes=3, outplanes=3):
#         layers = []
#         for i in range(number_blocks):
#             layers.append(BasicBlock(inplanes, outplanes, pooling))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.block1(x)
#         out = self.block2(out)
#         atten_out = self.block3(out)
#         out = self.block4(atten_out)
#         out = out.permute(0, 2, 3, 1)
#
#
#         return out

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class GridLSTMCell2d(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(GridLSTMCell2d, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        print(embed_dim + encoder_dim)
        print(decoder_dim)
        self.lstm1 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.lstm2 = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)


    def forward(self, x, hs, ms): # x : (batch, embedded), H : (batch, 2, h), M : (batch, 2, m)
        H = hs[0] +hs[1]
        m1 = ms[0]
        m2 = ms[1]

        h1, m1 = self.lstm1(x, (H, m1))
        h2, m2 = self.lstm2(x, (H, m2))

        return h1, h2, m1, m2


class StackGridLSTMCell2d(nn.Module):
    """
    Decoder.
    """

    def __init__(self,  attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(StackGridLSTMCell2d, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.lstm1 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
        self.lstm2 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)
        self.lstm3 = GridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)



    def forward(self, x, hs, ms): # x : (batch, embedded), H : (batch, 2, h), M : (batch, 2, m)
        hsA1, hsB1, hsB2, hsB3 = hs
        msA1, msB1, msB2, msB3 = ms

        hsA2, hsB1prime, msA2, msB1prime = self.lstm1(x, [hsA1, hsB1], [msA1, msB1])
        hsA3, hsB2prime, msA3, msB2prime = self.lstm2(x,  [hsA2, hsB2], [msA2, msB2])
        hsA4, hsB3prime, msA4, msB3prime = self.lstm3(x, [hsA3, hsB3], [msA3, msB3])

        #need
        return hsA4, msA4, [hsB1prime, hsB2prime, hsB3prime], [msB1prime, msB2prime, msB3prime]




class GridLSTMDecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(GridLSTMDecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step1 = StackGridLSTMCell2d(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths, teacher_forcing=True, use_first_state=False):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        #print("embedding shape ", embeddings.shape)

        # Initialize LSTM state
        h, m = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        newhs = [h, h, h]
        newms = [m, m, m]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])


            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            lstm_input = None
            #print("embedding size", embeddings.shape, attention_weighted_encoding.shape)

            if not teacher_forcing and t > 0:
                lstm_input = torch.cat([self.embedding(torch.max(predictions[:batch_size_t, t, :], dim=1)[1].long()), attention_weighted_encoding], dim=1)
            else:
                lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)

            if use_first_state and t == 0:
                lstm_input = torch.cat([embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t]], dim=1)

            newhs.insert(0, h)
            newms.insert(0, m)
            newhs = map(lambda x : x[:batch_size_t, ...], newhs)
            newms = map(lambda x : x[:batch_size_t, ...], newms)

            #print("input shape", lstm_input.shape)
            #print('running grid ', t)
            h, m, newhs, newms = self.decode_step1(lstm_input, newhs, newms)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind