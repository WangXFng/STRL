import math

import numpy as np
import torch.nn as nn
import os

import transformer.Constants as Constants
import torch
# from reformer_pytorch.reformer_pytorch import Reformer
from transformer.Layers import EncoderLayer
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import torch.nn.functional as F

from torch.autograd import Variable


if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, n_dis):
        super().__init__()

        self.d_model = d_model
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=Constants.PAD)  # dding 0

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, n_dis, dropout=dropout)  # 512 1024 4 512 512 M
            for _ in range(n_layers)])

    def forward(self, event_type, non_pad_mask, inner_dis, user_type, predicting_poi):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        # slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
        # slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
        #
        # slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        # # change type of slf_attn_mask_keypad as the same as slf_attn_mask_subseq
        #
        # slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        query_type = self.event_emb(event_type)  # (K M)  event_emb: Embedding (23 512)

        for enc_layer in self.layer_stack:
            query_type, _ = enc_layer(
                query_type,
                inner_dis,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=None)

        enc_output = query_type
        return enc_output, []  # [64, 62, 1024]


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, batch_size, device):
        super().__init__()

        self.linear1 = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.linear2.weight)

        self.linear3 = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.linear3.weight)

        self.linear4 = nn.Linear(dim, num_types, bias=False)  # in_features: int dim, out_features: int num_types
        nn.init.xavier_normal_(self.linear4.weight)

        self.batch_size = batch_size
        self.num_types = num_types
        self.device = device
        self.dim = dim
        # self.poi_avg_aspect = torch.transpose(torch.tensor(poi_avg_aspect, device=device, dtype=torch.float), dim0=0, dim1=1)

        self.a = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.a.data.fill_(1)

        self.b = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.b.data.fill_(1)

        self.c = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.c.data.fill_(1)

        self.d = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.d.data.fill_(1)

    def forward(self, enc_output, event_type, user_type):
        data = enc_output.sum(1)/enc_output.size()[1]
        # user_enc = user_type.sum(1)/user_type.size()[1]

        predicting1 = self.linear1(data)  # [16, 105, 512] -> [16, 105, 1]l
        predicting1 = F.normalize(predicting1, p=2, dim=-1, eps=1e-05)

        predicting2 = self.linear2(data)  # [16, 105, 512] -> [16, 105, 1]l
        predicting2 = F.normalize(predicting2, p=2, dim=-1, eps=1e-05)

        predicting3 = self.linear3(data)  # [16, 105, 512] -> [16, 105, 1]l
        predicting3 = F.normalize(predicting3, p=2, dim=-1, eps=1e-05)

        predicting4 = self.linear4(data)  # [16, 105, 512] -> [16, 105, 1]l
        predicting4 = F.normalize(predicting4, p=2, dim=-1, eps=1e-05)

        out = predicting1 * self.a + predicting2 * self.b + predicting3 * self.c + predicting4 * self.d

        out = torch.tanh(out)  # + self.e * torch.tanh(z)

        target_ = torch.ones(event_type.size()[0], self.num_types, device=self.device, dtype=torch.double)
        for i,e in enumerate(event_type):
            e = e[e!=0] - 1
            target_[i][e] = 0

        return out, target_




class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)  # input_size: d_model, gate_size: 4 * d_rnn
        self.projection = nn.Linear(d_rnn, d_model)  # in_features: int d_rnn, out_features: int d_model

    def forward(self, data, non_pad_mask):

        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)

        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024, n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            batch_size=32, device=0, ita=0.05, n_dis=4):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, n_dis=n_dis)
        # position embedding
        # event type embedding
        self.ita = ita
        self.num_types = num_types

        # convert hidden vectors into a scalar
        # self.linear = nn.Linear(d_model+0, num_types)  # in_features: int d_model, out_features: int num_types

        # parameter for the weight of time difference
        self.alpha = -0.1
        self.a = torch.nn.Parameter(torch.DoubleTensor(1), requires_grad=True)
        self.a.data.fill_(0.1)

        self.predictor = Predictor(d_model, num_types, batch_size, device)

    # event_type, inner_dis, user_type, True
    def forward(self, event_type, inner_dis, user_type, mask):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if inner_dis is not None:
            inner_dis = self.a * inner_dis

        non_pad_mask = get_non_pad_mask(event_type)  # event_type 1

        enc_output, user_output = self.encoder(event_type, non_pad_mask, inner_dis, user_type, mask)  # H(j,:)

        type_prediction, target_ = self.predictor(enc_output, event_type, user_output)  # [16, 105, 22]

        # enc_output = self.rnn(enc_output, non_pad_mask)  # [16, 166, 512]

        return type_prediction, target_
