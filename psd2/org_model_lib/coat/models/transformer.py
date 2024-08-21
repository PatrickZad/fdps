# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import math
import random
from functools import reduce
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils.mask import exchange_token, exchange_patch, get_mask_box, jigsaw_token, cutout_patch, erase_patch, mixup_patch, jigsaw_patch


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv1x1Linear(nn.Module):
    """
    to reduce memory usage
    """

    def __init__(self, in_planes, out_planes, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_planes, out_planes, bias)

    def forward(self, x):
        # x : b x c x h x w
        x_in = x.permute(0, 2, 3, 1)
        x_out = self.linear(x_in)
        return x_out.permute(0, 3, 1, 2)


class TransformerHeadMaxp(nn.Module):
    def __init__(
        self,
        cfg,
        kernel_size,
    ):
        super(TransformerHeadMaxp, self).__init__()
        d_model = cfg.DIM_MODEL

        self.transformer_encoder = Transformers(
            cfg=cfg,
            kernel_size=kernel_size,
        )
        self.conv1 = conv1x1(1024, d_model)  # Conv1x1Linear(1024, d_model)
        self.conv2 = conv1x1(d_model, 2048)  # Conv1x1Linear(d_model, 2048)

    def forward(self, box_features):

        trans_features = {}
        trans_features["before_trans"] = F.adaptive_max_pool2d(box_features, 1)
        box_features = self.conv1(box_features)
        box_features = self.transformer_encoder(box_features)
        box_features = self.conv2(box_features)
        trans_features["after_trans"] = F.adaptive_max_pool2d(box_features, 1)

        return trans_features


class TransformerHeadAvgp(nn.Module):
    def __init__(
        self,
        cfg,
        kernel_size,
    ):
        super(TransformerHeadAvgp, self).__init__()
        d_model = cfg.DIM_MODEL

        self.transformer_encoder = Transformers(
            cfg=cfg,
            kernel_size=kernel_size,
        )
        self.conv1 = conv1x1(1024, d_model)  # Conv1x1Linear(1024, d_model)
        self.conv2 = conv1x1(d_model, 2048)  # Conv1x1Linear(d_model, 2048)

    def forward(self, box_features):

        trans_features = {}
        trans_features["before_trans"] = F.adaptive_max_pool2d(box_features, 1)
        box_features = self.conv1(box_features)
        box_features = self.transformer_encoder(box_features)
        box_features = self.conv2(box_features)
        trans_features["after_trans"] = F.adaptive_max_pool2d(box_features, 1)

        return trans_features


class Transformers(nn.Module):
    def __init__(
        self,
        cfg,
        kernel_size,
    ):
        super(Transformers, self).__init__()
        d_model = cfg.DIM_MODEL
        self.trans_names = ["reid_trans"]
        self.scale_size = 1
        hidden = d_model // (
            1 * self.scale_size
        )  # NOTE try 1 scale,(2 * self.scale_size)

        # kernel_size: (padding, stride)
        kernels = {(1, 1): [(0, 0), (1, 1)], (3, 3): [(1, 1), (1, 1)]}
        # kernels = {(1, 1): [(0, 0), (1, 1)], (3, 3): [(2, 2), (1, 1)]}
        padding = []
        stride = []
        for ksize in kernel_size:
            ksize = (ksize[0], ksize[1])
            if ksize not in [(1, 1), (3, 3)]:
                raise ValueError("Undefined kernel size.")
            padding.append(kernels[ksize][0])
            stride.append(kernels[ksize][1])

        self.use_output_layer = cfg.USE_OUTPUT_LAYER
        self.use_global_shortcut = cfg.USE_GLOBAL_SHORTCUT

        self.blocks = nn.ModuleDict()
        for tname, ksize, psize, ssize in zip(
            self.trans_names, kernel_size, padding, stride
        ):
            transblock = Transformer(
                cfg,
                d_model // self.scale_size,
                ksize,
                psize,
                ssize,
                hidden,
            )
            self.blocks[tname] = nn.Sequential(transblock)

        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs):
        trans_feat = []
        enc_feat = inputs

        for tname, feat in zip(
            self.trans_names, torch.chunk(enc_feat, len(self.trans_names), dim=1)
        ):
            feat = self.blocks[tname](feat)
            trans_feat.append(feat)

        trans_feat = torch.cat(trans_feat, 1)
        if self.use_output_layer:
            trans_feat = self.output_linear(trans_feat)
        if self.use_global_shortcut:
            trans_feat = enc_feat + trans_feat
        return trans_feat


class Transformer(nn.Module):
    def __init__(self, cfg, channel, kernel_size, padding, stride, hidden):
        super(Transformer, self).__init__()
        self.k = kernel_size[0]
        stack_num = cfg.N_LAYERS
        num_head = cfg.N_HEAD
        dropout = cfg.DROPOUT
        output_size = (14, 14)
        token_size = tuple(map(lambda x, y: x // y, output_size, stride))
        blocks = []
        self.transblock = TransformerBlock(
            token_size, hidden=hidden, num_head=num_head, dropout=dropout
        )
        for _ in range(stack_num):
            blocks.append(self.transblock)
        self.transformer = nn.Sequential(*blocks)
        self.patch2vec = nn.Conv2d(
            channel, hidden, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.vec2patch = Vec2Patch(
            channel, hidden, output_size, kernel_size, stride, padding
        )
        self.use_local_shortcut = cfg.USE_LOCAL_SHORTCUT

    def forward(self, inputs):
        enc_feat = inputs
        b, c, h, w = enc_feat.size()

        trans_feat = self.patch2vec(enc_feat)

        _, c, h, w = trans_feat.size()
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)

        trans_feat = self.transformer(trans_feat)
        trans_feat = self.vec2patch(trans_feat)
        if self.use_local_shortcut:
            trans_feat = enc_feat + trans_feat

        return trans_feat


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, tokensize, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(
            tokensize, d_model=hidden, head=num_head, p=dropout
        )
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)

        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class Vec2Patch(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)
        feat = self.to_patch(feat)

        return feat


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, tokensize, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        value = self.value_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
