# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from functools import partial
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(Backbone):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=None, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,out_features=None, freeze_at=0,last_stride=True,ckpt_at=4
                 ):
        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        current_stride = 4
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        for i in range(3):
            if i==2 and not last_stride:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=1,padding=1,dilation=2),
                )
            else:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            self.downsample_layers.append(downsample_layer)
        self.stage_names=[]
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            name_u="stage"+str(i+1)+"_unorm"
            self.stage_names.append(name_u)
            self._out_feature_strides[name_u] = current_stride *2**i
            self._out_feature_channels[name_u] = dims[i]
            name="stage"+str(i+1)
            self.stage_names.append(name)
            self._out_feature_strides[name] = self._out_feature_strides[name_u]
            self._out_feature_channels[name] = self._out_feature_channels[name_u]
        self.stage_names = tuple(self.stage_names)
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        
        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        self.freeze(freeze_at)
        self.ckpt_at=ckpt_at

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        # TODO consider output norm
        for idx, (down_samp,stage) in enumerate(zip(self.downsample_layers,self.stages)):
            if freeze_at > idx:
                for p in down_samp.parameters():
                    p.requires_grad = False
                for p in stage.parameters():
                    p.requires_grad = False
        return self

    def forward_features(self, x):
        outputs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i>=self.ckpt_at-1:
                x=checkpoint(self.stages[i],x)
            else:
                x = self.stages[i](x)
            name_un="stage"+str(i+1)+"_unorm"
            if name_un in self._out_features:
                outputs[name_un]=x
                if self._out_features.index(name_un)==len(self._out_features)-1:
                    break
            name="stage"+str(i+1)
            if name in self._out_features:
                norm_layer=getattr(self,f'norm{i}')
                x_out=norm_layer(x)
                outputs[name]=x_out
                if self._out_features.index(name)==len(self._out_features)-1:
                    break
        return outputs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@BACKBONE_REGISTRY.register()
def convnext_tiny(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_tiny_22k'] if model_cfg.IN_22K else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model

@BACKBONE_REGISTRY.register()
def convnext_tiny_lite(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 2, 2, 2], dims=[96, 192, 384, 768], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_tiny_22k'] if model_cfg.IN_22K else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model

@BACKBONE_REGISTRY.register()
def convnext_tiny_minor(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 1, 1, 1], dims=[96, 192, 384, 768], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_tiny_22k'] if model_cfg.IN_22K else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model

@BACKBONE_REGISTRY.register()
def convnext_small(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_small_22k'] if model_cfg.IN_22K else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONE_REGISTRY.register()
def convnext_base(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_base_22k'] if model_cfg.IN_22K else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model

@BACKBONE_REGISTRY.register()
def convnext_large(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        url = model_urls['convnext_large_22k'] if model_cfg.IN_22K else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model

@BACKBONE_REGISTRY.register()
def convnext_xlarge(cfg,input_shape):
    model_cfg=cfg.MODEL.CONV_NEXT
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], out_features=model_cfg.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT,last_stride=model_cfg.LAST_STRIDE,ckpt_at=model_cfg.CHECKPOINT_AT)
    if model_cfg.PRETRAINED:
        assert model_cfg.IN_22K, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        checkpoint=checkpoint["model"]
        pnorm_weight=checkpoint.pop("norm.weight")
        pnorm_bias=checkpoint.pop("norm.bias")
        checkpoint["norm3.weight"]=pnorm_weight
        checkpoint["norm3.bias"]=pnorm_bias
        rst=model.load_state_dict(checkpoint,strict=False)
        print(rst)
    return model