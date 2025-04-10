# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from flash_attn.modules.embedding import GPT2Embeddings
from collections import namedtuple

class ConvNeXtBlock(nn.Module):
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
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        l = x.shape[1]
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)[:, :, :l]
        x = x.permute(0, 2, 1)
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = residual + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
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
    def __init__(self, max_length, vocab_size, pad_vocab_size_multiple=1, d_model=128, 
                 depths=[1, 2, 9, 3], dims=[128, 192, 384, 768], drop_path_rate=0.1, 
                 layer_scale_init_value=1e-6, head_init_scale=1., base=2, k_size=5, max_position_embeddings=0, stride=2,
                 device=None, dtype=None,**kwargs,
                 ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model=d_model
        in_chans=d_model
        self.max_length = max_length
        self.base = base
        self.k_size = k_size
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.embeddings = GPT2Embeddings(
                d_model, vocab_size, max_position_embeddings, **factory_kwargs
        )
        self.n_layer = 6
        depths = range(0, self.n_layer)
        self.stride = stride
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        stem = nn.Sequential(
            nn.Conv1d(in_chans, in_chans, kernel_size=self.k_size, stride=stride, groups=in_chans//8),
            LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(1,self.n_layer):
            downsample_layer = nn.Sequential(
                    LayerNorm(in_chans*(i), eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(in_chans*(i), in_chans*(i+1), kernel_size=self.k_size, stride=stride, groups=in_chans//8),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages_down = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        sum_depths = sum(depths)
        cur = 0
        for i in range(self.n_layer):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=in_chans*(i+1), drop_path=dp_rates[(cur + j)%sum_depths], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i%self.n_layer])]
            )
            self.stages_down.append(stage)
            cur += depths[i%self.n_layer]
        
        self.mlp = nn.Sequential(
            nn.Linear(in_chans*self.n_layer, in_chans*(self.n_layer-1)),
            nn.SiLU()
        )

        self.upsample_layers = nn.ModuleList()
        for i in range(self.n_layer, 1, -1):
            upsample_layer = nn.Sequential(
                    LayerNorm(in_chans*(i-1), eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(in_chans*(i-1), in_chans*(i-2  if i>2 else 1), kernel_size=self.k_size, padding=self.k_size-1, groups=in_chans//8),
            )
            self.upsample_layers.append(upsample_layer)
        stem = nn.Sequential(
            nn.Conv1d(in_chans, in_chans, kernel_size=self.k_size, padding=self.k_size-1),
            LayerNorm(in_chans, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem)

        self.stages_up = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        sum_depths = sum(depths)
        cur = 0
        for i in range(self.n_layer, 0, -1):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=in_chans*(i-2 if i>2 else 1), drop_path=dp_rates[(cur + j)%sum_depths], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i%self.n_layer])]
            )
            self.stages_up.append(stage)
            cur += depths[i%self.n_layer]

        self.norm = nn.LayerNorm(in_chans, eps=1e-6) # final norm layer
        # self.head = nn.Linear(in_chans, vocab_size)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        skip_list = []
        if 316>x.shape[1]+1:
            pad_l = 316-x.shape[1]
            pad_x = torch.zeros(x.shape[0], pad_l, x.shape[2], device=x.device)
            x = torch.concat([x, pad_x], dim=1)
        for i in range(self.n_layer):
            skip_list.append(x)
            x = x.permute(0, 2, 1)
            x = self.downsample_layers[i](x)
            x = x.permute(0, 2, 1)
            x = self.stages_down[i](x) # B, (L-k)/s+1, D
        for i in range(self.n_layer):
            # x = F.pad(x.permute(0, 2, 1), (1, -1))
            B = x.shape[0]
            D = x.shape[-1]
            skip = skip_list.pop() # B, L, D_i
            if (skip.shape[1]-self.k_size) % self.stride == 0:
                x = x[:, :-1, :]
            x = x[:, :, None, :].repeat(1, 1, self.stride, 1).reshape(B, -1, D)
            pad = torch.zeros(x.shape[0], skip.shape[1]-x.shape[1], x.shape[-1], dtype=x.dtype, device=x.device)
            if (skip.shape[1]-self.k_size) % self.stride == 0:
                pad[:, -1, :] = x[:, 0, :]
            x = torch.concat([pad, x], dim=1) # B, L, D_i+1
            if i == 0:
                x = self.mlp(x)
            x = self.upsample_layers[i]((x+skip).permute(0, 2, 1))[:, :, :skip.shape[1]]
            x = x.permute(0, 2, 1)
            x = self.stages_up[i](x)


        # return self.norm(x.mean(-2)) # global average pooling, (N, C, H, W) -> (N, C) (B L D)->(B D)
        return self.norm(x)

    def forward(self, input_ids, position_ids=None, state=None):
        # embedding_kwargs = (
        #     {"combine_batch_seqlen_dim": True}
        #     if self.process_group is not None and self.sequence_parallel
        #     else {}
        # )
        x = self.embeddings(
            input_ids, position_ids=position_ids,
        )
        # x = input_ids
        x = self.forward_features(x)
        # logits = self.head(x)
        # CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        # return CausalLMOutput(logits=logits), None
        return x, None
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model

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
            x = self.weight[:, None] * x + self.bias[:, None]
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

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


class NBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, i=0, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        if i == 0:
            self.dwconv1 = nn.Conv1d(dim, dim, kernel_size=9, padding=4) # depthwise conv
            self.dwconv2 = None
        else:
            self.dwconv1 = nn.Conv1d(dim, dim, kernel_size=9, dilation=4**(i-1), padding=4*(4**(i-1))) # depthwise conv
            # self.dwconv2 = nn.Conv1d(dim, dim, kernel_size=9, dilation=4**i, padding=4*(4**i)) # depthwise conv
            self.dwconv2 = None
        self.norm1 = nn.LayerNorm(dim)
        # self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        if self.dwconv2 is not None:
            self.act2 = nn.GELU()
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
            self.norm2 = LayerNorm(dim, eps=1e-6)
        # self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        x = self.dwconv1(x) # BDL 
        # x = self.pwconv1(x)
        x = self.act1(x)
        # x = self.pwconv2(x)
        # if self.gamma1 is not None:
        #     x = self.gamma1 * x
        # x = x.permute(0, 2, 1) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        if self.dwconv2 is not None:
            input = x
            x = self.dwconv2(x) # BDL
            x = x.permute(0, 2, 1) 
            x = self.norm2(x)
            # x = self.pwconv1(x)
            x = self.act2(x)
            # x = self.pwconv2(x)
            if self.gamma2 is not None:
                x = self.gamma2 * x
            x = x.permute(0, 2, 1) # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)
        return x

class NConvNeXthh(nn.Module):
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
    def __init__(self, d_model=128, in_chans=5, num_classes=1000, 
                 depths=[1, 1, 3, 5], dims=[72, 96, 128, 256], drop_path_rate=0.,  # 512, 812, 1024, 1460
                 layer_scale_init_value=1e-6, head_init_scale=1., alphabet_size = 5,
                 ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2),
            )
            self.downsample_layers.append(downsample_layer)
        # stem = nn.Sequential(
        #     nn.Conv1d(dims[-1], dims[0], kernel_size=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        # self.downsample_layers.append(stem)
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #             nn.Conv1d(dims[i], dims[i+1], kernel_size=2),
        #     )
        #     self.downsample_layers.append(downsample_layer)

        self.down_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            down_stage = nn.Sequential(
                *[NBlock(dim=dims[i], i=i, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.down_stages.append(down_stage)
            cur += depths[i]
        # cur = 0
        # for i in range(4):
        #     down_stage = nn.Sequential(
        #         *[NBlock(dim=dims[-1], i=i, drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.down_stages.append(down_stage)
        #     cur += depths[i]
        
        self.upsample_layers = nn.ModuleList()
        # for i in range(3,0,-1):
        #     upsample_layer = nn.Sequential(
        #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #             nn.Conv1d(dims[i], dims[i-1], kernel_size=2, stride=1),
        #     )
        #     self.upsample_layers.append(upsample_layer)
        stem = nn.Sequential(
            nn.Conv1d(dims[-1], d_model, kernel_size=4, stride=1),
            LayerNorm(d_model, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem)

        # self.up_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur -= depths[3]
        # for i in range(2,-1,-1):
        #     cur -= depths[i]
        #     up_stage = nn.Sequential(
        #         *[NBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.up_stages.append(up_stage)

        self.norm = nn.LayerNorm(d_model, eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            if i < 4:
                x = self.downsample_layers[i](x.permute(0, 2, 1))
                x = self.down_stages[i](x).permute(0, 2, 1)
            #     if i == 3:
            #         x = x.permute(0,2,1)
            # else:
            #     x = self.down_stages[i](x)
        # for i in range(3):
        #     x = self.upsample_layers[i](x.permute(0, 2, 1))
        #     x = self.up_stages[i](x).permute(0, 2, 1)
        x = self.upsample_layers[0](x.permute(0,2,1)).permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        return self.norm(torch.mean(x, dim=1, keepdim=True))

    def forward(self, x, state=None):
        x = torch.nn.functional.one_hot(x, num_classes=self.alphabet_size).type(torch.float32)
        x = self.forward_features(x)
        # x = self.head(x)
        return x, None
    
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model

class NConvNeXth(nn.Module):
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
    def __init__(self, d_model=128, in_chans=5, num_classes=1000, 
                 depths=[1, 1, 1, 1, 1], dims=[128, 128, 128, 128, 128], drop_path_rate=0.,  # 512, 812, 1024, 1460
                 layer_scale_init_value=1e-6, head_init_scale=1., alphabet_size = 5,
                 ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.dims= dims

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=5, padding=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=3, padding=1),
            )
            self.downsample_layers.append(downsample_layer)
        # stem = nn.Sequential(
        #     nn.Conv1d(dims[-1], dims[0], kernel_size=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        # self.downsample_layers.append(stem)
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #             nn.Conv1d(dims[i], dims[i+1], kernel_size=2),
        #     )
        #     self.downsample_layers.append(downsample_layer)

        self.down_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(5):
            down_stage = nn.Sequential(
                *[NBlock(dim=dims[i], i=i, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.down_stages.append(down_stage)
            cur += depths[i]
        # cur = 0
        # for i in range(4):
        #     down_stage = nn.Sequential(
        #         *[NBlock(dim=dims[-1], i=i, drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.down_stages.append(down_stage)
        #     cur += depths[i]
        
        self.upsample_layers = nn.ModuleList()
        # for i in range(3,0,-1):
        #     upsample_layer = nn.Sequential(
        #             LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #             nn.Conv1d(dims[i], dims[i-1], kernel_size=2, stride=1),
        #     )
        #     self.upsample_layers.append(upsample_layer)
        stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
            LayerNorm(d_model, eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(stem)

        # self.up_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        # dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # cur -= depths[3]
        # for i in range(2,-1,-1):
        #     cur -= depths[i]
        #     up_stage = nn.Sequential(
        #         *[NBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
        #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
        #     )
        #     self.up_stages.append(up_stage)

        self.norm = nn.LayerNorm(d_model, eps=1e-6) # final norm layer
        # self.milinear = nn.Sequential(nn.Linear(self.dims[-1], self.dims[-1]), nn.ReLU(), nn.Linear(self.dims[-1], self.d_model), nn.LayerNorm(self.d_model))

        # self.linear1 = nn.Sequential(nn.Linear(self.d_model+self.dims[0], self.dims[0]), nn.SiLU(), nn.Linear(self.dims[0], self.d_model), nn.LayerNorm(self.d_model))
        # self.linear2 = nn.Sequential(nn.Linear(self.d_model+self.dims[1], self.dims[1]), nn.SiLU(), nn.Linear(self.dims[1], self.d_model), nn.LayerNorm(self.d_model))
        # self.linear3 = nn.Sequential(nn.Linear(self.d_model+self.dims[2], self.dims[2]), nn.SiLU(), nn.Linear(self.dims[2], self.d_model), nn.LayerNorm(self.d_model))
        # self.linear4 = nn.Sequential(nn.Linear(self.d_model+self.dims[3], self.dims[3]), nn.SiLU(), nn.Linear(self.dims[3], self.d_model), nn.LayerNorm(self.d_model))
        # self.linear5 = nn.Sequential(nn.Linear(self.d_model+self.dims[4], self.dims[4]), nn.SiLU(), nn.Linear(self.dims[4], self.d_model), nn.LayerNorm(self.d_model))

        self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv1d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)
    def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        glu_act=False,
    ):
        torch.manual_seed(2222)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight", 'mha.in_proj_weight', 'Wqkv.weight']:
                    nn.init.kaiming_normal_(p)
                elif name in ["output_linear.0.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    if not glu_act:
                        nn.init.kaiming_normal_(p)
                    else:
                        out_features = p.shape[0]
                        nn.init.kaiming_normal_(p[: out_features//2])

    def forward_features(self, x):
        # x1 = self.downsample_layers[0](x.permute(0, 2, 1))
        x1 = x.permute(0,2,1)
        # x1 = x
        x1 = self.down_stages[0](x1).permute(0, 2, 1)

        # x2 = self.downsample_layers[1](x1.permute(0, 2, 1))
        x2 = x1.permute(0,2,1)
        x2 = self.down_stages[1](x2).permute(0, 2, 1)

        # x3 = self.downsample_layers[2](x2.permute(0, 2, 1))
        x3 = x2.permute(0,2,1)
        x3 = self.down_stages[2](x3).permute(0, 2, 1)

        # x4 = self.downsample_layers[3](x3.permute(0, 2, 1))
        x4 = x3.permute(0,2,1)
        x4 = self.down_stages[3](x4).permute(0, 2, 1)

        # x5 = self.downsample_layers[4](x4.permute(0, 2, 1))
        x5 = x4.permute(0,2,1)
        x5 = self.down_stages[4](x5).permute(0,2,1)

        # x = self.milinear(x5)

        # out = torch.concatenate([x,x1], dim=-1)
        # x = self.linear1(out)+x
        # out = torch.concatenate([x,x2], dim=-1)
        # x = self.linear2(out)+x
        # out = torch.concatenate([x,x3], dim=-1)
        # x = self.linear3(out)+x
        # out = torch.concatenate([x,x4], dim=-1)
        # x = self.linear4(out)+x
        # out = torch.concatenate([x,x5], dim=-1)
        # x = self.linear5(out)+x

        # x = self.upsample_layers[0](x.permute(0,2,1)).permute(0, 2, 1)
        return x5

        # x = x.permute(0, 2, 1)
        return self.norm(torch.mean(x, dim=1, keepdim=True))

    def forward(self, x, state=None):
        # x = torch.nn.functional.one_hot(x, num_classes=self.alphabet_size).type(torch.float32)
        x = self.forward_features(x)
        # x = self.head(x)
        return x
    
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model

class NConvNeXthi(nn.Module):
    def __init__(self, d_model=128, in_chans=5, num_classes=1000, 
                 depths=[1, 1, 2, 2, 3], dims=[72, 96, 128, 196, 256], drop_path_rate=0.,  # 512, 812, 1024, 1460
                 layer_scale_init_value=1e-6, head_init_scale=1., alphabet_size = 5,
                 ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.dims= dims
        self.num_layers = 2
        self.upconv = nn.Conv1d(self.alphabet_size, d_model, kernel_size=9, padding=4)
        self.layers = nn.ModuleList([NConvNeXth(d_model=d_model, in_chans=d_model) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(self.d_model, self.d_model, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv1d(self.d_model, self.d_model, kernel_size=1))
        # import copy
        # self.convs = [nn.Conv1d(self.d_model, self.d_model, kernel_size=9, padding=4),
        #                              nn.Conv1d(self.d_model, self.d_model, kernel_size=9, padding=4),
        #                              nn.Conv1d(self.d_model, self.d_model, kernel_size=9, dilation=4, padding=16),
        #                              nn.Conv1d(self.d_model, self.d_model, kernel_size=9, dilation=16, padding=64),
        #                              nn.Conv1d(self.d_model, self.d_model, kernel_size=9, dilation=64, padding=256)]
        # self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(2)])
        self.dropout = nn.Dropout(0)

    def forward(self, x, state=None):
        x = torch.nn.functional.one_hot(x, num_classes=self.alphabet_size).type(torch.float32)
        x = F.relu(self.upconv(x.permute(0,2,1))).permute(0,2,1)
        for i in range(self.num_layers):
            # h = self.norms[i](x)
            h = x
            h = self.layers[i].forward_features(h)
            x = h
        # x = x.permute(0,2,1)
        # for i in range(10):
        #     h = self.dropout(x.clone())
        #     # if not self.args.clean_data:
        #     #     h = h + self.time_layers[i](time_emb)[:, :, None]
        #     # if self.args.cls_free_guidance and not self.classifier:
        #     #     h = h + self.cls_layers[i](cls_emb)[:, :, None]
        #     h = self.norms[i]((h).permute(0, 2, 1))
        #     h = F.relu(self.convs[i](h.permute(0, 2, 1)))
        #     if h.shape == x.shape:
        #         x = h + x
        #     else:
        #         x = h
        # x = self.head(x)
        # x = x.permute(0,2,1)

        x = self.final_conv(x.permute(0,2,1))
        x = x.permute(0, 2, 1)
        return x, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model

"""
@author: Yuanhao Cai
@date:  2020.03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False, dilation=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
            efficient=False, i=0):
        super(Bottleneck, self).__init__()
        self.conv_bn_relu1 = conv_bn_relu(in_planes, planes, kernel_size=1,
                stride=stride, padding=0, has_bn=True, has_relu=True,
                efficient=efficient) 
        # if i == 0:
        #     self.conv_bn_relu2 = conv_bn_relu(planes, planes,
        #             stride=1, has_bn=True, has_relu=True,  kernel_size=9, dilation=2**i, padding=4*(2**i),
        #             efficient=efficient) 
        # else:
        #     self.conv_bn_relu2 = conv_bn_relu(planes, planes,
        #             stride=1, has_bn=True, has_relu=True,  kernel_size=9, dilation=2**i, padding=4*(2**i),
        #             efficient=efficient) 
        self.conv_bn_relu2 = conv_bn_relu(planes, planes,
                    stride=1, has_bn=True, has_relu=True,  kernel_size=9, padding=4,
                    efficient=efficient) 
        self.conv_bn_relu3 = conv_bn_relu(planes, planes * self.expansion,
                kernel_size=1, stride=1, padding=0, has_bn=True,
                has_relu=False, efficient=efficient) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)

        return out


class ResNet_top(nn.Module):

    def __init__(self):
        super(ResNet_top, self).__init__()
        self.conv = conv_bn_relu(5, 64, kernel_size=7, stride=1, padding=3,
                has_bn=True, has_relu=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.maxpool(x)

        return x


class ResNet_downsample_module(nn.Module):

    def __init__(self, block, layers, has_skip=False, efficient=False,
            zero_init_residual=False):
        super(ResNet_downsample_module, self).__init__()
        self.has_skip = has_skip 
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0],
                efficient=efficient)
        self.layer2 = self._make_layer(block, 72, layers[1], stride=2,
                efficient=efficient, i=0)
        self.layer3 = self._make_layer(block, 96, layers[2], stride=2,
                efficient=efficient, i=1)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                efficient=efficient, i=2)
        self.layer5 = self._make_layer(block, 164, layers[4], stride=2,
                efficient=efficient, i=3)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, efficient=False, i=0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, padding=0, has_bn=True,
                    has_relu=False, efficient=efficient)

        layers = list() 
        layers.append(block(self.in_planes, planes, stride, downsample,
            efficient=efficient, i=i))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, efficient=efficient, i=i))

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]
        x5 = self.layer5(x4)
        if self.has_skip:
            x5 = x5 + skip1[4] + skip2[4]

        return x5, x4, x3, x2, x1


class Upsample_unit(nn.Module): 

    def __init__(self, ind, in_planes, up_size, output_chl_num, output_shape,
            chl_num=128, gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_unit, self).__init__()
        self.output_shape = output_shape

        self.u_skip = conv_bn_relu(in_planes, chl_num, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
        self.relu = nn.ReLU(inplace=True)

        self.ind = ind
        if self.ind > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=False,
                    efficient=efficient)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
            self.skip2 = conv_bn_relu(chl_num, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.gen_cross_conv = gen_cross_conv
        if self.ind == 4 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(chl_num, 64, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)

        self.res_conv1 = conv_bn_relu(chl_num, chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.res_conv2 = conv_bn_relu(chl_num, output_chl_num, kernel_size=3,
                stride=1, padding=1, has_bn=True, has_relu=False,
                efficient=efficient)

    def forward(self, x, up_x):
        out = self.u_skip(x)

        if self.ind > 0:
            self.up_size = x.shape[-1]
            up_x = F.interpolate(up_x, size=self.up_size, mode='linear',
                    align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x 
        out = self.relu(out)

        res = self.res_conv1(out)
        res = self.res_conv2(res)
        res = F.interpolate(res, size=self.output_shape, mode='linear',
                align_corners=True)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.ind == 4 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, res, skip1, skip2, cross_conv


class Upsample_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, chl_num=128,
            gen_skip=False, gen_cross_conv=False, efficient=False):
        super(Upsample_module, self).__init__()
        self.in_planes = [164, 128, 96, 72, 64] 
        # h, w = output_shape
        # self.up_sizes = [
        #         (h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.up_sizes = [
            output_shape//16, output_shape//8, output_shape//4, output_shape//2, output_shape
        ]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up1 = Upsample_unit(0, self.in_planes[0], self.up_sizes[0],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up2 = Upsample_unit(1, self.in_planes[1], self.up_sizes[1],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up3 = Upsample_unit(2, self.in_planes[2], self.up_sizes[2],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up4 = Upsample_unit(3, self.in_planes[3], self.up_sizes[3],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)
        self.up5 = Upsample_unit(4, self.in_planes[4], self.up_sizes[4],
                output_chl_num=output_chl_num, output_shape=output_shape,
                chl_num=chl_num, gen_skip=self.gen_skip,
                gen_cross_conv=self.gen_cross_conv, efficient=efficient)

    def forward(self, x5, x4, x3, x2, x1):
        out1, res1, skip1_1, skip2_1, _ = self.up1(x5, None)
        out2, res2, skip1_2, skip2_2, _ = self.up2(x4, out1)
        out3, res3, skip1_3, skip2_3, _ = self.up3(x3, out2)
        out4, res4, skip1_4, skip2_4, _ = self.up4(x2, out3)
        out5, res5, skip1_5, skip2_5, cross_conv = self.up5(x1, out4)

        # 'res' starts from small size
        res = [res1, res2, res3, res4, res5]
        skip1 = [skip1_5, skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_5, skip2_4, skip2_3, skip2_2, skip2_1]

        return res, skip1, skip2, cross_conv


class Single_stage_module(nn.Module):

    def __init__(self, output_chl_num, output_shape, has_skip=False,
            gen_skip=False, gen_cross_conv=False, chl_num=128, efficient=False,
            zero_init_residual=False,**kwargs):
        super(Single_stage_module, self).__init__()
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.chl_num = chl_num
        self.zero_init_residual = zero_init_residual 
        self.layers = [1, 1, 1, 1, 1]
        self.downsample = ResNet_downsample_module(Bottleneck, self.layers,
                self.has_skip, efficient, self.zero_init_residual)
        self.upsample = Upsample_module(output_chl_num, output_shape,
                self.chl_num, self.gen_skip, self.gen_cross_conv, efficient)

    def forward(self, x, skip1, skip2):
        x5, x4, x3, x2, x1 = self.downsample(x, skip1, skip2)
        res, skip1, skip2, cross_conv = self.upsample(x5, x4, x3, x2, x1)
        
        return res, skip1, skip2, cross_conv


class NConvNeXtp(nn.Module):
    
    def __init__(self, d_model, stage_num=4, output_chl_num=128,output_shape=500, upsample_chl_num=128, run_efficient=False, **kwargs):
        super(NConvNeXt, self).__init__()
        self.d_model = d_model
        self.top = ResNet_top()
        self.stage_num = stage_num
        self.output_chl_num = output_chl_num
        self.output_shape = output_shape
        self.upsample_chl_num = upsample_chl_num
        # self.ohkm = cfg.LOSS.OHKM
        # self.topk = cfg.LOSS.TOPK
        # self.ctf = cfg.LOSS.COARSE_TO_FINE
        self.mspn_modules = list() 
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False
            else:
                has_skip = True
            if i != self.stage_num - 1:
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False 
                gen_cross_conv = False 
            self.mspn_modules.append(
                    Single_stage_module(
                        self.output_chl_num, self.output_shape,
                        has_skip=has_skip, gen_skip=gen_skip,
                        gen_cross_conv=gen_cross_conv,
                        chl_num=self.upsample_chl_num,
                        efficient=run_efficient,
                        **kwargs
                        )
                    )
            setattr(self, 'stage%d' % i, self.mspn_modules[i])
        
    def forward(self, seq, valids=None, labels=None, state=None):
        seq = torch.nn.functional.one_hot(seq, num_classes=5).type(torch.float32)
        seq = seq.permute(0,2,1)
        x = self.top(seq)
        skip1 = None
        skip2 = None
        outputs = list()
        for i in range(self.stage_num):
            res, skip1, skip2, x = eval('self.stage' + str(i))(x, skip1, skip2)
            outputs.append(res)

        return outputs[-1][-1].permute(0,2,1), None
    
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model





import torch
import torch.nn as nn
# from sequnet_utils import Crop1d, Crop1dFrontBack

class Crop1d(nn.Module):
    def __init__(self, mode="both"):
        super(Crop1d, self).__init__()
        self.mode = mode

    def forward(self, x, target):
        if x is None:
            return None
        if target is None:
            return x

        target_shape = target.shape
        diff = x.shape[-1] - target_shape[-1]
        if self.mode == "both":
            assert(diff % 2 == 0)
            crop = diff // 2
        else:
            crop = diff

        if crop == 0:
            return x
        if crop < 0:
            raise ArithmeticError

        if self.mode == "front":
            return x[:, :, crop:].contiguous()
        elif self.mode == "back":
            return x[:, :, :-crop].contiguous()
        else:
            assert(self.mode == "both")
            return x[:, :, crop:-crop].contiguous()

class Crop1dFrontBack(nn.Module):
    def __init__(self, crop_front, crop_back):
        super(Crop1dFrontBack, self).__init__()
        self.crop_front = crop_front
        self.crop_back = crop_back

    def forward(self, x):
        if self.crop_back > 0:
            return x[:, :, self.crop_front:-self.crop_back].contiguous()
        else:
            return x[:, :, self.crop_front:].contiguous()


class ConvolutionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout, causal, activation=nn.LeakyReLU(), transpose=False, dilation=1, padding=0):
        super(ConvolutionBlock, self).__init__()

        ops = list()
        if transpose:
            ops.append(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size, stride=2, dilation=dilation))

            if causal:
                crop_front = kernel_size - 1 - padding  # By default, crop at front and end to get only valid output, but crop less if padding is activated to get zero-padded outputs at start
                crop_back = kernel_size - 1
            else:
                assert (padding % 2 == 0)  # Non-causal: Crop less in front and back, equally
                crop_front = kernel_size - 1 - padding // 2
                crop_back = kernel_size - 1 - padding // 2

            ops.append(Crop1dFrontBack(crop_front, crop_back))

        else: # Normal convolution
            if padding > 0:
                if causal:
                    ops.append(torch.nn.ConstantPad1d((padding, 0), 0.0))
                elif dilation==1:
                    ops.append(torch.nn.ConstantPad1d((padding//2, padding//2), 0.0))
                else:
                    ops.append(torch.nn.ConstantPad1d((padding, padding), 0.0))

            ops.append(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, dilation=dilation))

        if activation is not None:
            ops.append(activation)

        if dropout > 0.0:
            ops.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*ops)

    def forward(self, x):
        return self.block(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_shortcut, kernel_size, stride, padding, causal, dropout, i=0):
        super(UpsamplingBlock, self).__init__()

        # CONV 1 for UPSAMPLING
        self.conv1 = ConvolutionBlock(n_inputs, n_inputs, kernel_size, stride, dropout, causal, transpose=True, padding=padding)

        # Crop operation for the shortcut connection that might have more samples!
        self.crop = Crop1d("front") if causal else Crop1d("both")

        # CONV 2 to combine high- with low-level information (from shortcut)
        self.conv2 = ConvolutionBlock(n_inputs + n_shortcut, n_outputs, kernel_size, 1, dropout, causal, dilation=4**i, padding=4**i*(kernel_size-1)//2)

    def forward(self, x, shortcut):
        upsampled = self.conv1(x)
        shortcut_crop = self.crop(shortcut, upsampled)
        combined = torch.cat([upsampled, shortcut_crop], 1)
        return self.conv2(combined)

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, padding, causal, dropout):
        super(DownsamplingBlock, self).__init__()

        # CONV 1
        self.conv1 = ConvolutionBlock(n_inputs, n_outputs, kernel_size, 1, dropout, causal, padding=padding)

        # CONV 2 with decimation
        self.conv2 = ConvolutionBlock(n_outputs, n_outputs, kernel_size, stride, dropout, causal, padding=padding)

    def forward(self, x):
        shortcut = self.conv1(x)
        out = self.conv2(shortcut)
        return out, shortcut


class SeqUnet(nn.Module):
    def __init__(self, num_inputs=5, num_channels=[64,72,96,108,128,128,128,128], num_outputs=128, kernel_size=7, d_model=128, causal=False, dropout=0.2, target_output_size=None, **kwargs):
        super(SeqUnet, self).__init__()
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        self.d_model = d_model

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        # Handle padding
        self.set_output_size(target_output_size)

        for i in range(self.num_levels-1):
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            self.downsampling_blocks.append(DownsamplingBlock(in_channels, num_channels[i], kernel_size, stride=2,
                                     padding=self.padding, dropout=dropout, causal=causal))

            self.upsampling_blocks.append(UpsamplingBlock(num_channels[i+1], num_channels[i], num_channels[i], kernel_size, stride=2, causal=causal,
                                                      padding=self.padding, dropout=dropout, i=(self.num_levels-i)//2))

        self.bottleneck_conv = ConvolutionBlock(num_channels[-2], num_channels[-1], kernel_size, stride=1, causal=causal, padding=self.padding, dropout=dropout)
        self.output_conv = ConvolutionBlock(num_channels[0], num_outputs, 1, 1, 0.0, False, None, False, padding=0)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size
        if target_output_size is not None:
            self.padding = 0
            self.input_size, self.output_size = self.check_padding(target_output_size)
            print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")
        else:
            print("No target output size specified. Using zero-padded convolutions assuming input does NOT have further context! Input size = output size")
            self.padding = self.kernel_size - 1

    def check_padding(self, target_output_size):
        bottleneck_size = 2
        while True:
            out = self.check_padding_for_bottleneck(bottleneck_size, target_output_size)
            if out is not False:
                return out
            bottleneck_size += 1

    def check_padding_for_bottleneck(self, bottleneck_size, target_output_size):
        # Calculate output size with current bottleneck, check if its large enough, and if layer sizes on the way are correct
        curr_size = bottleneck_size
        for i in range(self.num_levels - 1):
            curr_size = curr_size * 2 - self.kernel_size + self.padding  # UpsampleConv
            if curr_size < 2: # We need at least two samples to interpolate
                return False
            curr_size = curr_size - self.kernel_size + 1 + self.padding # Conv
            if curr_size <  2 ** (i + 1): # All computational paths created from upsampling need to be covered
                return False

        output_size = curr_size
        if output_size < target_output_size:
            return False

        # Calculate input size with current bottleneck
        curr_size = bottleneck_size
        curr_size = curr_size + self.kernel_size - 1 - self.padding # Bottleneck-Conv
        for i in range(self.num_levels - 1):
            curr_size = curr_size*2 - 2 + self.kernel_size - self.padding # Strided conv
            if curr_size % 2 == 0: # Input to strided conv needs to have odd number of elements so we can keep the edge values in decimation!
                return False
            curr_size = curr_size + self.kernel_size - 1 - self.padding # Conv

        return curr_size, output_size

    def forward(self, x, state=None):
        curr_input_size = x.shape[-1]
        if self.target_output_size is None:
            # Input size = output size. Dynamically pad input so that we can provide outputs for all inputs
            self.input_size, self.output_size = self.check_padding(curr_input_size)
            # Pad input to required input size
            pad_op = torch.nn.ConstantPad1d((self.input_size - curr_input_size, 0), 0.0)
            x = pad_op(x)
        else:
            assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        # COMPUTE OUTPUT
        # DOWNSAMPLING BLOCKS
        shortcuts = list()
        out = x
        for block in self.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        out = self.bottleneck_conv(out)

        # UPSAMPLING BLOCKS
        for block, short in reversed(list(zip(self.upsampling_blocks, shortcuts))):
            out = block(out, short)

        # OUTPUT CONVOLUTION
        out = self.output_conv(out)

        # CROP OUTPUT, IF INPUT WAS PADDED EARLIER, TO MATCH SIZES
        if self.target_output_size is None:
            assert(out.shape[-1] == x.shape[-1]) # Output size = input size (including previous padding)
            # Crop output to required output size (since input was padded earlier)
            out = out[:, :,out.shape[-1] - curr_input_size:].contiguous()

        #print(out.shape)
        return out
    
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model
    
class NConvNeXt(nn.Module):
    def __init__(self, d_model=128, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([SeqUnet()])
        self.num_layers = 2
        for i in range(self.num_layers-1):
            self.layers.append(SeqUnet(128))
    def forward(self, x, state=None):
        x = torch.nn.functional.one_hot(x, num_classes=5).type(torch.float32).permute(0,2,1)
        for i in range(self.num_layers-1):
            x = self.layers[i](x)
        return x.permute(0,2,1), None
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model



if __name__=="__main__":
    x = torch.rand(3, 100, 128)
    model = ConvNeXt(x.shape[1],12, d_model=x.shape[-1])
    y = model(x)