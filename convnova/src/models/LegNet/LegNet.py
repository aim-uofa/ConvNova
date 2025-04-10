import torch
import torch.nn as nn
import torch.nn.functional as F
# from typing import Type

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), oup),

                # Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                # nn.SiLU(),
                # nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),

                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, 
                 in_ch, 
                 ks, 
                 resize_factor,
                 filter_per_group,
                 activation, 
                 out_ch=None,
                 se_reduction=None,
                 se_type="simple",
                 inner_dim_calculation="in"
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim_calculation = inner_dim_calculation

        '''
        `in` refers to the original method of EfficientNetV2 to set the dimensionality of the EfficientNetV2-like block
        `out` is the mode used in the original LegNet approach

        This parameter slighly changes the mechanism of channel number calculation 
        which can be seen in the figure above (C, channel number is highlighted in red).
        '''
        if inner_dim_calculation == "out":
            self.inner_dim = self.out_ch * self.resize_factor
        elif inner_dim_calculation == "in":
            self.inner_dim = self.in_ch * self.resize_factor
        else:
            raise Exception(f"Wrong inner_dim_calculation: {inner_dim_calculation}")
            
        
        self.filter_per_group = filter_per_group

        se_constructor = SELayer

        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim // self.filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       se_constructor(self.in_ch, 
                                      self.inner_dim,
                                      reduction=self.se_reduction), # self.in_ch is not good
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
        )
        
      
        self.block = block
    
    def forward(self, x):
        return self.block(x)

'''
The `activation()` in the optimized architecture simply equals `nn.Identity`
In the original LegNet approach it was `nn.SiLU`
'''

class MappingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.block =  nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)


'''
Residual concatenation block is implemented below and is common between optimized and original approaches
'''

import torch

class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)

class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        
        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.out_ch,
                            kernel_size=self.ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.out_ch),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)

class legnet(nn.Module):
    """
    LegNet neural network.

    Parameters
    ----------
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                use_single_channel: bool, 
                use_reverse_channel: bool,
                final_ch,
                block_sizes=[256, 128, 128, 64, 64, 64, 64], 
                ks: int=7, 
                resize_factor: int=4, 
                activation=nn.SiLU,
                final_activation=nn.Identity,
                filter_per_group: int=1,
                se_reduction: int=4,
                res_block_type: str="concat",
                se_type: str="simple",
                inner_dim_calculation: str="in"):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.use_single_channel = use_single_channel
        self.use_reverse_channel = use_reverse_channel
        self.filter_per_group = filter_per_group
        self.final_ch = final_ch # number of bins in the competition
        self.inner_dim_calculation= inner_dim_calculation
        self.res_block_type = res_block_type
        

        residual = ResidualConcat
        
        self.stem_block = LocalBlock(in_ch=self.in_channels,
                           out_ch=block_sizes[0],
                           ks=ks,
                           activation=activation)

        blocks = []
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                residual(EffBlock(in_ch=prev_sz, 
                         out_ch=sz,
                         ks=ks,
                         resize_factor=4,
                         activation=activation,
                         filter_per_group=self.filter_per_group,
                         se_type=se_type,
                         inner_dim_calculation=inner_dim_calculation)),
                LocalBlock(in_ch=2 * prev_sz,
                               out_ch=sz,
                               ks=ks,
                               activation=activation)
            )
            blocks.append(block)

        
        self.main = nn.Sequential(*blocks)

        self.mapper =  MappingBlock(in_ch=block_sizes[-1],
                                    out_ch=self.final_ch,
                                    activation=final_activation)
        
        
        self.register_buffer('bins', torch.arange(start=0, end=final_ch, step=1, requires_grad=False))

    @property
    def in_channels(self) -> int:
        return 4 + self.use_reverse_channel + self.use_single_channel
    
    def forward(self, x):    
        x = self.stem_block(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        logprobs = F.log_softmax(x, dim=1) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        return x.unsqueeze(2), logprobs, score
       
class LegNet(nn.Module):
    def __init__(self, d_output):
        super().__init__()
        self.legnet = legnet(use_single_channel=True, use_reverse_channel=False,
                             final_ch=d_output)
    
    def forward(self, input_ids, position_ids=None, inference_params=None, state=None): # state for the repo interface
        if isinstance(input_ids, list):
            input_ids_tensor = input_ids[0]
            attention_mask = input_ids[1]
        else:
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask = None
        if position_ids is not None:
            position_ids_tensor = position_ids
        else:
            position_ids_tensor = None
        
        x = F.one_hot(input_ids, num_classes=5).float()

        outputs = self.legnet(
            x=x.permute(0,2,1),
        )[0]
        hidden_states = outputs.permute(0,2,1)

        return hidden_states, None