import copy
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
            return x    

def get_dilation_schedule(dilation_base, dilation_max, num_layers, small=True):
    if small:
        return [
            1 if i < 2 else min(dilation_max, dilation_base ** (i - 2))
            for i in range(num_layers)
        ]
    else:
        return [
            min(dilation_max, dilation_base ** i)
            for i in range(num_layers)
        ]

def build_conv_layers(hidden_dim, kernel_size, num_layers, num_stacks, dilation_base, dilation_max):
    dilations = get_dilation_schedule(dilation_base, dilation_max, num_layers)
    conv_layers = []
    
    for d in dilations:
        padding = (kernel_size - 1) // 2 * d
        conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=d, padding=padding)) # this is one "GCB" in the paper
    
    conv_layers = nn.ModuleList([copy.deepcopy(layer) for layer in conv_layers for _ in range(num_stacks)])
    return conv_layers

class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, for_representation=False, pretrain=False, dilation=2, kernel_size=9, num_conv1d=5, d_inner=2, final_conv=False, ffn=True, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.for_representation = for_representation
        self.d_model = args.hidden_dim
        self.pretrain = pretrain
        self.num_conv1d = num_conv1d
        self.d_inner = d_inner
        self.use_final_conv = final_conv
        self.num_layers = self.num_conv1d * args.num_cnn_stacks
        self.num_cnn_stacks = args.num_cnn_stacks
        self.hidden_dim = int(1.42*args.hidden_dim)
        self.ffn = ffn

        inp_size = self.alphabet_size

        self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
        
        self.rc_linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
        self.convs = build_conv_layers(args.hidden_dim, kernel_size, self.num_conv1d, args.num_cnn_stacks, dilation_base=dilation, dilation_max=1024)
        self.gates = build_conv_layers(args.hidden_dim, kernel_size, self.num_conv1d, args.num_cnn_stacks, dilation_base=dilation, dilation_max=1024)

        if ffn:
            # self.mlpgroup1 = nn.ModuleList([
            #     nn.Sequential(
            #         nn.Linear(args.hidden_dim, int(args.hidden_dim)),
            #         nn.GELU(),
            #         nn.LayerNorm(args.hidden_dim),
            #     )
            #     for _ in range(self.num_layers)
            # ])
            self.mlpgroup2 = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(args.hidden_dim, int(args.hidden_dim)),
                    nn.GELU(),
                    nn.LayerNorm(args.hidden_dim),
                )
                for _ in range(self.num_layers)
            ])

        self.milinear = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim*self.d_inner),
                                        nn.GELU(),
                                        nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                        nn.LayerNorm(args.hidden_dim*self.d_inner),
                                        nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                        nn.GELU(),
                                        nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim),
                                        nn.LayerNorm(args.hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.rc_norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        if self.use_final_conv:
            self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                    nn.GELU(),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1))
        if pretrain:
            self.out_linear = nn.Linear(args.hidden_dim, self.alphabet_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        if self.pretrain:
            mask = seq[1]
            seq = seq[0]
        
        # ACGTN - 01234
        N = seq==4
        rc_seq = 3-seq
        rc_seq[N] = 4

        rc_seq = torch.nn.functional.one_hot(rc_seq, num_classes=self.alphabet_size).type(torch.float32)
        seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)

        feat = seq.permute(0, 2, 1)
        feat = F.gelu(self.linear(feat))
        rc_feat = rc_seq.permute(0,2,1)
        rc_feat = F.gelu(self.rc_linear(rc_feat))

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            rc_h = self.dropout(rc_feat.clone())
            h = self.norms[i]((h).permute(0, 2, 1))
            rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
            g = F.sigmoid(self.gates[i](rc_h.permute(0, 2, 1)))
            h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
            feat = h*g + feat
            rc_feat = g + rc_feat
            if self.ffn:
                feat = self.mlpgroup2[i](feat.permute(0,2,1)).permute(0,2,1) + feat
        
        feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat
        if self.use_final_conv:
            feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)

        if not self.pretrain:
            if self.for_representation:
                return feat, None
            else:
                feat = self.out_linear(feat)
            return feat
        else:               
            lm_logits = self.out_linear(feat)
            
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=(lm_logits,mask)), None
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model