import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy
from src.models.sequence.convNext import NConvNeXth
from collections import namedtuple
try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None
from flash_attn.utils.distributed import  all_gather_raw

class CNNModel(nn.Module):
    def __init__(self, args, alphabet_size, num_cls, classifier=False, for_representation=False, pretrain=False, dilation=2, kernel_size=9, mlp=True, out_dim=2, length=248, use_outlinear=False, forget=False, num_conv1d=5, d_inner=2, final_conv=False, use_comp=True, **kwargs):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        self.classifier = classifier
        self.num_cls = num_cls
        self.for_representation = for_representation
        self.d_model = args.hidden_dim
        self.pretrain = pretrain
        self.mlp = mlp
        self.use_outlinear = use_outlinear
        self.forget = forget
        self.num_conv1d = num_conv1d
        self.d_inner = d_inner
        self.use_final_conv = final_conv
        self.use_comp = use_comp

        if self.args.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=args.hidden_dim)
        else:
            expanded_simplex_input = args.cls_expanded_simplex or not classifier and (args.mode == 'dirichlet' or args.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            inp_size = self.alphabet_size
            if (args.mode == 'ardm' or args.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            if self.use_comp:
                self.rc_linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
            # self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = self.num_conv1d * args.num_cnn_stacks
        self.num_cnn_stacks = args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))][:self.num_conv1d]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.gates = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**1, padding=(kernel_size-1)//2*(dilation**1)),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**2, padding=(kernel_size-1)//2*(dilation**2)),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=kernel_size, dilation=dilation**3, padding=(kernel_size-1)//2*(dilation**3))][:self.num_conv1d]
        self.gates = nn.ModuleList([copy.deepcopy(layer) for layer in self.gates for i in range(args.num_cnn_stacks)])
        # self.convs = nn.ModuleList([NConvNeXth(d_model=args.hidden_dim, in_chans=args.hidden_dim) for _ in range(args.num_cnn_stacks)])
        # self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        if self.mlp:
            self.milinear = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim*self.d_inner),
                                          nn.GELU(),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                          nn.LayerNorm(args.hidden_dim*self.d_inner),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim*self.d_inner),
                                          nn.GELU(),
                                          nn.Linear(args.hidden_dim*self.d_inner, args.hidden_dim),
                                          nn.LayerNorm(args.hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        if self.use_comp:
            self.rc_norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        if self.use_final_conv:
            self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                    nn.GELU(),
                                    nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1))
        if pretrain:
            self.out_linear = nn.Linear(args.hidden_dim, self.alphabet_size)
        elif self.use_outlinear:
            self.out_linear = nn.Linear(args.hidden_dim, out_dim)
            self.pool = nn.MaxPool1d(length)

        self.dropout = nn.Dropout(args.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(args.hidden_dim, self.num_cls))

        if self.args.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=args.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])

    def forward(self, seq, t=None, cls = None, return_embedding=False, state=None):
        # if t is None and self.for_representation:
        #     # t = torch.tensor([self.args.alpha_max])[None].expand(seq.shape[0]).to(seq.device)
        #     seq, alphas = sample_cond_prob_path(self.args, seq, self.alphabet_size)
        #     seq, prior_weights = expand_simplex(seq,alphas, self.args.prior_pseudocount)
        #     t = alphas
        if not self.pretrain:
            if self.use_comp:
                # ACGTN - 01234
                N = seq==4
                rc_seq = 3-seq
                rc_seq[N] = 4
                seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
                rc_seq = torch.nn.functional.one_hot(rc_seq, num_classes=self.alphabet_size).type(torch.float32)
                
                # time_emb = F.relu(self.time_embedder(t))
                feat = seq.permute(0, 2, 1)
                feat = F.gelu(self.linear(feat))
                rc_feat = rc_seq.permute(0,2,1)
                rc_feat = F.gelu(self.rc_linear(rc_feat))

                # if self.args.cls_free_guidance and not self.classifier:
                #     cls_emb = self.cls_embedder(cls)

                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                        rc_feat = g + rc_feat
                    else:
                        feat = h
                    # if self.mlp and i == (self.num_layers//2-1):
                    #     feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat
                if self.mlp:
                    feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat
                
                if self.use_final_conv:
                    feat = self.final_conv(feat)
                feat = feat.permute(0, 2, 1)
                if self.for_representation:
                    if self.use_outlinear:
                        feat = self.pool(feat.permute(0,2,1)).permute(0,2,1)
                        return self.out_linear(feat), None
                    else:
                        return feat, None
                if self.classifier:
                    feat = feat.mean(dim=1)
                    if return_embedding:
                        embedding = self.cls_head[:1](feat)
                        return self.cls_head[1:](embedding), embedding
                    else:
                        return self.cls_head(feat)
                else:
                    feat = self.out_linear(feat)
                return feat
            else:
                seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
                
                # time_emb = F.relu(self.time_embedder(t))
                feat = seq.permute(0, 2, 1)
                feat = F.gelu(self.linear(feat))

                if self.args.cls_free_guidance and not self.classifier:
                    cls_emb = self.cls_embedder(cls)

                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                    else:
                        feat = h
                if self.mlp:
                    feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat
                
                if self.use_final_conv:
                    feat = self.final_conv(feat)
                feat = feat.permute(0, 2, 1)
                if self.for_representation:
                    if self.use_outlinear:
                        feat = self.pool(feat)
                        print(feat.shape)
                        return self.out_linear(feat.reshape(feat.shape[0], -1)), None
                    else:
                        return feat, None
                if self.classifier:
                    feat = feat.mean(dim=1)
                    if return_embedding:
                        embedding = self.cls_head[:1](feat)
                        return self.cls_head[1:](embedding), embedding
                    else:
                        return self.cls_head(feat)
                else:
                    feat = self.out_linear(feat)
                return feat
        else:
            if self.use_comp:
                mask = seq[1]
                seq = seq[0]
                inference_params = None
                ColumnParallelLinear = None
                N = seq==4
                rc_seq = 3-seq
                rc_seq[N] = 4
                rc_seq = torch.nn.functional.one_hot(rc_seq, num_classes=self.alphabet_size).type(torch.float32)
                seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
                # time_emb = F.relu(self.time_embedder(t))
                feat = seq.permute(0, 2, 1)
                feat = F.relu(self.linear(feat))
                rc_feat = rc_seq.permute(0,2,1)
                rc_feat = F.gelu(self.rc_linear(rc_feat))


                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    rc_h = self.dropout(rc_feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    rc_h = self.rc_norms[i]((rc_h).permute(0,2,1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](rc_h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](rc_h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                    else:
                        feat = h
                    # if self.mlp and i == (self.num_layers//2-1):
                    #     feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat

                if self.mlp:
                    feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat

                if self.use_final_conv: 
                    feat = self.final_conv(feat)
                feat = feat.permute(0, 2, 1)
                lm_logits = self.out_linear(feat)
                # During inference, we want the full logit for sampling
                if ColumnParallelLinear is not None and inference_params is not None:
                    if isinstance(self.out_linear, ColumnParallelLinear):
                        lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                        lm_logits = rearrange(
                            lm_logits, "(n b) s d -> b s (n d)", b=feat.shape[0]
                        )
                CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            else:
                mask = seq[1]
                seq = seq[0]
                inference_params = None
                ColumnParallelLinear = None
                seq = torch.nn.functional.one_hot(seq, num_classes=self.alphabet_size).type(torch.float32)
                # time_emb = F.relu(self.time_embedder(t))
                feat = seq.permute(0, 2, 1)
                feat = F.relu(self.linear(feat))

                for i in range(self.num_layers):
                    h = self.dropout(feat.clone())
                    h = self.norms[i]((h).permute(0, 2, 1))
                    if self.forget:
                        g = F.sigmoid(self.gates[i](h.permute(0, 2, 1)))
                    else:
                        g = F.gelu(self.gates[i](h.permute(0, 2, 1)))
                    h = F.gelu(self.convs[i](h.permute(0, 2, 1)))
                    if h.shape == feat.shape:
                        if self.forget:
                            feat = h*g + feat
                        else:
                            feat = h + g + feat
                    else:
                        feat = h
                    # if self.mlp and i == (self.num_layers//2-1):
                    #     feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat

                if self.mlp:
                    feat = self.milinear(feat.permute(0,2,1)).permute(0,2,1)+feat

                if self.use_final_conv: 
                    feat = self.final_conv(feat)
                feat = feat.permute(0, 2, 1)
                lm_logits = self.out_linear(feat)
                # During inference, we want the full logit for sampling
                if ColumnParallelLinear is not None and inference_params is not None:
                    if isinstance(self.out_linear, ColumnParallelLinear):
                        lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                        lm_logits = rearrange(
                            lm_logits, "(n b) s d -> b s (n d)", b=feat.shape[0]
                        )
                CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=(lm_logits,mask)), None
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model