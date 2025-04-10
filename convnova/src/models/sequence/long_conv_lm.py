import copy
import math
import re
from functools import partial

from collections import namedtuple, OrderedDict
from collections.abc import Sequence  
from typing import Optional, Union, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from einops import rearrange

from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedMLP, ParallelFusedMLP
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.distributed import sync_shared_params, all_gather_raw

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm, dropout_add_rms_norm = None, None
from flash_attn.ops.rms_norm import dropout_add_rms_norm
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from src.utils import instantiate
import src.utils.registry as registry
from src.models.sequence.mha import BertEmbeddings
from src.models.sequence.bert_padding import (index_first_axis,
                                            index_put_first_axis, pad_input,
                                            unpad_input, unpad_input_only)

try:
    from src.models.sequence.flash_attn_triton import flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None
from torchvision.ops import StochasticDepth

class state_2_token_MLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        expand,
        token_length,
        p=0.1
    ):
        super().__init__()
        self.l1 = nn.Linear(d_state, 1)
        self.af1 = nn.GELU()
        self.l2 = nn.Linear(expand*d_model, d_model)
        self.af2 = nn.GELU()
        self.l3 = nn.Linear(d_model, token_length*d_model)
        self.drop = nn.Dropout(p=p)
        self.d_model = d_model
        self.token_length = token_length
    def forward(self, x):
        x = torch.squeeze(self.l1(x), dim=-1)
        x = self.drop(x)
        x = self.af1(x)
        x = self.l2(x)
        x = self.af2(x)
        x = self.l3(x)
        return x.reshape(-1, self.token_length, self.d_model)

class AlibiBlock(nn.Module):

    def __init__(self, dim, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., fused_dropout_add_ln=False, return_residual=False,
                 residual_in_fp32=False, sequence_parallel=False, mark_shared_params=False, alibi_starting_size=512,
                 d_inner=None,process_group=None,layer=None,attn_layer_idx=None,attn_cfg=None,layer_norm_epsilon=1e-5,
                 fused_mlp=False,identity_mlp=False,layer_idx=None,checkpoint_mlp=False,checkpoint_mixer=False,device=None,dtype=None,):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        d_model=dim
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        mixer_cls = create_mixer_cls(
            layer=layer,
            process_group = process_group,
            attn_layer_idx = attn_layer_idx,
            attn_cfg = attn_cfg,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
            **factory_kwargs
        )
        mlp_cls = create_mlp_cls(
            d_model,
            d_inner=d_inner,
            process_group=process_group,
            fused_mlp=fused_mlp,
            identity_mlp=identity_mlp,
            sequence_parallel=sequence_parallel,
            **factory_kwargs
        )
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_layer_norm is not installed'
            assert dropout_add_rms_norm is not None, 'dropout_layer_norm is not installed'
            assert (isinstance(self.norm1, (nn.LayerNorm, RMSNorm))
                    and isinstance(self.dropout1, nn.Dropout))

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, 'norm2'):
                for p in self.norm2.parameters():
                    p._shared_params = True
        self.num_attention_heads = 8
        self._current_alibi_size = int(alibi_starting_size)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.alibi = torch.zeros(
            (1, self.num_attention_heads, self._current_alibi_size,
             self._current_alibi_size))
        self.rebuild_alibi_tensor(size=alibi_starting_size, device=device)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def rebuild_alibi_tensor(self,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:

            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = (2**(-2**-(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(
            n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """

        attention_mask = torch.triu(torch.ones(hidden_states.size(-2), hidden_states.size(-2),
                                              dtype=torch.bool, device=hidden_states.device),
                                       diagonal=1)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        def _fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)
        attention_mask = torch.triu(
                _fill_with_neg_inf(torch.zeros([hidden_states.size(-2), hidden_states.size(-2)])), 1
            )
        extended_attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32, device=hidden_states.device)  # fp16 compatibility

        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        # Unpad inputs and mask. It will remove tokens that are padded.
        # Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens.
        # Then unpadding performs the following compression of the inputs:
        # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        attention_mask = torch.ones(batch, seqlen, device=hidden_states.device).type(torch.bool)
        origin_shape = hidden_states.shape
        hidden_states, indices, cu_seqlens, _ = unpad_input(
            hidden_states, attention_mask)

        # Add alibi matrix to extended_attention_mask
        if self._current_alibi_size < seqlen:
            # Rebuild the alibi tensor when needed
            # warnings.warn(
            #     f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
            # )
            self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
        elif self.alibi.device != hidden_states.device:
            # Device catch-up
            self.alibi = self.alibi.to(hidden_states.device)
        alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
        attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
        alibi_attn_mask = attn_bias + alibi_bias
        
        fused_add_norm_fn = (dropout_add_rms_norm if RMSNorm and isinstance(self.norm1, RMSNorm)
                             else dropout_add_layer_norm)
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                if residual is not None:
                    residual = residual.reshape(dropped.shape)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        hidden_states.shape[:-1], device=hidden_states.device,
                        dtype=hidden_states.dtype)
                    )
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, residual, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=True, residual_in_fp32=self.residual_in_fp32
                )
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states,
                                        cu_seqlens,
                                        seqlen,
                                        None,
                                        indices,
                                        attn_mask=attention_mask,
                                        bias=alibi_attn_mask,
                                        **mixer_kwargs).reshape(origin_shape)
            residual = residual.reshape(origin_shape)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            hidden_states.shape[:-1], device=hidden_states.device,
                            dtype=hidden_states.dtype)
                        )
                    hidden_states, residual = fused_add_norm_fn(
                        hidden_states, residual, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=True, residual_in_fp32=self.residual_in_fp32
                    )
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states,
                cu_seqlens,
                seqlen,
                None,
                indices,
                attn_mask=attention_mask,
                bias=alibi_attn_mask, 
                **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                            + hidden_states).to(dtype=self.norm1.weight.dtype))
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(torch.ones(
                        mixer_out.shape[:-1], device=mixer_out.device, dtype=mixer_out.dtype)
                    )
                hidden_states = fused_add_norm_fn(
                    mixer_out, hidden_states, self.norm1.weight, self.norm1.bias,
                    self.dropout1.p if self.training else 0.0, self.norm1.eps,
                    rowscale=rowscale1, prenorm=False
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                                + hidden_states).to(dtype=self.norm2.weight.dtype))
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(torch.ones(
                            mlp_out.shape[:-1], device=mlp_out.device, dtype=mlp_out.dtype)
                        )
                    hidden_states = fused_add_norm_fn(
                        mlp_out, hidden_states, self.norm2.weight, self.norm2.bias,
                        self.dropout2.p if self.training else 0.0, self.norm2.eps,
                        rowscale=rowscale2, prenorm=False
                    )
            return hidden_states
       
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
                                       as_tuple=False).flatten()
            hidden_states = self.layer[-1](hidden_states,
                                           cu_seqlens,
                                           seqlen,
                                           subset_idx=subset_idx,
                                           indices=indices,
                                           attn_mask=attention_mask,
                                           bias=alibi_attn_mask)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class Pyramid(nn.Module):
    def __init__(
        self,
        d_model,
        d_inner=None,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        layer_norm_epsilon=1e-5,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        residual_in_fp32=False,
        fused_mlp=False,
        identity_mlp=False,
        fused_dropout_add_ln=False,
        layer_idx=None,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        mha_layer = layer['mha']
        ssm_layer = layer['ssm']
        self.use_residual = layer['use_residual']
        self.num_tf = layer['num_tf']
        self.global_token_length = layer['global_token_length']
        self.layer_idx = layer_idx
        self.return_last_state = ssm_layer['return_last_state']
        # if self.return_last_state:
        #     self.state_2_token = state_2_token_MLP(d_model, ssm_layer['d_state'], ssm_layer['expand'], self.global_token_length, 0.1)
        
        self.tf = Block(
            d_model,
            create_mixer_cls(
                layer=mha_layer,
                process_group=process_group,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                layer_idx=layer_idx,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            ),
            create_mlp_cls(
                d_model,
                d_inner=d_inner,
                process_group=process_group,
                fused_mlp=fused_mlp,
                identity_mlp=identity_mlp,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            ),
            norm_cls=partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs),
            prenorm=True,
            resid_dropout1=resid_dropout1,
            resid_dropout2=resid_dropout2,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
        )
        self.tf.layer_idx = layer_idx

        self.ssm = MambaBlock(
            d_model, 
            create_mixer_cls(
                layer=ssm_layer,
                process_group=process_group,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                layer_idx=layer_idx,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            ), 
            create_mlp_cls(
                d_model,
                d_inner=d_inner,
                process_group=process_group,
                fused_mlp=fused_mlp,
                identity_mlp=identity_mlp,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            ),
            norm_cls=partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs),
            prenorm=True, 
            resid_dropout1=resid_dropout1, 
            resid_dropout2=resid_dropout2,
            fused_dropout_add_ln=fused_dropout_add_ln, 
            return_residual=self.use_residual,
            residual_in_fp32=residual_in_fp32, 
            sequence_parallel=sequence_parallel and process_group is not None, 
            mark_shared_params=process_group is not None, 
            return_last_state=self.return_last_state
        )
        self.ssm.layer_idx = layer_idx

        if checkpoint_mlp:
            self.tf.mlp = CheckpointedModule(self.tf.mlp)
            self.ssm.mlp = CheckpointedModule(self.ssm.mlp)
        if checkpoint_mixer:
            self.tf.mixer = CheckpointedModule(self.tf.mixer)
            self.ssm.mixer = CheckpointedModule(self.ssm.mlp)
    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None,
                mixer_subset=None, mixer_kwargs=None):
        if self.use_residual:
            hidden_states = rearrange(hidden_states, "b (t l) d -> (b t) l d", t=self.num_tf)
            if residual != None: # the first layer of transforemer that do not have residual
                residual = rearrange(residual, "b (t l) d -> (b t) l d", t=self.num_tf)
            if self.layer_idx != 0: # abort the global tokens
                if self.return_last_state:
                    hidden_states = hidden_states[:, :-self.global_token_length, :]
                    if residual != None: # the first layer of transforemer that do not have residual
                        residual = residual[:, :-self.global_token_length, :]
                else:
                    hidden_states = hidden_states[:, :-1, :]
                    if residual != None: # the first layer of transforemer that do not have residual
                        residual = residual[:, :-1, :]
            hidden_states, residual = self.tf(hidden_states, residual, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
            hidden_states = rearrange(hidden_states, "(b t) l d -> b (t l) d", t=self.num_tf)
            residual = rearrange(residual, "(b t) l d -> b (t l) d", t=self.num_tf)
            if self.return_last_state:
                hidden_states, residual, last_ssm_state = self.ssm(hidden_states, residual, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
                # global_token_hidden = self.state_2_token(last_ssm_state)
                global_token_hidden = rearrange(last_ssm_state, "b (d e) s -> b (e s) d", e=2)
                global_token_residual = global_token_hidden
            else:
                hidden_states, residual = self.ssm(hidden_states, residual, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
                global_token_hidden = hidden_states[:, -1:, :]
                global_token_residual = residual[:, -1:, :]
            global_token_hidden = global_token_hidden.repeat(self.num_tf, 1, 1)
            global_token_residual = global_token_residual.repeat(self.num_tf, 1, 1)
            hidden_states = rearrange(hidden_states, "b (t l) d -> (b t) l d", t=self.num_tf)
            residual = rearrange(residual, "b (t l) d -> (b t) l d", t=self.num_tf)
            hidden_states = torch.concat([hidden_states, global_token_hidden], dim=1)
            residual = torch.concat([residual, global_token_residual], dim=1)
            hidden_states = rearrange(hidden_states, "(b t) l d -> b (t l) d", t=self.num_tf)
            residual = rearrange(residual, "(b t) l d -> b (t l) d", t=self.num_tf)
            return hidden_states, residual
        else:
            hidden_states = rearrange(hidden_states, "b (t l) d -> (b t) l d", t=self.num_tf)
            if self.layer_idx != 0: # abort the global tokens
                if self.return_last_state:
                    hidden_states = hidden_states[:, :-self.global_token_length, :]
                else:
                    hidden_states = hidden_states[:, :-1, :]
            hidden_states = self.tf(hidden_states, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
            hidden_states = rearrange(hidden_states, "(b t) l d -> b (t l) d", t=self.num_tf)
            if self.return_last_state:
                hidden_states, residual, last_ssm_state = self.ssm(hidden_states, residual, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
                global_token_hidden = self.state_2_token(last_ssm_state)
            else:
                hidden_states, residual = self.ssm(hidden_states, residual, mixer_kwargs=mixer_kwargs, mixer_subset=mixer_subset)
                global_token_hidden = hidden_states[:, -1:, :]
            global_token_hidden = global_token_hidden.repeat(self.num_tf, 1, 1)
            hidden_states = rearrange(hidden_states, "b (t l) d -> (b t) l d", t=self.num_tf)
            hidden_states = torch.concat([hidden_states, global_token_hidden], dim=1)
            hidden_states = rearrange(hidden_states, "(b t) l d -> b (t l) d", t=self.num_tf)
            return hidden_states

        

class CheckpointedModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return checkpoint(self.layer, x)


def create_mixer_cls(
    layer=None,
    process_group=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_idx=None,
    sequence_parallel=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    parallel_kwargs = (
        {"process_group": process_group, "sequence_parallel": sequence_parallel}
        if process_group is not None
        else {}
    )
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
        fused_bias_fc = (
            False if attn_cfg is None else attn_cfg.get("fused_bias_fc", False)
        )
        if not fused_bias_fc:
            assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
        mha_cls = MHA if process_group is None else ParallelMHA
        # ParallelMHA doesn't take 'fused_bias_fc', it is assumed that we fuse matmul + bias
        if process_group is not None:
            attn_cfg = copy.deepcopy(attn_cfg)  # Don't modify the original cfg
            attn_cfg.pop("fused_bias_fc", None)
        mixer_cls = partial(
            mha_cls,
            causal=causal,
            layer_idx=layer_idx,
            **(attn_cfg if attn_cfg is not None else {}),
            **parallel_kwargs,
            **factory_kwargs,
        )
    # elif layer is not None and layer['_name_']=='mha':
    #     causal = True if attn_cfg is None else attn_cfg.pop("causal", True)
    #     fused_bias_fc = (
    #         False if attn_cfg is None else attn_cfg.get("fused_bias_fc", False)
    #     )
    #     if not fused_bias_fc:
    #         assert process_group is None, "TensorParallel MHA requires fused_bias_fc"
    #     mha_cls = MHA if process_group is None else ParallelMHA
    #     # ParallelMHA doesn't take 'fused_bias_fc', it is assumed that we fuse matmul + bias
    #     if process_group is not None:
    #         attn_cfg = copy.deepcopy(attn_cfg)  # Don't modify the original cfg
    #         attn_cfg.pop("fused_bias_fc", None)
    #     mixer_cls = partial(
    #         mha_cls,
    #         causal=causal,
    #         layer_idx=layer_idx,
    #         **(attn_cfg if attn_cfg is not None else {}),
    #         **parallel_kwargs,
    #         **factory_kwargs,
    #     )
    else:
        fused_bias_fc = False if layer is None else layer.get("fused_bias_fc", False)
        if process_group is not None:
            assert fused_bias_fc, "TensorParallel SSM requires fused_bias_fc"
        mixer_cls = instantiate(
            registry.layer,
            layer,
            partial=True,
            layer_idx=layer_idx,
            **factory_kwargs,
            **parallel_kwargs,
        )
        # mixer_cls = partial(ssm_cls, layer_idx=layer_idx,
        #                     **(ssm_cfg if ssm_cfg is not None else {}),
        #                     **parallel_kwargs, **factory_kwargs)
    return mixer_cls


def create_mlp_cls(
    d_model,
    d_inner=None,
    process_group=None,
    fused_mlp=False,
    sequence_parallel=True,
    identity_mlp=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if process_group is not None:
        assert fused_mlp, "Tensor Parallel is only implemented for FusedMLP"

    if not fused_mlp and not identity_mlp:
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate="tanh"),
            **factory_kwargs,
        )
    elif fused_mlp:
        mlp_cls = FusedMLP if process_group is None else ParallelFusedMLP
        parallel_kwargs = (
            {"process_group": process_group, "sequence_parallel": sequence_parallel}
            if process_group is not None
            else {}
        )
        mlp_cls = partial(
            mlp_cls, hidden_features=inner_dim, **parallel_kwargs, **factory_kwargs
        )
    else:
        mlp_cls = nn.Identity
    return mlp_cls


def create_block(
    d_model,
    d_inner=None,
    process_group=None,
    layer=None,
    attn_layer_idx=None,
    attn_cfg=None,
    layer_norm_epsilon=1e-5,
    resid_dropout1=0.0,
    resid_dropout2=0.0,
    residual_in_fp32=False,
    fused_mlp=False,
    identity_mlp=False,
    fused_dropout_add_ln=False,
    layer_idx=None,
    sequence_parallel=True,
    checkpoint_mlp=False,
    checkpoint_mixer=False,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(
        layer=layer,
        process_group=process_group,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        layer_idx=layer_idx,
        sequence_parallel=sequence_parallel,
        **factory_kwargs,
    )
    mlp_cls = create_mlp_cls(
        d_model,
        d_inner=d_inner,
        process_group=process_group,
        fused_mlp=fused_mlp,
        identity_mlp=identity_mlp,
        sequence_parallel=sequence_parallel,
        **factory_kwargs,
    )
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=True,
        resid_dropout1=resid_dropout1,
        resid_dropout2=resid_dropout2,
        fused_dropout_add_ln=fused_dropout_add_ln,
        residual_in_fp32=residual_in_fp32,
        sequence_parallel=sequence_parallel and process_group is not None,
        mark_shared_params=process_group is not None,
    )

    block.layer_idx = layer_idx

    if checkpoint_mlp:
        block.mlp = CheckpointedModule(block.mlp)
    if checkpoint_mixer:
        block.mixer = CheckpointedModule(block.mixer)
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
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
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # nn.init.normal_(
                #     p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                # )
                nn.init.kaiming_normal_(p)
            # elif name in ['mha.in_proj_bias', 'mha.out_proj_bias']:
            #     nn.init.kaiming_uniform_(p)
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    # nn.init.normal_(
                    #     p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                    # )
                    nn.init.kaiming_normal_(p)
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    # nn.init.normal_(
                    #     p[: out_features // 2],
                    #     mean=0.0,
                    #     std=initializer_range / math.sqrt(2 * n_layer) * 2,
                    # )
                    nn.init.kaiming_normal_(p[: out_features//2])

class LMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        identity_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.residual_in_fp32 = residual_in_fp32

        if process_group is None:
            self.embeddings = GPT2Embeddings(
                d_model, vocab_size, max_position_embeddings, **factory_kwargs
            )
            # self.embeddings = BertEmbeddings(
            #     vocab_size=vocab_size,
            #     hidden_size=d_model,
            #     max_position_embeddings=kwargs['length'],
            #     pad_token_id=None,
            #     **factory_kwargs
            # )
        else:
            self.embeddings = ParallelGPT2Embeddings(
                d_model,
                vocab_size,
                max_position_embeddings,
                process_group=process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        self.layer_name = None
        if layer != None:
            self.layer_name = layer['_name_']
        if layer!=None and layer['_name_'] == 'pyramid':
            self.layers = nn.ModuleList(
                [
                    Pyramid(
                        d_model,
                        d_inner=d_inner,
                        process_group=process_group,
                        layer=layer,
                        attn_layer_idx=attn_layer_idx,
                        attn_cfg=attn_cfg,
                        layer_norm_epsilon=layer_norm_epsilon,
                        resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                        resid_dropout2=resid_dropout,
                        residual_in_fp32=residual_in_fp32,
                        fused_mlp=fused_mlp,
                        identity_mlp=identity_mlp,
                        fused_dropout_add_ln=fused_dropout_add_ln,
                        layer_idx=i,
                        sequence_parallel=self.sequence_parallel,
                        checkpoint_mlp=checkpoint_mlp,
                        checkpoint_mixer=checkpoint_mixer,
                        **factory_kwargs,
                    )
                    for i in range(n_layer)
                ]
            )
        # elif layer!=None and layer['_name_'] == 'nt':
        #     self.layers = nn.ModuleList(
        #         [
        #             create_block(
        #                 d_model,
        #                 d_inner=d_inner,
        #                 process_group=process_group,
        #                 layer=layer,
        #                 attn_layer_idx=attn_layer_idx,
        #                 attn_cfg=attn_cfg,
        #                 layer_norm_epsilon=layer_norm_epsilon,
        #                 resid_dropout1=embed_dropout if i == 0 else resid_dropout,
        #                 resid_dropout2=resid_dropout,
        #                 residual_in_fp32=residual_in_fp32,
        #                 fused_mlp=fused_mlp,
        #                 identity_mlp=identity_mlp,
        #                 fused_dropout_add_ln=fused_dropout_add_ln,
        #                 layer_idx=i,
        #                 sequence_parallel=self.sequence_parallel,
        #                 checkpoint_mlp=checkpoint_mlp,
        #                 checkpoint_mixer=checkpoint_mixer,
        #                 **factory_kwargs,
        #             )
        #             for i in range(n_layer)
        #         ]
        #     )
        elif layer!=None and layer['_name_'] == 'bert':
            self.layers = nn.ModuleList(
                [
                    AlibiBlock(
                        d_model,
                        d_inner=d_inner,
                        process_group=process_group,
                        layer=layer,
                        attn_layer_idx=attn_layer_idx,
                        attn_cfg=attn_cfg,
                        layer_norm_epsilon=layer_norm_epsilon,
                        resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                        resid_dropout2=resid_dropout,
                        residual_in_fp32=residual_in_fp32,
                        fused_mlp=fused_mlp,
                        identity_mlp=identity_mlp,
                        fused_dropout_add_ln=fused_dropout_add_ln,
                        layer_idx=i,
                        sequence_parallel=self.sequence_parallel,
                        checkpoint_mlp=checkpoint_mlp,
                        checkpoint_mixer=checkpoint_mixer,
                        **factory_kwargs,
                    )
                    for i in range(n_layer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    create_block(
                        d_model,
                        d_inner=d_inner,
                        process_group=process_group,
                        layer=layer,
                        attn_layer_idx=attn_layer_idx,
                        attn_cfg=attn_cfg,
                        layer_norm_epsilon=layer_norm_epsilon,
                        resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                        resid_dropout2=resid_dropout,
                        residual_in_fp32=residual_in_fp32,
                        fused_mlp=fused_mlp,
                        identity_mlp=identity_mlp,
                        fused_dropout_add_ln=fused_dropout_add_ln,
                        layer_idx=i,
                        sequence_parallel=self.sequence_parallel,
                        checkpoint_mlp=checkpoint_mlp,
                        checkpoint_mixer=checkpoint_mixer,
                        **factory_kwargs,
                    )
                    for i in range(n_layer)
                ]
            )

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        if process_group is not None:
            for p in self.ln_f.parameters():
                # Mark the norm parameters as "shared_params" so that we sync their values at init.
                p._shared_params = True
                # Mark the norm params as "sequence_parallel" so we run all-reduce on their grads.
                if self.sequence_parallel:
                    p._sequence_parallel = True

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None):
        # If using Tensor Parallel with sequence parallel, we combine the batch and the seqlen
        # dimensions so that we can split on it easily, in case of small batch size.
        # Only the attention/SSM layers need to know the seqlen.
        embedding_kwargs = (
            {"combine_batch_seqlen_dim": True}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, **embedding_kwargs
        )
        assert not torch.isnan(hidden_states).any()
        residual = None
        mixer_kwargs = (
            {"seqlen": input_ids.shape[1]}
            if self.process_group is not None and self.sequence_parallel
            else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.layers:
            try:
                assert not torch.isnan(hidden_states).any()
            except:
                raise RuntimeError(f"{layer.layer_idx}")
            hidden_states, residual = layer(
                hidden_states, residual, mixer_kwargs=mixer_kwargs
            )
        if self.layer_name == "pyramid":
            hidden_states = rearrange(hidden_states, "b (t l) d -> (b t) l d", t=self.layers[0].num_tf)
            if self.layers[-1].use_residual:
                residual = rearrange(residual, "b (t l) d -> (b t) l d", t=self.layers[-1].num_tf)
                if self.layers[-1].return_last_state:
                    residual = residual[:, :-self.layers[-1].global_token_length, :]
                else:
                    residual = residual[:, :-1, :]
                residual = rearrange(residual, "(b t) l d -> b (t l) d", t=self.layers[-1].num_tf)
            if self.layers[-1].return_last_state:
                hidden_states = hidden_states[:, :-self.layers[-1].global_token_length, :]
            else:
                hidden_states = hidden_states[:, :-1, :]
            hidden_states = rearrange(hidden_states, "(b t) l d -> b (t l) d", t=self.layers[-1].num_tf)
                    
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class BertLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
            **kwargs,
        )
        if process_group is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                d_model,
                vocab_size,
                process_group,
                bias=False,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(
        self, input_ids, position_ids=None, inference_params=None, state=None
    ):  # state for the repo interface
        mask = input_ids[1]
        input_ids = input_ids[0]
        hidden_states = self.backbone(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if ColumnParallelLinear is not None and inference_params is not None:
            if isinstance(self.lm_head, ColumnParallelLinear):
                lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                lm_logits = rearrange(
                    lm_logits, "(n b) s d -> b s (n d)", b=hidden_states.shape[0]
                )
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=(lm_logits,mask)), None

class ConvLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_inner: int,
        vocab_size: int,
        process_group=None,
        layer=None,
        attn_layer_idx=None,
        attn_cfg=None,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        dropout_cls=nn.Dropout,
        layer_norm_epsilon: float = 1e-5,
        initializer_cfg=None,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        pad_vocab_size_multiple: int = 1,
        sequence_parallel=True,
        checkpoint_mlp=False,
        checkpoint_mixer=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.backbone = LMBackbone(
            d_model=d_model,
            n_layer=n_layer,
            d_inner=d_inner,
            vocab_size=vocab_size,
            process_group=process_group,
            layer=layer,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            dropout_cls=dropout_cls,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg,
            fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            checkpoint_mlp=checkpoint_mlp,
            checkpoint_mixer=checkpoint_mixer,
            **factory_kwargs,
            **kwargs,
        )
        if process_group is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                d_model,
                vocab_size,
                process_group,
                bias=False,
                sequence_parallel=sequence_parallel,
                **factory_kwargs,
            )
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(
        self, input_ids, position_ids=None, inference_params=None, state=None
    ):  # state for the repo interface
        hidden_states = self.backbone(
            input_ids, position_ids=position_ids, inference_params=inference_params
        )
        lm_logits = self.lm_head(hidden_states)
        # During inference, we want the full logit for sampling
        if ColumnParallelLinear is not None and inference_params is not None:
            if isinstance(self.lm_head, ColumnParallelLinear):
                lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
                lm_logits = rearrange(
                    lm_logits, "(n b) s d -> b s (n d)", b=hidden_states.shape[0]
                )
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits), None


class DNAEmbeddingModel(nn.Module, GenerationMixin):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, sequence_parallel=True,
                 device=None, dtype=None, return_hidden_state=False, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model  # for decoder
        self.process_group = process_group
        self.return_hidden_state = return_hidden_state
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            process_group=process_group,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            dropout_cls=dropout_cls, layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel,
            **factory_kwargs, **kwargs
        )
        if process_group is None:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError('fused_dense_lib is not installed')
            self.lm_head = ColumnParallelLinear(
                d_model, vocab_size, process_group, bias=False,
                sequence_parallel=sequence_parallel, **factory_kwargs
            )
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None): # state for the repo interface
        hidden_states = self.backbone(input_ids, position_ids=position_ids,
                                      inference_params=inference_params)
        # we only need the last hidden state for embeddings (decoder head will predict classification task)
        return hidden_states, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model


def load_backbone(model, state_dict, freeze_backbone=False, ignore_head=True):
    """

    Modifies state dict loading with custom function.  Every layer in new model will be

    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes (eg, imagenet to hmdb51), then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # consumes prefix from pretrained model, if necessary
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict, "model."
    )

    model_new_params_dict = model.state_dict()
    updated_model_state_dict = {}

    # loop through scratch model keys (pretrained may have extra stuff)
    for key in sorted(model_new_params_dict.keys()):

        loaded_params = state_dict.get(key, None)
        # make sure key is in the loaded params first, if not, then print it out
    
        if loaded_params is None:
            # This should never happen, it should be there!
            print("Missing key in pretrained model!", key)
            raise Exception

        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, load from scratch", key)
            # using scratch by default, nothing needed
            used_params = model_new_params_dict[key]

        elif "decoder" in key:
            print("found decoder key / parameter, load from scratch", key)
            used_params = model_new_params_dict[key]
        else:
            print('key: shape MATCH, loading', key)  # load matched weights
            used_params = loaded_params

        # we need to pass back a state dict with the '.model' prefix!!!!!
        key_with_prefix = 'model.' + key
        updated_model_state_dict[key_with_prefix] = used_params

    if freeze_backbone:
        print("freezing model backbone params!")
        # note, decoder not included in backbone
        for name, param in model.named_parameters():
            param.requires_grad = False

    # we have updated the new model state dict with pretrained now
    return updated_model_state_dict


def shard_state_dict_tp(state_dict, world_size, rank, pad_vocab_size_multiple=1):
    """Convert the state_dict of a standard SSM model to the state_dict of a SSM model
    with tensor parallel.
    """
    layer_idx_match = [
        re.search(r"backbone\.layers\.(\d+)\.", k) for k in state_dict.keys()
    ]
    num_hidden_layers = len(set(m.group(1) for m in layer_idx_match if m is not None))
    vocab_size = state_dict["backbone.embeddings.word_embeddings.weight"].shape[0]
    inner_dim, hidden_size = state_dict["backbone.layers.0.mlp.fc1.weight"].shape
    vocab_size = (
        math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    assert vocab_size % world_size == 0
    assert hidden_size % world_size == 0
    assert inner_dim % world_size == 0

    def shard_dim(state_dict, key, dim=0):
        x = state_dict[key]
        dimension = x.shape[dim] // world_size
        state_dict[key] = x.narrow(dim, rank * dimension, dimension)

    def shard_qkv_headdim(state_dict, key):
        x = rearrange(state_dict[key], "(three d) ... -> three d ...", three=3)
        dim = x.shape[1] // world_size
        state_dict[key] = rearrange(
            x[:, rank * dim : (rank + 1) * dim], "three d ... -> (three d) ..."
        )

    shard_dim(state_dict, "backbone.embeddings.word_embeddings.weight", 0)
    if "lm_head.weight" in state_dict:
        shard_dim(state_dict, "lm_head.weight", 0)
    if "backbone.embeddings.position_embeddings.weight" in state_dict:
        shard_dim(state_dict, "backbone.embeddings.position_embeddings.weight", -1)
    for i in range(num_hidden_layers):
        shard_qkv_headdim(state_dict, f"backbone.layers.{i}.mixer.Wqkv.weight")
        shard_qkv_headdim(state_dict, f"backbone.layers.{i}.mixer.Wqkv.bias")
        shard_dim(state_dict, f"backbone.layers.{i}.mixer.out_proj.weight", -1)
        if rank != 0:
            state_dict.pop(f"backbone.layers.{i}.mixer.out_proj.bias")
        shard_dim(state_dict, f"backbone.layers.{i}.mlp.fc1.weight", 0)
        shard_dim(state_dict, f"backbone.layers.{i}.mlp.fc1.bias", 0)
        shard_dim(state_dict, f"backbone.layers.{i}.mlp.fc2.weight", -1)
        if rank != 0:
            state_dict.pop(f"backbone.layers.{i}.mlp.fc2.bias")
        if f"backbone.layers.{i}.mixer.kernel.kernel.B" in state_dict:
            for name in [
                "D",
                "ssm_k_D",
                "kernel.kernel.B",
                "kernel.kernel.inv_A_real",
                "kernel.kernel.A_imag",
                "ssm_k_kernel.kernel.B",
                "kernel.kernel.log_dt",
            ]:
                if f"backbone.layers.{i}.mixer.{name}" in state_dict:
                    shard_dim(state_dict, f"backbone.layers.{i}.mixer.{name}", 0)
            for name in ["kernel.kernel.C", "ssm_k_kernel.kernel.C"]:
                if f"backbone.layers.{i}.mixer.{name}" in state_dict:
                    shard_dim(state_dict, f"backbone.layers.{i}.mixer.{name}", 1)
    return state_dict