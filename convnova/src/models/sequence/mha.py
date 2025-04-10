""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange
from typing import Optional
import math
from typing import List, Union
from torch.cuda.amp import autocast

mha_keywords = ["dropout", "bias", "add_bias_kv", "add_zero_attn", "kdim", "vdim", "batch_first", "device", "dtype"]
class BertEmbeddings(nn.Module):

    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings, layer_norm_eps=1e-5, hidden_dropout_prob=0.1, device=None, dtype=None, type_vocab_size=2):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id,
                                            **factory_kwargs)
        # ALiBi doesn't use position embeddings
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  hidden_size,
                                                  **factory_kwargs)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size,
                                      eps=layer_norm_eps,
                                      **factory_kwargs)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer('token_type_ids',
                             torch.zeros(max_position_embeddings,
                                         dtype=torch.long),
                             persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # great! ALiBi
            pass

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                # assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(input_shape,  # type: ignore
                                             dtype=torch.long,
                                             device=self.word_embeddings.device) # type: ignore  # yapf: disable

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
@TransposedModule
class MultiheadAttention(SequenceModule):
    """ Simple wrapper for MultiheadAttention """
    def __init__(self, d_model, n_heads, *args, causal=True,  **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        for key in list(kwargs.keys()):
            if key not in mha_keywords:
                setattr(self, key, kwargs[key])
                kwargs.pop(key)
                
        self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.causal = causal

        alibi_starting_size = 512
        self.num_attention_heads=n_heads
        self._current_alibi_size = int(alibi_starting_size)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.alibi = torch.zeros(
            (1, self.num_attention_heads, self._current_alibi_size,
             self._current_alibi_size))
        self.rebuild_alibi_tensor(size=alibi_starting_size, device=device)

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
        slopes = torch.tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """ state should represent a mask and key padding mask """
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
            # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            def _fill_with_neg_inf(t):
                """FP16-compatible function that fills a tensor with -inf."""
                return t.float().fill_(float("-inf")).type_as(t)
            attention_mask = torch.triu(
                    _fill_with_neg_inf(torch.zeros([src.size(-2), src.size(-2)], dtype=torch.float32)), 1
                )
            extended_attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.bfloat16, device=src.device) 
            assert not torch.isnan(extended_attention_mask).any()
            alibi_bias = self.alibi
            assert not torch.isnan(alibi_bias).any()
            attn_bias = extended_attention_mask[:, :, :src.shape[1], :src.shape[1]]
            with autocast(dtype=torch.float32):
                alibi_attn_mask = attn_bias + alibi_bias.to(attn_bias.device)
                assert not torch.isnan(alibi_attn_mask).any()
            assert not torch.isnan(alibi_attn_mask).any()
            attn_mask = (alibi_attn_mask.reshape(-1, 512, 512)).repeat(src.shape[0], 1, 1)
        assert not torch.isnan(attn_mask).any()

        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, _ = self.mha(src.clone(), src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        if self.return_state:
            return y, None
        return y

    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        pass


class VitAttention(SequenceModule):
    """Copied from implementation for ViT: only used for ViT model

    This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.
    """

    @property
    def d_output(self):
        return self.dim

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        # proj_drop=0.,
        packed_linear=True,
        linear_cfg=None,
        **kwargs,
    ):
        """packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights, where there's only one
        projection weight matrix.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            if linear_cfg is None:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # Removing this dropout because we do this in SequenceResidualBlock
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, state=None):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
        attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                         '(b h) t s -> b h t s', h = self.num_heads)

        attn = F.softmax(attn, dim=-1, dtype=v.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, None
