
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np


"""

Dataset for sampling arbitrary intervals from the human genome.

"""


# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def random_mask(seq, mask_token_id, mask_prob=0.15):
    rand = torch.rand(seq.shape)
    
    mask = rand < mask_prob
    
    masked_seq = seq.clone()
    masked_seq[mask] = mask_token_id
    
    return (masked_seq, mask)

def bert_mask(seq, mask_token_id, pad_token_id, vocab_size, mask_prob=0.15, random_token_prob=0.1, unchanged_token_prob=0.1, special_token_ids=None):
    """
    Applies BERT masking strategy to a sequence of BPE tokens.

    Args:
        seq: Input sequence of BPE tokens (shape: [batch_size, seq_length]).
        mask_token_id: ID of the [MASK] token.
        pad_token_id: ID of the padding token.
        vocab_size: Size of the vocabulary.
        mask_prob: Probability of masking a token.
        random_token_prob: Probability of replacing a masked token with a random token.
        unchanged_token_prob: Probability of keeping a masked token unchanged.
    
    Returns:
        A tuple containing:
            - masked_seq: The masked sequence.
            - labels: The ground truth labels for the masked positions.
            - mask: The mask used to identify which tokens were masked.
    """
    # 避免遮蔽padding
    mask = (seq != pad_token_id) & (torch.rand(seq.shape) < mask_prob)

    # 复制一份用于保存label
    labels = seq.clone()
    labels[~mask] = -100  # PyTorch忽略-100的label

    # 生成随机数矩阵
    rand = torch.rand(seq.shape)

    # 80% [MASK]
    indices_masked = mask & (rand < (1 - random_token_prob - unchanged_token_prob))
    seq[indices_masked] = mask_token_id

    # 10% random token
    indices_random = mask & (rand >= (1 - random_token_prob - unchanged_token_prob)) & (rand < (1 - unchanged_token_prob))
    # 生成随机token，排除特殊token
    random_tokens = torch.randint(0, vocab_size, seq.shape, dtype=torch.long)
    
    # 确保不会选到special tokens
    special_token_ids = torch.tensor(special_token_ids)
    while (torch.isin(random_tokens, special_token_ids)).any():
        random_tokens[torch.isin(random_tokens, special_token_ids)] = torch.randint(0, vocab_size, (random_tokens[torch.isin(random_tokens, special_token_ids)].shape[0],), dtype=torch.long)

    seq[indices_random] = random_tokens[indices_random]

    # 10% unchanged
    # 不需要修改seq，因为它已经保持不变

    return (seq, mask, labels)

class DNABERTSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # split 参数可能不再需要
        text_file,
        max_length,
        seq_name='seq_a',
        pad_max_length=None,
        tokenizer_name=None,
        add_eos=False,
        replace_N_token=False,
        pad_interval=False,
        use_tokenizer=True,
        tokenizer=None,
        return_augs=False,
        objective="stdmlm",
    ):
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval
        self.use_tokenizer = use_tokenizer
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.objective = objective

        # 读取文本文件
        assert seq_name in ['seq_a', 'seq_b'], 'seq_name must be either "seq_a" or "seq_b"'
        if split=="train":
            split="train_2m"
        elif split=="dev" or split=="test" or split=="val":
            split="val_48k"
        else:
            split="debug_train"
        text_path = Path(text_file+"/"+split+"_"+seq_name+".txt")
        assert text_path.exists(), 'path to text file must exist'
        self.text = []
        with open(text_path, 'r') as f:
            line = f.readline()
            while line:
                self.text.append(line.rstrip('\n'))
                line = f.readline()



    def __len__(self):
        return len(self.text)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        line = self.text[idx]

        if self.use_tokenizer:
            # 使用 tokenizer 处理文本
            tokens = self.tokenizer.encode(line, add_special_tokens=False)

            if self.add_eos:
                tokens.append(self.tokenizer.eos_token_id)

            # 处理长度超过 max_length 的情况
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # 填充到 pad_max_length
            if len(tokens) < self.pad_max_length:
                if self.pad_interval:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens))
                else:
                    tokens = [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens)) + tokens

            # convert to tensor
            seq = torch.LongTensor(tokens)

            if self.replace_N_token:
                # replace N token with a pad token, so we can ignore it in the loss
                seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

            data = seq.clone()  # remove eos
            target = seq.clone()  # offset by 1, includes eos
            special_token_ids = self.tokenizer.all_special_ids
            
            if self.objective=="stdmlm":
                return bert_mask(data, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.vocab_size, special_token_ids=special_token_ids), target
            else:
                return random_mask(data, self.tokenizer.mask_token_id), target
        else:
            # 不使用 tokenizer，直接返回文本行
            return {'text': line}
