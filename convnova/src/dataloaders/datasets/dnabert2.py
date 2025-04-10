from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
from mmap import mmap, ACCESS_READ
from transformers import AutoTokenizer
import tracemalloc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from torch.utils.data import DataLoader
import gc
# from src.dataloaders.utils.comm import TorchShmSerializedList
from torch.utils.data import get_worker_info
import torch.distributed as dist
root = os.getenv('PROJECT_ROOT')
# Helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# Augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        else:
            rev_comp += base
    return rev_comp

def random_mask(seq, mask_token_id, mask_prob=0.15):
    rand = torch.rand(seq.shape)
    mask = rand < mask_prob
    masked_seq = seq.clone()
    masked_seq[mask] = mask_token_id
    return masked_seq, mask

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
    rand_mask = torch.rand(seq.shape) < mask_prob
    mask = (seq != pad_token_id) & (rand_mask)

    labels = seq.clone()
    labels[~mask] = -100

    # 生成随机数矩阵
    rand = torch.rand(seq.shape)

    # 80% [MASK]
    indices_masked = mask & (rand < (1 - random_token_prob - unchanged_token_prob))
    seq[indices_masked] = mask_token_id
    assert (seq<5).all()

    # 10% random token
    indices_random = mask & (rand >= (1 - random_token_prob - unchanged_token_prob)) & (rand < (1 - unchanged_token_prob))
    # 生成随机token，排除特殊token
    random_tokens = torch.randint(0, vocab_size, seq.shape, dtype=torch.long)
    assert (random_tokens<5).all()
    
    # 确保不会选到special tokens
    special_token_ids = torch.tensor(special_token_ids)
    while (torch.isin(random_tokens, special_token_ids)).any():
        random_tokens[torch.isin(random_tokens, special_token_ids)] = torch.randint(0, vocab_size, (random_tokens[torch.isin(random_tokens, special_token_ids)].shape[0],), dtype=torch.long)

    seq[indices_random] = random_tokens[indices_random]
    assert (seq<5).all()

    assert (mask == (seq != pad_token_id) & (rand_mask)).all()
    assert (seq>=0).all()
    assert (seq<5).all()

    # 10% unchanged
    # 不需要修改seq，因为它已经保持不变

    return (seq, mask, labels)

class DNABERT2Dataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        split,
        max_length,
        text_file=None,
        pad_max_length=None,
        tokenizer_name=None,
        add_eos=False,
        replace_N_token=False,
        pad_interval=False,
        use_tokenizer=True,
        tokenizer=None,
        return_augs=False,
        objective="stdmlm",
        num_worker=4,
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

        if text_file is None:
            text_file = root+"/data/dnabert2"

        self.split = split if split != "val" and split != "test" else "dev"
        self.text_path = Path(f"{text_file}/{self.split}.txt")

        with open(self.text_path, 'r') as file:
            self.length = sum(1 for _ in file)
        self.start, self.end = None, None

    # def __len__(self):
    #     return self.length

    def __iter__(self):
        if self.start is None:
            worker_info = get_worker_info()

            # 检查是否已经初始化分布式
            if dist.is_initialized():
                # 获取当前 GPU 的 rank 和总的 world size (GPU 数量)
                local_rank = dist.get_rank()  # 当前 GPU 的 rank
                world_size = dist.get_world_size()  # 总 GPU 数
            else:
                # 如果没有初始化分布式，假设为单 GPU
                local_rank = 0
                world_size = 1

            if worker_info is None:
                # 单进程加载的情况
                self.start = 0
                self.end = self.length
            else:
                # 多进程、多GPU加载的情况

                # 获取 worker 信息
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

                # 全局每个 GPU 上处理的样本总数（等分给 world_size 个 GPU）
                per_gpu = self.length // world_size
                gpu_remainder = self.length % world_size

                # 每个 GPU 上每个 worker 的样本数
                per_worker = per_gpu // num_workers
                worker_remainder = per_gpu % num_workers

                # 分配给每个 GPU 的起始和结束索引
                if local_rank < gpu_remainder:
                    gpu_start = local_rank * (per_gpu + 1)
                    gpu_end = gpu_start + per_gpu + 1
                else:
                    gpu_start = local_rank * per_gpu + gpu_remainder
                    gpu_end = gpu_start + per_gpu

                # 在 GPU 内部分配给 workers 的起始和结束索引
                if worker_id < worker_remainder:
                    self.start = gpu_start + worker_id * (per_worker + 1)
                    self.end = self.start + per_worker + 1
                else:
                    self.start = gpu_start + worker_id * per_worker + worker_remainder
                    self.end = self.start + per_worker
        with open(self.text_path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip("\n")
                if idx < self.start:  # Skip lines until we reach the start for this worker
                    continue
                if idx >= self.end:  # Stop once we've processed enough lines
                    break
                tokens = self.tokenizer.encode(line, add_special_tokens=False, truncation=True, max_length=self.max_length)
                if self.add_eos:
                    tokens.append(self.tokenizer.eos_token_id)
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                if len(tokens) < self.pad_max_length:
                    if self.pad_interval:
                        tokens += [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens))
                    else:
                        tokens = [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokens)) + tokens
                seq = torch.LongTensor(tokens)
                if self.replace_N_token:
                    seq[seq == self.tokenizer._vocab_str_to_int['N']] = self.tokenizer.pad_token_id
                data = seq
                target = seq.clone()
                special_token_ids = self.tokenizer.all_special_ids
                if not self.use_tokenizer:
                    seq = seq - 7
                    mask = (seq >= 4) | (seq < 0)
                    seq[mask] = 4
                    assert (seq>=0).all()
                    assert (seq<5).all()
                    data = seq
                    target = seq.clone()

                    seq, mask, labels = bert_mask(data, 4, 5, 5, special_token_ids=[4])
                    yield bert_mask(data, 4, 5, 5, special_token_ids=[4]), target
                else:
                    if self.objective == "stdmlm":
                        yield bert_mask(data, self.tokenizer.mask_token_id, self.tokenizer.pad_token_id, self.tokenizer.vocab_size, special_token_ids=special_token_ids), target
                    else:
                        yield random_mask(data, self.tokenizer.mask_token_id), target


import os
import tracemalloc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Assuming DNABERT2Dataset is already defined and imported
# from your_dataset_module import DNABERT2Dataset

class DNABERT2DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, text_file, max_length, pad_max_length, batch_size, num_workers):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_file = text_file
        self.max_length = max_length
        self.pad_max_length = pad_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = DNABERT2Dataset(
            split="train",
            text_file=self.text_file,
            max_length=self.max_length,
            pad_max_length=self.pad_max_length,
            tokenizer_name="bpe",
            add_eos=False,
            replace_N_token=False,
            pad_interval=False,
            use_tokenizer=True,
            tokenizer=self.tokenizer,
            return_augs=False,
            objective="stdmlm"
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False
        )

class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(128, 128)

    def training_step(self, batch, batch_idx):
        # Simple dummy implementation to satisfy Lightning requirements
        return None

    def configure_optimizers(self):
        # Dummy implementation
        return None

class MemoryProfileCallback(pl.Callback):
    def __init__(self):
        self.tracemalloc_started = False

    def on_train_start(self, trainer, pl_module):
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True

    def on_train_epoch_end(self, trainer, pl_module):
        current, peak = tracemalloc.get_traced_memory()
        print(f"Epoch {trainer.current_epoch}: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    def on_train_end(self, trainer, pl_module):
        tracemalloc.stop()


import psutil
import functools

mem_0 = psutil.virtual_memory().available / (1024 * 1024)  # in MB


def profile_memory(func, min_threshold=0.1):
    """Decorator to profile CPU memory usage before and after a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Record the memory usage before the function call
        memory_info = psutil.virtual_memory()
        mem_before = memory_info.available / (1024 * 1024)  # in MB

        module_name = func.__module__ if hasattr(func, '__module__') else 'unknown module'
        
        # Call the actual function
        result = func(*args, **kwargs)

        # Record the memory usage after the function call
        memory_info = psutil.virtual_memory()
        mem_after = memory_info.available / (1024 * 1024)  # in MB
        mem_change = mem_before - mem_after

        if abs(mem_change) > min_threshold:
            print(f'[FUNCTION] {func.__name__} from {module_name}: {func.__code__.co_argcount} args, {func.__code__.co_posonlyargcount} pos-only args, {func.__code__.co_kwonlyargcount} keyword-only args')
            # print(f'[ARGUMENTS] {args}, {kwargs}')
            print(f"[MEMORY] Before calling {func.__name__} from {module_name}: {mem_0 - mem_before:.2f} MB")
            print(f"[MEMORY] After calling {func.__name__} from {module_name}: {mem_0 - mem_after:.2f} MB")
            print(f"[MEMORY] Memory change during {func.__name__} from {module_name}: {mem_change:.2f} MB\n")
            gc.collect()

        return result

    return wrapper

import types

# Function to patch all methods of a class with a given decorator
def patch_class_with_memory_profiler(cls, decorator, min_threshold=0.1, exclude_fn=None):
    if exclude_fn is None:
        exclude_fn = ['state_dict', 'profile_hook_step', '_to_hparams_dict', 'fit']
    # Iterate through all attributes of the class
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        # Check if the attribute is a method (not a dunder method or property)
        if isinstance(attr, types.FunctionType) and not (attr_name.startswith("__") and attr_name.endswith("__")) and not any(fn in attr_name for fn in exclude_fn):
            # Check if the method is already decorated
            if not hasattr(attr, "_is_decorated"):
                # Apply the decorator
                decorated_attr = decorator(attr, min_threshold=min_threshold)
                # Mark the method as decorated
                decorated_attr._is_decorated = True
                setattr(cls, attr_name, decorated_attr)

# Patch the Trainer class with the memory profiling decorator
patch_class_with_memory_profiler(DNABERT2DataModule, profile_memory, min_threshold=10)

if __name__ == "__main__":
    root = os.getenv('PROJECT_ROOT', '/gpfs/gibbs/pi/gerstein/xt86/by/hyena-dna')

    tokenizer = AutoTokenizer.from_pretrained(f'{root}/DNABERT-2-117M')
    data_module = DNABERT2DataModule(
        tokenizer=tokenizer,
        text_file="/gpfs/gibbs/pi/gerstein/xt86/by/hyena-dna/data/dnabert2",
        max_length=512,
        pad_max_length=512,
        batch_size=4000,
        num_workers=4
    )

    model = DummyModel()

    trainer = pl.Trainer(
        max_epochs=2,  # Adjust as needed
        callbacks=[MemoryProfileCallback()],
        gpus=1  # Adjust according to your hardware setup
    )

    trainer.fit(model, datamodule=data_module)