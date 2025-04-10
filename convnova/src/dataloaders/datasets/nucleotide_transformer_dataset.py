from pyfaidx import Fasta
import torch
# from random import random 
import random
from pathlib import Path

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

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



class NucleotideTransformerDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        use_tokenizer=True 
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        self.use_tokenizer = use_tokenizer

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name 
        assert base_path.exists(), 'path to fasta file must exist'

        for file in (base_path.iterdir()):
            if str(file).endswith('.fasta') and split in str(file):
                self.seqs = Fasta(str(file), read_long_names=True)    

        self.label_mapper = {}
        for i, key in enumerate(self.seqs.keys()):
            self.label_mapper[i] = (key, int(key.rstrip()[-1]))


    def __len__(self):
        return len(self.seqs.keys())

    def __getitem__(self, idx):
        seq_id = self.label_mapper[idx][0]
        x = self.seqs[seq_id][:].seq # only one sequence
        y = self.label_mapper[idx][1] # 0 or 1 for binary classification

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        assert type(seq_ids)==list
        seq_ids = torch.LongTensor(seq_ids)
        if not self.use_tokenizer:
            seq_ids = seq_ids-7
            mask = (seq_ids >= 4) | (seq_ids < 0)
            seq_ids[mask] = 4

        # convert to tensor
        # seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # need to wrap in list
        target = torch.LongTensor([y])  # offset by 1, includes eos

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
# class NucleotideTransformerDataset(torch.utils.data.Dataset):
#     '''
#     Dataset for loading DNA sequences and labels for binary classification.
#     Allows noise injection on positive sequences for testing.
#     '''

#     def __init__(
#         self,
#         split,
#         max_length,
#         dataset_name=None,
#         d_output=2,  # default binary classification
#         dest_path=None,
#         tokenizer=None,
#         tokenizer_name=None,
#         use_padding=None,
#         add_eos=False,
#         rc_aug=False,
#         return_augs=False,
#         return_mask=False,
#         use_tokenizer=True,
#         # noise_type="shuffle",  # Type of noise to inject, e.g., "shuffle", "random_substitution"
#         noise_type=None,
#         noise_prob=0.01,  # Probability of introducing noise in the sequence,
#     ):
#         self.max_length = max_length
#         self.use_padding = use_padding
#         self.tokenizer_name = tokenizer_name
#         self.tokenizer = tokenizer
#         self.return_augs = return_augs
#         self.add_eos = add_eos
#         self.d_output = d_output
#         self.rc_aug = rc_aug
#         self.return_mask = return_mask
#         self.use_tokenizer = use_tokenizer
#         self.noise_type = noise_type
#         self.noise_prob = noise_prob


#         if split == "val":
#             split = "test"

#         # Handling path for dataset
#         base_path = Path(dest_path) / dataset_name
#         assert base_path.exists(), 'path to fasta file must exist'

#         for file in (base_path.iterdir()):
#             if str(file).endswith('.fasta') and split in str(file):
#                 self.seqs = Fasta(str(file), read_long_names=True)

#         # Create label mapper (assumes binary classification with 0 and 1 labels)
#         self.label_mapper = {}
#         for i, key in enumerate(self.seqs.keys()):
#             self.label_mapper[i] = (key, int(key.rstrip()[-1]))

#     def __len__(self):
#         return len(self.seqs.keys())

#     def inject_noise(self, sequence):
#         ''' Injects noise into a DNA sequence based on the chosen noise type '''
#         if self.noise_type == "shuffle":
#             # 计算需要打乱的核苷酸数量
#             sequence_length = len(sequence)
#             num_to_shuffle = int(sequence_length * self.noise_prob)  # 根据噪声概率计算核苷酸数量

#             # 随机选择要打乱的位置
#             indices_to_shuffle = random.sample(range(sequence_length), num_to_shuffle)

#             # 提取选中的核苷酸
#             sequence_part = [sequence[i] for i in indices_to_shuffle]

#             # 打乱选中的核苷酸
#             random.shuffle(sequence_part)

#             # 创建一个列表以构建新的序列
#             new_sequence = list(sequence)
            
#             # 将打乱后的核苷酸放回原序列中的相应位置
#             for idx, new_base in zip(sorted(indices_to_shuffle), sequence_part):
#                 new_sequence[idx] = new_base

#             sequence = ''.join(new_sequence)
#             return sequence
#         elif self.noise_type == "random_substitution":
#             # Randomly substitute bases in the sequence with some probability
#             bases = ['A', 'C', 'G', 'T']
#             noisy_sequence = []
#             for base in sequence:
#                 if random.random() < self.noise_prob:
#                     noisy_base = random.choice([b for b in bases if b != base])
#                     noisy_sequence.append(noisy_base)
#                 else:
#                     noisy_sequence.append(base)
#             return ''.join(noisy_sequence)
#         else:
#             return sequence  # Return the sequence unchanged if no noise type is specified

#     def __getitem__(self, idx):
#         seq_id = self.label_mapper[idx][0]
#         x = self.seqs[seq_id][:].seq  # Get DNA sequence
#         y = self.label_mapper[idx][1]  # Binary label: 0 or 1

#         # Apply reverse complement augmentation if enabled
#         if self.rc_aug and coin_flip():
#             x = string_reverse_complement(x)

#         # Inject noise only for positive sequences (label == 1)
#         if y == 1 and self.noise_type:
#             x = self.inject_noise(x)

#         # Tokenize the sequence
#         seq = self.tokenizer(
#             x,
#             add_special_tokens=True if self.add_eos else False,
#             padding="max_length" if self.use_padding else 'do_not_pad',
#             max_length=self.max_length,
#             truncation=True,
#         )

#         seq_ids = seq["input_ids"]
#         assert isinstance(seq_ids, list)
#         seq_ids = torch.LongTensor(seq_ids)

#         # Optionally handle cases without tokenizer
#         if not self.use_tokenizer:
#             seq_ids = seq_ids - 7
#             mask = (seq_ids >= 4) | (seq_ids < 0)
#             seq_ids[mask] = 4

#         # Convert label to tensor
#         target = torch.LongTensor([y])

#         # Return with attention mask if specified
#         if self.return_mask:
#             return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
#         else:
#             return seq_ids, target
