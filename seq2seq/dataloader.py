import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define the tokenizers
token_transform = {}
vocab_transform = {}
token_transform['de'] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform['en'] = get_tokenizer('spacy', language='en_core_web_sm')


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


# Load the dataset
train_dataset, val_dataset, test_dataset = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))


def yield_tokens(data_iter, language: str):
    language_index = {'de': 0, 'en': 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


for ln in ['de', 'en']:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in ['de', 'en']:
  vocab_transform[ln].set_default_index(UNK_IDX)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in ['de', 'en']:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform['de'](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform['en'](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
