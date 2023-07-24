import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
DATASET_PATH = "./dataset/eng-fra.txt"


# Unicode-TO-Ascii with normalization
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip().lower()


"""
class for language vocab, mappings and num_words


"""
class Lang:
    def __init__(self, name) -> None:
        self.name = name
        self.word_count = {}
        self.word2idx = {
            "SOS" : 0,
            "EOS" : 1,
            "UNK" : 2
        }
        
        self.idx2word = {
            0: "SOS",
            1: "EOS",
            3: "UNK"
        }
        self.n_words = 3

    def add_word(self, word):
        if word in self.word_count.keys():
            self.word_count[word] += 1
        else:
            self.word2idx[word] = self.n_words
            self.word_count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words+=1
    
    def add_sentence(self, sentence):
        words = sentence.split(' ')
        for word in words:
            self.add_word(word)


def readLangs(ds_path, lang_1, lang_2):
    lines = open(ds_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(lx) for lx in line.split('\t')][::-1] for line in lines]
    return Lang(lang_1), Lang(lang_2), pairs

def filter_pair(p, maxlen):
    eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
    )
    return len(p[0].split(' ')) < maxlen and len(p[1].split(' ')) < maxlen and p[1].startswith(eng_prefixes)

def prepareData(path, maxlen=10, lang1="french", lang2="english"):
    l_fre, l_eng, pairs  = readLangs(path, lang1, lang2)
    pairs = [pair for pair in pairs if filter_pair(pair, maxlen)]
    for fx, ex in pairs:
        l_eng.add_sentence(ex)        
        l_fre.add_sentence(fx)
    return l_fre, l_eng, pairs