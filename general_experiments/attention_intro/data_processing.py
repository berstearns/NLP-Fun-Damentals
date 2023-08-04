import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import evaluate

bleu_calc = evaluate.load('evaluate-metric/bleu')

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

def translate_evaluate(inputlang, outputlang, dataset: list, encoder:nn.Module, decoder:nn.Module) -> dict:
    EOS_TOKEN = 1
    MAXLEN = 10
    criterion = nn.NLLLoss() 
    inp_arr = list(map(lambda x: x[0], dataset))
    references = list(map(lambda x: x[1], dataset))
    all_preds = []
    n = len(inp_arr)  
    input_ids = np.zeros((n, MAXLEN), dtype=np.int32)
    output_ids = np.zeros((n, MAXLEN), dtype=np.int32)

    for idx, (in_sent, out_sent) in enumerate(dataset):
        in_sent = in_sent.split(' ')
        out_sent = out_sent.split(' ')
        in_enc_arr = [inputlang.word2idx.get(word, 2) for word in in_sent] + [EOS_TOKEN]
        out_enc_arr = [outputlang.word2idx.get(word, 2) for word in out_sent] + [EOS_TOKEN]
        input_ids[idx, :len(in_enc_arr)] = in_enc_arr
        output_ids[idx, :len(out_enc_arr)] = out_enc_arr

    test_dset = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(output_ids))
    dloader = DataLoader(test_dset, shuffle=False, batch_size=16)
    
    encoder.eval(), decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        dpoints = 0
        for batch in dloader:
            X, y = batch
            encoder_outputs, encoder_hidden = encoder(X)
            decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)
            pred_vals = decoder_outputs.argmax(dim=-1)
            loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            y.view(-1)
            )
            epoch_loss+=loss.item()
            pred_labs = [" ".join([outputlang.idx2word[word.item()] for word in dec_t if word not in [0,1]]) for dec_t in pred_vals]
            all_preds+=pred_labs
            dpoints+=X.shape[0]
    # return all_preds, references
    return bleu_calc.compute(predictions=all_preds, references=references), epoch_loss/dpoints

