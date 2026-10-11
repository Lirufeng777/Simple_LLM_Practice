import math
import torch
import torch.nn as nn
import torch.functional as F

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, text):
        tokens = text.split()
        return tokens

    def ecode(self, text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        else:
            ids.append(self.vocab["<UNK>"])
        return ids
    






