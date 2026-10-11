import math
import torch
import torch.nn as nn
import torch.functional as F

from improved_Tokenizer import Tokenizer
from improved_transformer import SimpleTransformer

##--------------Tokenization---------------##
text = "i love cjy because cjy is my only baby"

vocab = {
    "<PAD>": 0,
    "<UNK>": 1,
    "i": 2,
    "love": 3,
    "cjy": 4,
    "because": 5,
    "is": 6,
    "my": 7,
    "only": 8,
    "baby": 9
}

ids = Tokenizer(vocab).ecode(text)
input_ids = torch.tensor(ids[:-1], dtype=torch.long)
target_ids = torch.tensor(ids[1:], dtype=torch.long)

vocab_size = len(vocab)

token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=4  # Transformer 输入4维
)

X = token_embedding(input_ids)

##--------------Tokenization---------------##

model = SimpleTransformer(
    input_dim=4,
    d_model=8,
    d_ff=32,
    num_layers=3
)

H_in, H_out = model(X)

##--------------Transformer---------------##



