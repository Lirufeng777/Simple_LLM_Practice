import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 原始文本
text = "i love cjy"

# 2. 建立词表
vocab = {
    "<PAD>": 0,  # 空格token
    "<UNK>": 1,  # UNK是unknown的意思
    "i": 2,
    "love": 3,
    "cjy": 4
}

# 3. 分词
tokens = text.split()
print(tokens)

# 4. token -> token id
ids = []

# for token in tokens:
#     if token in tokens:
#         ids.append(vocab[token])
#     else:
#         ids.append(vocab["<UNK>"])

for token in tokens:
    if token in vocab:
        token_id = vocab[token]
    else:
        token_id = vocab["<UNK>"]

    ids.append(token_id)

print(ids)

# 5. 固定序列长度为10
max_length = 10

# 如果超过10个token就截断
if len(ids) > max_length:
    ids = ids[:max_length]

# 如果不足10个token就补上PAD
while len(ids) < max_length:
    ids.append(vocab["<PAD>"])

print("after padding:")
print(ids)

# 6. 转成 Tensor
input_ids = torch.tensor(ids, dtype=torch.long)


print("input_ids:")
print(input_ids)

print("input_ids shape:")
print(input_ids.shape)

# 7. 创建Embedding
vocab_size = len(vocab)
d_model = 4

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model
)

# 8. token id -> 4维向量
x = embedding(input_ids)

print("embedding output:")
print(x)

print("embedding shape:")
print(x.shape)

print(embedding.weight)