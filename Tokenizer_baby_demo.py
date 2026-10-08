import torch
import torch.nn as nn


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

tokens = text.split()
ids = []

for token in tokens:
    if token in vocab:
        ids.append(vocab[token])
    else:
        ids.append(vocab["<UNK>"])

print(tokens)
print(ids)

input_ids = torch.tensor(ids[:-1], dtype=torch.long)
target_ids = torch.tensor(ids[1:], dtype=torch.long)

print(input_ids)
print(target_ids)

vocab_size = len(vocab)

token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=4
)

X = token_embedding(input_ids)


##--------------Tokenization---------------##


##--------------Transformer---------------##
from improved_transformer import SimpleTransformer

model = SimpleTransformer(
    input_dim=4,
    d_model=8,
    d_ff=32,
    num_layers=3
)

H_in, H_out = model(X)

print("X shape:", X.shape)
print("H_in shape:", H_in.shape)
print("H_out shape:", H_out.shape)

##--------------Transformer---------------##


##----------------LM Head-----------------##

# 这里的[8, 10]的意思是有8个输入位置，每一个位置有10个token的预测分数
lm_head = nn.Linear(8, vocab_size)
logits = lm_head(H_out)
print("logits shape:", logits.shape)

##----------------LM Head-----------------##



##----------------Loss-----------------##
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target_ids)
print("loss:", loss.item())

##----------------Loss-----------------##




##----------------Train-----------------##
optimizer = torch.optim.Adam(
    list(token_embedding.parameters())
    +
    list(model.parameters())
    +
    list(lm_head.parameters()),
    lr=0.01
)

for epoch in range(1000):
    X = token_embedding(input_ids)
    H_in, H_out = model(X)
    logits = lm_head(H_out)

    loss = criterion(logits, target_ids)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    if epoch %100 == 0:
        print(epoch, loss.item()) 

##----------------Train-----------------##



##----------------生成部分-----------------##

# id -> token
id_to_token = {}

for token, token_id in vocab.items():
    id_to_token[token_id] = token


# 用户输入
user_input = input("\nYou love who?\n> ")

input_tokens = user_input.lower().split()

generated_ids = []

for token in input_tokens:

    if token in vocab:
        generated_ids.append(vocab[token])

    else:
        generated_ids.append(vocab["<UNK>"])


# 保存原始输入长度
prompt_length = len(generated_ids)


# 生成6个新的token
for _ in range(6):

    current_ids = torch.tensor(
        generated_ids,
        dtype=torch.long
    )

    X = token_embedding(current_ids)

    H_in, H_out = model(X)

    logits = lm_head(H_out)

    last_logits = logits[-1]

    next_id = torch.argmax(
        last_logits
    ).item()

    generated_ids.append(next_id)


# 只取模型新生成的部分
answer_ids = generated_ids[prompt_length:]


answer_tokens = []

for token_id in answer_ids:

    answer_tokens.append(
        id_to_token[token_id]
    )


answer = " ".join(answer_tokens)

print("\nWhy you love who?")
print(">", answer)


##----------------生成部分-----------------##
