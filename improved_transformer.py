import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 单头注意力层
# 定义名为SingleHeadSelfAttention的类，并继承nn.Module
class SingleHeadSelfAttention(nn.Module):

    # __init__：创建并保存层需要长期使用的东西，例如 nn.Linear、维度等。创建对象时运行一次。
    # forward：定义每次输入数据后要执行的计算。调用 attention(H) 时运行。
    
    # 初始化方法，self是当前的注意力层对象，传入模型的维度d_model
    def __init__(self, d_model):
        # 调用父类 nn.Module 的初始化方法，常在定义自己的 nn.Module 时都需要调用它
        super().__init__()

        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    # 关键字def，forward 是这个注意力层的前向计算方法，定义输入 H 如何变成输出。
    def forward(self, H):
        
        Q = self.W_q(H)
        K = self.W_k(H)
        V = self.W_v(H)

        score = Q @ K.T / math.sqrt(self.d_model)

        # H的张量形状为序列长度，特征维度,所以是shape[0]
        seq_len = H.shape[0]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=score.device), #score和H的形状相同写哪个都行
            diagonal = 1
        ).bool()

        score = score.masked_fill(mask, float('-inf'))

        attention_weight = F.softmax(score, dim=-1)
        A = attention_weight @ V

        return A

# Attention后的输出，残差连接，Layernorm归一化，mlp，残差连接，Layernorm，输出

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        self.attention = SingleHeadSelfAttention(d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, H):
        
        A = self.attention(H)
        R1 = A + H
        Z = self.norm1(R1)
        F = self.mlp(Z)
        R2 = F + Z
        H_next = self.norm2(R2)

        return H_next


class SimpleTransformer(nn.Module):
    def __init__(
            self,
            input_dim = 4,
            d_model = 8,
            d_ff = 32,
            num_layers = 3
            ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                d_ff
            )
            for _ in range(num_layers)
        )


    def forward(self, X):
        H_in = self.embedding(X)
        H = H_in

        for layer in self.layers:
            H = layer(H)

        H_out = H

        return H_in, H_out


X = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
    [4.0, 5.0, 6.0, 7.0],
    [5.0, 6.0, 7.0, 8.0],
    [6.0, 7.0, 8.0, 9.0],
    [7.0, 8.0, 9.0, 10.0],
    [8.0, 9.0, 10.0, 11.0],
    [9.0, 10.0, 11.0, 12.0],
    [10.0, 11.0, 12.0, 13.0]
])


model =  SimpleTransformer(
    input_dim=4,
    d_model=8,
    d_ff=32,
    num_layers=3
)

H_in, H_out = model(X)

print("X:")
print(X)

print("\nX shape:")
print(X.shape)

print("\nH_in:")
print(H_in)

print("\nH_in shape:")
print(H_in.shape)

print("\nH_out:")
print(H_out)

print("\nH_out shape:")
print(H_out.shape)