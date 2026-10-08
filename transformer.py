import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# 随机输入的写法
# torch.manual_seed(0) #设置随机种子，保证每次运行结果相同
# X = torch.rand(10,4)
# print(X)
# print(X.shape)

# 1.1 输入向量x1-x10
# 固定输入，方便调试与观察
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
print(X)
print(X.shape)


# 1.2 x1-x10转e1-e10
# 输入向量转为高维嵌入向量
# 此处应该有多种映射方式，以下是一种简单的线性映射方式H_in=xW+b
embedding = nn.Linear(4, 8)  # 将输入维度从4映射到8
H_in = embedding(X)  # 线性变换10*4 -> 10*8
print(H_in)
print(H_in.shape)  


# 1.3 Q,K,V矩阵
# qt = h_t * W_q, Qt = H_in * W_q,其他同理
W_q1 = nn.Linear(8, 8) 
W_k1 = nn.Linear(8, 8)
W_v1 = nn.Linear(8, 8)

Q1 = W_q1(H_in)  #10*8
K1 = W_k1(H_in)
V1 = W_v1(H_in)

# 1.4 计算注意力分数
# score = QK^T/c
c = math.sqrt(8)  # 缩放因子，一般是sqrt(d_h)，这里d_h=8，若为看清可写1
score1 = Q1 @ K1.T / c
print(score1)
print(score1.shape)  #10*10

# 补充：加入mask机制，防止模型看到未来信息
# 创建一个上三角为1的矩阵，表示未来信息，mask为True的地方会被忽略
mask1 = torch.triu(
    torch.ones(10, 10, device=score1.device), 
    diagonal=1
).bool()
# 将负无穷加到1的位置
score1 = score1.masked_fill(mask1, float('-inf'))


# 1.5 对注意力分数进行Softmax
attention_weights1 = F.softmax(score1, dim=-1)  # 对每一行进行softmax,我们的输入是行向量
A1 = attention_weights1 @ V1 # 10*10 @ 10*8 -> 10*8

# 1.6 第一层第一次残差连接
R1_1 = H_in + A1  # 10*8 + 10*8 -> 10*8

# 1.7 第一层第一次LayerNorm
norm1_1 = nn.LayerNorm(8)  # 这里对每一行进行归一化，归一化的维度是8
Z1 = norm1_1(R1_1)  # 10*8 -> 10*8

# 1.8 第一层MLP
mlp1 = nn.Sequential(
    nn.Linear(8, 32),   #10*8 @ 8*32 -> 10*32
    nn.GELU(),
    nn.Linear(32, 8)    #10*32 @ 32*8 -> 10*8
)
F1 = mlp1(Z1)

# 1.9 第一层第二次残差后第一层输出H1
R1_2 = Z1 + F1
norm1_2 = nn.LayerNorm(8)
H1 = norm1_2(R1_2)
print(H1)
print(H1.shape)  

# 2.1 第二层Q，K，V矩阵
W_q2 = nn.Linear(8, 8)
W_k2 = nn.Linear(8, 8)
W_v2 = nn.Linear(8, 8)

Q2 = W_q2(H1)
K2 = W_k2(H1)
V2 = W_v2(H1)

# 2.2 第二层注意力分数
score2 = Q2 @ K2.T / c  #10*10

# mask
mask2 = torch.triu(
    torch.ones(10, 10, device=score2.device),
    diagonal=1
).bool()
score2 = score2.masked_fill(mask2, float('-inf'))


# 2.3 softmax
attention_weights2 = F.softmax(score2, dim=-1)
A2 = attention_weights2 @ V2

# 2.4 第二层第一次残差
R2_1 = A2 + H1   #10*8

# 2.5 第二层第一次归一化
norm2_1 = nn.LayerNorm(8)
Z2 = norm2_1(R2_1)  #10*8

# 2.6 mlp2
mlp2 = nn.Sequential(
    nn.Linear(8, 32),
    nn.GELU(),
    nn.Linear(32,8)
)
F2 = mlp2(Z2)   #10*8

# 2.7 第二层残差后归一得到H2
R2_2 = F2 +Z2
norm2_2 = nn.LayerNorm(8)
H2 = norm2_2(R2_2)
print(H2)
print(H2.shape)

# 3.1 第三层Q，K，V矩阵
W_q3 = nn.Linear(8, 8)
W_k3 = nn.Linear(8, 8)
W_v3 = nn.Linear(8, 8)

Q3 = W_q3(H2)
K3 = W_k3(H2)
V3 = W_v3(H2)

# 3.2 第三层注意力分数
score3 = Q3 @ K3.T / c

# mask
mask3 = torch.triu(
    torch.ones(10, 10, device=score3.device),
    diagonal=1
).bool()
score3 = score3.masked_fill(mask3, float('-inf'))

# 3.3 softmax
attention_weights3 = F.softmax(score3, dim=-1)
A3 = attention_weights3 @ V3

# 3.4 第三层第一次残差
R3_1 = A3 + H2

# 3.5 第三层第一次归一
norm3_1 = nn.LayerNorm(8)
Z3 = norm3_1(R3_1)

# 3.6 mlp3
mlp3 = nn.Sequential(
    nn.Linear(8, 32),
    nn.GELU(),
    nn.Linear(32, 8)
)
F3 = mlp3(Z3)

# 3.7 第三层第二次残差归一输出H_out
R3_2 = F3 + Z3
norm3_2 = nn.LayerNorm(8)
H_out = norm3_2(R3_2)
print(H_out)
print(H_out.shape)
