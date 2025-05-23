import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=4, ln=False, dropout=0.1):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.ln = ln

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / sqrt(self.dim_V // self.num_heads), 2)
        A = self.dropout(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.fc_o(O)
        if self.ln:
            O = self.ln0(O)
            O = O + Q
            O = self.ln1(O)
        else:
            O = O + Q
        return O


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)
        self.o = nn.Linear(dim_v, dim_in)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
        bs = x.size(0)
        dim_k_per_head = self.dim_k // self.num_heads
        dim_v_per_head = self.dim_v // self.num_heads
        q = self.q(x).view(bs, -1, self.num_heads, dim_k_per_head).transpose(1, 2)
        k = self.k(y).view(bs, -1, self.num_heads, dim_k_per_head).transpose(1, 2)
        v = self.v(y).view(bs, -1, self.num_heads, dim_v_per_head).transpose(1, 2)
        attn = q.matmul(k.transpose(2, 3)) * self.norm_fact
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = attn.matmul(v).transpose(1, 2).contiguous()
        output = output.view(bs, -1, self.dim_v)
        output = self.o(output)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=8, in_chans=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, 64, kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dim_linear_block=256, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, dim_head * heads, dim_head * heads, num_heads=heads, dropout=dropout)
        self.feed_forward = FeedForward(dim, dim_linear_block, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=2, heads=4, dim_head=64, dim_linear_block=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderLayer(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dim_linear_block=dim_linear_block,
                dropout=dropout
            ))

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class GRUTransformerHybrid(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_heads=4, dropout=0.1):
        super(GRUTransformerHybrid, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size=8, in_chans=input_dim, embed_dim=hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dim_linear_block=hidden_dim * 4,
            dropout=dropout
        )
        self.mab = MAB(hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        gru_out, _ = self.gru(x)
        transformer_out = self.transformer_encoder(gru_out)
        mab_out = self.mab(transformer_out, transformer_out)
        output = self.fc(mab_out)
        return output
