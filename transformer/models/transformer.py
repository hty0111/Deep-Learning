import torch
import torch.nn as nn
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=1024):
        super().__init__()

        self.dropout = nn.Dropout(p=p_drop)

        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

        # 10000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.Tensor([10000])) / d_model)) # [d_model / 2]
        
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model})
        # PE(pos, 2i+1) = sin(pos/10000^{2i/d_model})
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

        # [max_len, d_model] -> [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(1)

        # register 'pe' to buffer and require no grads
        self.register_buffer('pe', positional_encoding)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), ...]
        
        return self.dropout(x)
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, p_drop=0.1):
        super().__init__()

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p_drop)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)

        return self.layer_norm(residual + x)
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, attn_mask=None):
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_k, d_v]
        # attn_attn_mask: [batch_size, n_heads, seq_len, seq_len]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k) # [batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.attn_masked_fill(attn_mask, 1e-9)
        
        # uniform attention scores in key dim
        attn = nn.Softmax(dim=-1)(scores) # [batch_size, n_heads, len_q, len_k]
        prob = torch.matmul(attn, v) # [batch_size, n_heads, len_q, d_v]
        return prob, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model cannot be divided by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model / n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_k, d_v]
        # attn_attn_mask: [batch_size, n_heads, seq_len, seq_len]
        residual = q
        batch_size = q.size(0)

        # [batch_size, len_q, d_model] -- matmul w_q --> [batch, len_q, d_k * n_heads] -- view --> 
        # [batch, len_q, n_heads, d_k] -- transpose --> [batch, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_v]

        prob, attn = ScaledDotProductAttention()(k, q, v, attn_mask)
        output = self.fc(prob) # [batch_size, len_q, n_heads * d_v]

        return self.layer_norm(residual + output), attn

if __name__ == "__main__":
    d_model, d_ff = 16, 32
    seq_len = 10
    ffn = FeedForwardNetwork(d_model, d_ff)
    input = torch.ones(4, seq_len, d_model)
    output = ffn(input)
    print(output.shape)
