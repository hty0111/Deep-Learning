import torch
import torch.nn as nn
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, d_model, p_drop=0.1, max_len=1024):
        super().__init__()

        self.dropout = nn.Dropout(p=p_drop)

        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

        # 10000^{2i/d_model}
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.Tensor([10000])) / d_model)) # [d_model / 2]
        
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model})
        # PE(pos, 2i+1) = sin(pos/10000^{2i/d_model})
        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
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

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.ff1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)

        return x
    

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
        assert d_model % n_heads == 0, "d_model must be multiple of n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = self.d_k

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch_size, len_q, d_k]
        # k: [batch_size, len_k, d_k]
        # v: [batch_size, len_k, d_v]
        # attn_attn_mask: [batch_size, seq_len, seq_len]
        batch_size = q.size(0)

        # [batch_size, len_q, d_model] -- matmul w_q --> [batch, len_q, d_k * n_heads] -- view --> 
        # [batch, len_q, n_heads, d_k] -- transpose --> [batch, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_k, d_v]

        prob, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        output = prob.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # [batch_size, len_q, d_model]
        output = self.fc(output) # [batch_size, len_q, d_model]

        return output, attn

if __name__ == "__main__":
    d_model = 512 # embedding size 
    max_len = 1024 # max length of sequence
    d_ff = 2048 # feedforward nerual network  dimension
    n_layers = 6 # number of encoder and decoder layers
    n_heads = 8 # number of heads in multihead attention
    p_drop = 0.1 # propability of dropout

    src_vocab_size = 1000 
    tgt_vocab_size = 1000
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    pad_idx = 0

    q = torch.ones(batch_size, src_seq_len, d_model)
    k = torch.ones(batch_size, tgt_seq_len, d_model)
    v = torch.ones(batch_size, tgt_seq_len, d_model)
    multi_head_attention = MultiHeadAttention(d_model, n_heads)
    output, _ = multi_head_attention(q, k, v)
    print(output.shape)
