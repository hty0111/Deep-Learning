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
            scores.masked_fill(attn_mask, -1e9)
        
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
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, p_drop):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, p_drop)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x, mask):
        # x: [batch_size, src_seq_len, d_model]
        # mask: [batch_size, src_seq_len, src_seq_len]

        # self-attention + residual + layernorm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))

        # feed-forward + residual + layernorm
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout2(ff_output))

        return x, attn_weights
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, p_drop):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, p_drop)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)
        self.dropout3 = nn.Dropout(p_drop)
    
    def forward(self, x, enc_output, self_mask, cross_mask):
        # x: [batch_size, tgt_seq_len, d_model]
        # enc_output: [batch_size, src_seq_len, d_model]
        # self_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        # cross_mask: [batch_size, tgt_seq_len, src_seq_len]

        # self-attention + residual + layernorm
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, self_mask)
        x = self.layernorm1(x + self.dropout1(self_attn_output))

        # cross-attention + residual + layernorm
        cross_attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.layernorm2(x + self.dropout2(cross_attn_output))

        # feed-forward + residual + layernorm
        ff_output = self.feed_forward(x)
        x = self.layernorm3(x + self.dropout3(ff_output))        

        return x, self_attn_weights, cross_attn_weights
    

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, p_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pe = PositionEncoding(d_model, p_drop, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, p_drop)
            for _ in range(n_layers)
        ])
    
    def forward(self, x, mask):
        # x: [batch_size, src_seq_len]
        x = self.embedding(x) # [batch_size, src_seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1) # [batch_size, src_seq_len, d_model]

        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        
        return x, attn_weights_list
    

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, p_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionEncoding(d_model, p_drop, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, p_drop)
            for _ in range(n_layers)
        ])

    def forward(self, x, enc_output, self_mask, cross_mask):
        # x: [batch_size, tgt_seq_len]
        x = self.embedding(x) # [batch_size, tgt_seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_seq_len, d_model]

        self_attn_weights_list = []
        cross_attn_weights_list = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, enc_output, self_mask, cross_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        
        return x, self_attn_weights_list, cross_attn_weights_list


def create_pad_mask(seq, pad_idx):
    # seq: [batch_size, seq_len]
    # return: [batch_size, 1, 1, seq_len]

    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_subsequent_mask(seq_len, device):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), device=device)).bool()


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, n_layers, d_model, n_heads, d_ff, max_len, p_drop):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, p_drop)
        self.decoder = Decoder(tgt_vocab_size, n_layers, d_model, n_heads, d_ff, max_len, p_drop)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        # src: [batch_size, src_seq_len]
        # tgt: [batch_size, tgt_seq_len]

        if src_mask is None:
            src_mask = create_pad_mask(src, self.src_pad_idx)

        if tgt_mask is None:
            pad_mask = create_pad_mask(tgt, self.tgt_pad_idx)
            subseq_mask = create_subsequent_mask(tgt.size(1), device=tgt.device)
            tgt_mask = pad_mask & subseq_mask
        
        if cross_mask is None:
            cross_mask = src_mask

        enc_output, enc_attn_weights = self.encoder(src, src_mask)
        dec_output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
        output = self.fc(dec_output)

        return output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights


if __name__ == "__main__":
    d_model = 512 # embedding size 
    max_len = 1024 # max length of sequence
    d_ff = 2048 # feedforward nerual network  dimension
    n_layers = 6 # number of encoder and decoder layers
    n_heads = 8 # number of heads in multihead attention
    p_drop = 0.1 # propability of dropout
    src_vocab_size = 1000 
    tgt_vocab_size = 2000
    pad_idx = 0

    transformer = Transformer(src_vocab_size, tgt_vocab_size, pad_idx, pad_idx, n_layers, d_model, n_heads, d_ff, max_len, p_drop)

    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    output, enc_attn_weights, dec_self_attn_weights, dec_cross_attn_weights = transformer(src, tgt)
    print(output.shape) # [batch_size, tgt_seq_len, tgt_vocab_size]
