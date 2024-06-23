#a 1 layer implementation of the GPT architecture
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import warnings
import numpy as np


class MLP(nn.Module):
    def __init__(self, d_model, mlp_hidden, output_dim=None,dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = d_model
        self.c_fc = nn.Linear(d_model, mlp_hidden, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(mlp_hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, temp=None):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.temp = temp
        self.kq = nn.Linear(d_model, 2 * d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, pe_softmax=None):
        batch, length, dim = x.size()

        if pe_softmax is not None:
            k, q = self.kq(x + pe_softmax).split(self.d_model, dim=2)
        else:
            k, q = self.kq(x).split(self.d_model, dim=2)

        v = self.v(x)

        k = k.view(batch, length, self.n_head, dim // self.n_head).transpose(
            1, 2
        )  # (batch,n_head,length,head_dim)
        q = q.view(batch, length, self.n_head, dim // self.n_head).transpose(1, 2)
        v = v.view(batch, length, self.n_head, dim // self.n_head).transpose(1, 2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        scale_factor = 1 / math.sqrt(q.size(-1)) if self.temp is None else self.temp
        attn_bias = torch.zeros(length, length, dtype=q.dtype, device=x.device)
        temp_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

        attn_weight = q @ k.transpose(-2, -1) * scale_factor

        attn_weight += attn_bias
        self.attn_scores = torch.softmax(attn_weight, dim=-1)

        out = self.attn_scores @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(batch, length, dim)
        out = self.attn_out(out)

        return out.squeeze()


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, d_model, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, d_model, n_head, temp=None, mlp_hidden=None):
        super().__init__()

        self.ln_1 = LayerNorm(d_model, bias=True)
        self.attn = CausalSelfAttention(d_model, n_head, temp)
        self.attn_out_resid_dummy = nn.Identity()

        self.ln_2 = LayerNorm(d_model, bias=True)

    def forward(self, x, pe_softmax=None):
        x = self.ln_1(x)
        o = self.attn(x, pe_softmax)

        self.attn_out = x + o
        x = x + o
        x = self.attn_out_resid_dummy(x)  # dummy so we can hook
        x = self.ln_2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int,
    ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[
            :, : len(torch.arange(1, embed_dim, 2))
        ]
        self.register_buffer("pe", pe)

    def forward(self, pos):
        return self.pe[:, pos]


class GPT(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        n_head,
        context_length,
        mlp_hidden=None,
        seed=10,
        temp=None,
        use_pe=True,
        pe_type='learnable' #learnable vs sinusoid
    ):
        super().__init__()

        # set seed
        # torch.manual_seed(seed)
        self.context_length = context_length
        self.use_pe = use_pe

        if mlp_hidden is None:
            mlp_hidden = d_model * 4

        if use_pe not in {True, False, "pe_softmax"}:
            raise ValueError(f"use_pe must be one of True, False, or 'pe_softmax', got {use_pe}")

        if pe_type not in {'learnable','sinusoid'}:
            raise ValueError(f"pe_type must be one of 'learnable' or 'sinusoid', got {pe_type}")


        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_dim, d_model),
                wpe=nn.Embedding(context_length, d_model) if pe_type == 'learnable' else PositionalEncoding(d_model, context_length),
                h=Block(d_model, n_head, temp, mlp_hidden),
                mlp=MLP(d_model, mlp_hidden, output_dim=input_dim),
            )
        )

    def forward(self, x):
        device = x.device
        # rather than asserting,just raise a warning
        if x.size(1) > self.context_length:
            warnings.warn(
                f"This model is not designed to handle sequences longer than the context length, current length {x.size(1)}, block size is only {self.context_length}"
            )
            # cut the sequence to the context length
            # loop through the sequence, iterating by context length chunks
            # then concatenate
            return self.forward_long(x)

        # forward the model itself
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)  # shape (t)
        embed = self.transformer.wte(x)  # token embeddings of shape (b, t, n_embd)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        if self.use_pe == "pe_softmax":
            x = self.transformer.h(embed, pe_softmax=pos_emb)
        elif self.use_pe:
            x = embed + pos_emb
            x = self.transformer.h(x)
        else:
            x = embed
            x = self.transformer.h(x)

        x = self.transformer.mlp(x)
        return x, self.transformer.h.attn_out

    def forward_long(self, x):
        device = x.device
        chunks = x.size(1) // self.context_length
        outs = []
        attn_outs = []
        pos = torch.arange(
            0, self.context_length, dtype=torch.long, device=device
        )  # shape (t)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        for i in range(chunks):
            o = x[:, i * self.context_length : (i + 1) * self.context_length]
            embed = self.transformer.wte(o)
            o = embed + pos_emb
            o = self.transformer.h(o)
            outs.append(self.transformer.out(o))
            attn_outs.append(self.transformer.h.attn_out)
        # stack and return
        outs = torch.cat(outs, dim=1)
        attn_outs = torch.cat(attn_outs, dim=1)
        return outs, attn_outs
