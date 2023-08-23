import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, cl, n_heads, emb_dim):
        """
        We compute the linear transformation of Q,V,K and all heads in
        parallel (this assumes n_heads*head_size == emb_dim, which is the
        convention). Secondly we need the Linear Projection back in to the
        residual path.
        """
        super().__init__()
        self.n_heads = n_heads
        self.lin_t = nn.Linear(emb_dim, 3 * emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.register_buffer("tril", torch.tril(torch.ones(cl, cl)))

    def forward(self, x):
        # Input has Batch (B) and Time (T) and Embedding (C) dimension
        B, T, C = x.shape
        q, k, v = self.lin_t(x).split(C)
        # treat the heads as another Batch dimension
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # masked self attention
        wei = q @ k.transpose(-1, -2)
        wei.masked_fill(self.tril == 0, float("-inf"))
        wei *= C**-0.5
        out = F.softmax(wei, dim=-1) @ v

        # concatenate heads
        out = out.transpose(1, 2).view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        self.linear = nn.Linear(emb_dim, 4 * emb_dim)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x):
        out = self.gelu(self.linear(x))
        return self.proj(out)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, cl, n_heads, emb_dim):
        self.ln1 = nn.LayerNorm()
        self.mha = MaskedMultiHeadAttention(cl, n_heads, emb_dim)
        self.ln2 = nn.LayerNorm()
        self.ff = FeedForward(emb_dim)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_blocks, vocab_size, cl, n_heads, emb_dim):
        assert emb_dim % n_heads == 0, "emb_dim not divisible by n_heads"
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.position_emb = nn.Embedding(cl, emb_dim)
        self.blocks = nn.Sequential(
            *[TransformerDecoderBlock(cl, n_heads, emb_dim) for _ in range(n_blocks)]
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_emb(x) + self.position_emb(torch.arange(T))
        x = self.blocks(x)
        return self.lm_head(x)

    def getProbas(self, x):
        return F.softmax(self(x), dim=-1)