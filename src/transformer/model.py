import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # Not used for feature input
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    input_dim: int = 35 # Adjusted for our dataset
    num_classes: int = 2 # Normal, Trojan
    dropout: float = 0.1

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads if n_kv_heads < n_heads
        if self.n_kv_heads != self.n_heads:
             xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
             xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bsz, n_heads, seqlen, seqlen)
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        output = torch.matmul(scores, xv)  # (bsz, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.dropout(self.attention(self.attention_norm(x), freqs_cis, mask))
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Feature Encoder: input_dim -> 64 -> dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(args.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim)
        )
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.num_classes, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = self.precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def forward(self, x: torch.Tensor):
        # x: (bsz, seqlen, input_dim)
        bsz, seqlen, _ = x.shape
        h = self.feature_encoder(x)
        
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)
        
        for layer in self.layers:
            h = layer(h, freqs_cis, mask=None) # No mask for bidirectional context or full sequence
            
        h = self.norm(h)
        
        # Sequence-level classification pooling
        # Use Max pooling to detect if ANY part of the sequence has a Trojan signature
        # This fits the anomaly detection nature of the task better than Mean pooling
        h = h.max(dim=1)[0] # (bsz, dim)
        
        logits = self.output(h) # (bsz, num_classes)
        return logits
