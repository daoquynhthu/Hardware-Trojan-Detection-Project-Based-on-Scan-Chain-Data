import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------
# Model arguments
# ---------------------------
@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    input_dim: int = 35
    num_classes: int = 2
    dropout: float = 0.2

    # MoE - 优化参数
    use_moe: bool = True
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 1
    moe_inter_dim: int = 256
    moe_balance_coef: float = 0.01
    gate_temperature: float = 0.5


# ---------------------------
# Basic Layers
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


# ---------------------------
# Rotary Embeddings
# ---------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    if device:
        freqs_cis = freqs_cis.to(device)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    seq_len = x.shape[2]
    if freqs_cis.shape[0] > seq_len:
        freqs_cis = freqs_cis[:seq_len]

    assert freqs_cis.shape == (x.shape[2], x.shape[-1]), \
        f"freqs_cis shape {freqs_cis.shape} doesn't match x shape {x.shape[2:]}"
    
    shape = [1, 1, freqs_cis.shape[0], freqs_cis.shape[1]]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------
# Attention
# ---------------------------
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            xk = torch.repeat_interleave(xk, repeat, dim=2)
            xv = torch.repeat_interleave(xv, repeat, dim=2)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        output = torch.matmul(scores, xv)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


# ---------------------------
# FeedForward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, args, hidden_dim_override: Optional[int] = None):
        super().__init__()
        if hidden_dim_override is not None:
            hidden_dim = hidden_dim_override
        else:
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


# ---------------------------
# Gate & MoE
# ---------------------------
class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_experts = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.temperature = args.gate_temperature
        self.weight = nn.Parameter(torch.empty(self.n_experts, args.dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        scores = F.linear(x, self.weight) / self.temperature
        probs = F.softmax(scores, dim=-1)
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        return probs, topk_vals, topk_idx


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_experts = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.gate = Gate(args)
        
        self.experts = nn.ModuleList([
            FeedForward(args, hidden_dim_override=args.moe_inter_dim)
            for _ in range(self.n_experts)
        ])
        
        if args.n_shared_experts > 0:
            self.shared = FeedForward(args, hidden_dim_override=args.moe_inter_dim * args.n_shared_experts)
        else:
            self.shared = None

        self.aux_loss = None

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        N = x_flat.shape[0]
        
        probs, topk_vals, topk_idx = self.gate(x_flat)
        
        if self.topk == 0:
            self.aux_loss = torch.tensor(0.0, device=x.device)
            return x_flat.view(orig_shape)
        
        expert_mask = torch.zeros(N, self.n_experts, device=x.device)
        for i in range(self.topk):
            expert_mask.scatter_(1, topk_idx[:, i:i+1], topk_vals[:, i:i+1])
        
        y = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            mask = expert_mask[:, i] > 0
            if not mask.any():
                continue
            
            expert_input = x_flat[mask]
            expert_output = self.experts[i](expert_input)
            y[mask] += expert_output * expert_mask[mask, i].unsqueeze(-1)
        
        shared_out = self.shared(x_flat) if self.shared is not None else 0
        
        y = y + shared_out
        y = y.view(orig_shape)
        
        importance = probs.mean(0)
        load = expert_mask.sum(0) / (expert_mask.sum() + 1e-9)
        self.aux_loss = self.n_experts * (importance * load).sum()
        
        return y


# ---------------------------
# Block
# ---------------------------
class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.att_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attn = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = MoE(args) if args.use_moe else FeedForward(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis):
        seqlen = x.shape[1]
        freqs_cis = freqs_cis[:seqlen]
        h = x + self.dropout(self.attn(self.att_norm(x), freqs_cis))
        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return h


# ---------------------------
# Token-Level Transformer
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, args.dim),
            nn.BatchNorm1d(args.dim),
            nn.ReLU()
        )

        # Transformer blocks
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # TOKEN CLASSIFICATION HEAD
        # Direct projection from hidden dim to num_classes
        # No pooling involved. Output shape will be (B, Seq_Len, Num_Classes)
        self.classifier = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.dim // 2, args.num_classes)
        )

        # RoPE
        head_dim = args.dim // args.n_heads
        freqs = precompute_freqs_cis(head_dim, args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, x):
        bsz, seqlen, feat = x.shape

        device = x.device
        x_flat = x.reshape(-1, feat).to(device)
        h = self.feature_encoder(x_flat).reshape(bsz, seqlen, -1)

        for layer in self.layers:
            h = layer(h, self.freqs_cis)

        h = self.norm(h) # (B, Seq_Len, Dim)

        # Token Classification Head
        logits = self.classifier(h) # (B, Seq_Len, Num_Classes)

        return logits
    
    def get_aux_loss(self):
        device = next(self.parameters()).device
        aux_loss = torch.tensor(0.0, device=device)
        
        for layer in self.layers:
            layer_aux = getattr(layer.ffn, 'aux_loss', None)
            if layer_aux is not None:
                aux_loss = aux_loss + layer_aux
        return aux_loss
