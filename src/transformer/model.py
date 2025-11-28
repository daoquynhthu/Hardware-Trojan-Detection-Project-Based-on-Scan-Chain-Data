"""
Transformer with MoE
- 修复 B1: RoPE broadcast 维度错误 (关键)
- 修复 B2: aux_loss float+tensor TypeError (中等)
- 修复 B3: topk=0 边界情况 (低)
- 启用所有性能优化
"""

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
    dim: int = 256  # 增大模型容量
    n_layers: int = 4
    n_heads: int = 8  # 增加attention heads
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
    moe_inter_dim: int = 256  # 增大expert容量
    moe_balance_coef: float = 0.01
    gate_temperature: float = 0.5  # 锐化专家路由


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
# Rotary Embeddings - 完全修复版
# ---------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    """
    预计算RoPE频率
    Args:
        dim: head_dim (必须是偶数)
        end: max sequence length
    Returns:
        freqs_cis: (end, dim//2) complex tensor
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (end, dim//2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    if device:
        freqs_cis = freqs_cis.to(device)
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    """
    🔧 B1修复: 调整freqs_cis形状以匹配x的(bsz, n_heads, seqlen, head_dim)
    
    Args:
        freqs_cis: (seqlen, head_dim//2) complex
        x: (bsz, n_heads, seqlen, head_dim//2) complex
    """
    ndim = x.ndim
    # 🔧 修复: shape[2]是seqlen, shape[-1]是head_dim//2
    assert freqs_cis.shape == (x.shape[2], x.shape[-1]), \
        f"freqs_cis shape {freqs_cis.shape} doesn't match x shape {x.shape[2:]}"
    
    # 🔧 修复: broadcast到(1, 1, seqlen, head_dim//2)
    shape = [1, 1, freqs_cis.shape[0], freqs_cis.shape[1]]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    应用RoPE到query和key
    Args:
        xq: (bsz, n_heads, seqlen, head_dim)
        xk: (bsz, n_heads, seqlen, head_dim)
        freqs_cis: (seqlen, head_dim//2) complex tensor
    """
    # Reshape to complex: (..., head_dim//2, 2) -> (..., head_dim//2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 旋转
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
        
        # Project
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape: (bsz, seqlen, n_heads * head_dim) -> (bsz, seqlen, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # Transpose: (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 应用RoPE (维度现在正确: bsz, n_heads, seqlen, head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seqlen])
        
        # GQA: repeat k/v if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = torch.repeat_interleave(xk, n_rep, dim=1)
            xv = torch.repeat_interleave(xv, n_rep, dim=1)

        # Attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).to(x.dtype)
        scores = self.dropout(scores)

        out = torch.matmul(scores, xv)
        out = out.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        return self.wo(out)


# ---------------------------
# Feed-Forward
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, args, hidden_dim_override=None):
        super().__init__()
        hidden = hidden_dim_override or int(2 * (4 * args.dim) / 3)
        hidden = args.multiple_of * ((hidden + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------
# Gating - 增强版
# ---------------------------
class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.topk = args.n_activated_experts
        self.temperature = args.gate_temperature

    def forward(self, x):
        # 训练时添加噪声鼓励探索
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        scores = F.linear(x, self.weight) / self.temperature
        probs = F.softmax(scores, dim=-1)
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_vals, topk_idx, probs


# ---------------------------
# Mixture of Experts - 完全修复版
# ---------------------------
class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_experts = args.n_routed_experts
        self.topk = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList(
            [FeedForward(args, hidden_dim_override=args.moe_inter_dim) 
             for _ in range(self.n_experts)]
        )

        if args.n_shared_experts > 0:
            self.shared_expert = FeedForward(
                args, 
                hidden_dim_override=args.moe_inter_dim * args.n_shared_experts
            )
        else:
            self.shared_expert = None

        self.aux_loss = None

    def forward(self, x):
        # 🔧 B3修复: 处理topk=0边界情况
        if self.topk == 0:
            self.aux_loss = torch.tensor(0.0, device=x.device)
            if self.shared_expert:
                orig_shape = x.shape
                x_flat = x.reshape(-1, x.shape[-1])
                return self.shared_expert(x_flat).reshape(orig_shape)
            return x
        
        orig_shape = x.shape
        bsz, seqlen, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # Shared expert
        shared_out = self.shared_expert(x_flat) if self.shared_expert else None

        # Gate
        topk_vals, topk_idx, probs = self.gate(x_flat)

        # Expert routing
        y = torch.zeros_like(x_flat)
        
        # 计算expert mask
        expert_mask = torch.zeros(len(x_flat), self.n_experts, device=x.device)
        for i in range(self.topk):
            expert_mask.scatter_(1, topk_idx[:, i:i+1], topk_vals[:, i:i+1])
        
        # Forward through experts
        for i in range(self.n_experts):
            mask = expert_mask[:, i] > 0
            if not mask.any():
                continue
            
            expert_input = x_flat[mask]
            expert_output = self.experts[i](expert_input)
            y[mask] += expert_output * expert_mask[mask, i:i+1]

        # Add shared expert
        if shared_out is not None:
            y = y + shared_out

        y = y.reshape(orig_shape)

        # Auxiliary loss (balance loss)
        importance = probs.mean(0)  # (n_experts,) - 每个expert的平均激活概率
        load = expert_mask.sum(0) / (expert_mask.sum() + 1e-9)  # (n_experts,) - 实际负载
        
        # 鼓励均匀分布: importance和load应该都接近1/n_experts
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
        h = x + self.dropout(self.attn(self.att_norm(x), freqs_cis))
        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return h


# ---------------------------
# Transformer - 终极增强版
# ---------------------------
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 深度特征编码器 with BatchNorm
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

        # 双分支池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 🔧 增强: 添加attention pooling作为第三分支
        self.attention_pool = nn.Sequential(
            nn.Linear(args.dim, 1),
            nn.Softmax(dim=1)
        )

        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.Linear(args.dim * 3, args.dim * 2),  # *3因为有3个pooling分支
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(args.dim * 2, args.dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.dim, args.dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.dim // 2, args.num_classes)
        )

        # 预计算RoPE (使用head_dim)
        head_dim = args.dim // args.n_heads
        freqs = precompute_freqs_cis(head_dim, args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, x):
        bsz, seqlen, feat = x.shape

        # Feature encoding
        x_flat = x.reshape(-1, feat)
        h = self.feature_encoder(x_flat).reshape(bsz, seqlen, -1)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h, self.freqs_cis)

        h = self.norm(h)

        # 三分支池化
        h_t = h.transpose(1, 2)  # (bsz, dim, seqlen)
        
        # Max pooling
        h_max = self.max_pool(h_t).squeeze(-1)
        
        # Average pooling
        h_avg = self.avg_pool(h_t).squeeze(-1)
        
        # Attention pooling
        attn_weights = self.attention_pool(h)  # (bsz, seqlen, 1)
        h_attn = (h * attn_weights).sum(dim=1)  # (bsz, dim)
        
        # Concatenate all pooling outputs
        h_cat = torch.cat([h_max, h_avg, h_attn], dim=1)  # (bsz, dim*3)

        logits = self.classifier(h_cat)

        return logits
    
    def get_aux_loss(self):
        """
        🔧 B2修复: 正确处理tensor类型
        获取所有MoE层的auxiliary loss
        """
        aux_loss = 0.0
        for layer in self.layers:
            if hasattr(layer.ffn, 'aux_loss') and layer.ffn.aux_loss is not None:
                # 🔧 修复: 处理tensor类型
                if isinstance(layer.ffn.aux_loss, torch.Tensor):
                    aux_loss += layer.ffn.aux_loss.item()
                else:
                    aux_loss += layer.ffn.aux_loss
        return aux_loss