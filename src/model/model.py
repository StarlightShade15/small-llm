"""
Decoder-only Transformer model (~70M params).
Components: pre-norm, weight-tied embeddings, RoPE, RMSNorm, SwiGLU MLP.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    ffn_hidden: int = 2048
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    rmsnorm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_weights: bool = True
    use_bias: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to q and k."""
    # q, k: (batch, seq_len, n_heads, head_dim)
    q_rot = q[..., :q.shape[-1] // 2]
    q_pass = q[..., q.shape[-1] // 2:]
    k_rot = k[..., :k.shape[-1] // 2]
    k_pass = k[..., k.shape[-1] // 2:]
    
    # cos, sin: (seq_len, head_dim // 2) -> broadcast
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    q_rotated = q_rot * cos - q_pass * sin
    k_rotated = k_rot * cos - k_pass * sin
    
    q = torch.cat([q_rotated, q_pass], dim=-1)
    k = torch.cat([k_rotated, k_pass], dim=-1)
    return q, k


class RoPE(nn.Module):
    """Rotary Positional Embeddings."""
    
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0, scaling: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin for max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if self.scaling and self.scaling.get("type") == "linear":
            factor = self.scaling.get("factor", 1.0)
            t = t / factor
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    
    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(Wx) with gating."""
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional GQA."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling)
        self.dropout = nn.Dropout(0.0)  # No dropout in attention
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        bsz, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        # Update cache
        new_kv_cache = (k, v) if kv_cache is not None or self.training else None
        
        # Repeat KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)
        
        # Transpose for attention: (bsz, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        # Use flash attention if available (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=mask is None and kv_cache is None,
            )
        else:
            # Manual attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            if kv_cache is None:  # Causal mask for training
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        return output, new_kv_cache


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention and SwiGLU MLP."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.use_bias)
    
    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-norm attention
        residual = x
        x = self.attn_norm(x)
        x, kv_cache = self.attn(x, cos, sin, mask, kv_cache)
        x = x + residual
        
        # Pre-norm FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        
        return x, kv_cache


class SmallLM(nn.Module):
    """Decoder-only transformer language model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final norm
        self.norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        
        # Output head (weight-tied with token embeddings)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print parameter count
        self._print_param_count()
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def _print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Non-embedded params (exclude tok_emb if tied)
        non_embed_params = trainable_params
        if self.config.tie_weights:
            non_embed_params -= self.tok_emb.weight.numel()
        print(f"Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-embedding: {non_embed_params:,}")
        print(f"  ~{total_params / 1e6:.1f}M parameters")
    
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        kv_cache: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[list]]:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) with -100 for masked positions
            kv_cache: List of (k, v) per layer for inference
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if labels provided
            new_kv_cache: list of (k, v) per layer
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.tok_emb(input_ids)
        
        # RoPE
        cos, sin = self.blocks[0].attn.rope(seq_len)
        cos = cos.to(device)
        sin = sin.to(device)
        
        # Causal mask for training
        mask = None
        if labels is not None:  # Training
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
        
        # Forward through blocks
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x, layer_kv = block(x, cos, sin, mask, layer_kv)
            if kv_cache is not None:
                new_kv_cache.append(layer_kv)
        
        # Final norm
        x = self.norm(x)
        
        # Logits
        logits = self.head(x)
        
        # Loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss, new_kv_cache if kv_cache is not None else None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        bsz = input_ids.shape[0]
        kv_cache = None
        
        for _ in range(max_new_tokens):
            # Forward with KV cache
            logits, _, kv_cache = self.forward(input_ids, kv_cache=kv_cache)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(bsz):
                    for token_id in set(input_ids[i].tolist()):
                        next_logits[i, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1:]
                next_logits[indices_to_remove] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    @classmethod
    def from_config(cls, config_dict: dict) -> "SmallLM":
        config = ModelConfig(**config_dict)
        return cls(config)
    
    def save_pretrained(self, path: str):
        """Save model and config."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        import yaml
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config.__dict__, f)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "SmallLM":
        """Load model from saved path."""
        import yaml
        with open(os.path.join(path, "config.yaml"), "r") as f:
            config_dict = yaml.safe_load(f)
        model = cls.from_config(config_dict)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device)
        model.load_state_dict(state_dict)
        return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model creation
    config = ModelConfig()
    model = SmallLM(config)
    
    # Test forward
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :seq_len//2] = -100  # Mask first half
    
    logits, loss, _ = model(input_ids, labels)
    print(f"\nForward test:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test generate
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print(f"  Generated: {generated.shape}")