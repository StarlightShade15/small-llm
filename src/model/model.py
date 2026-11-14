"""
Decoder-only Transformer model (~156M params, optimized for RTX 5070 Ti).
Components: pre-norm, weight-tied embeddings, RoPE, RMSNorm, SwiGLU MLP.
Optimized: fused RoPE, flash attention, fused SwiGLU, compile-friendly.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA
    ffn_hidden: int = 3072
    max_seq_len: int = 1024
    rope_theta: float = 500000.0
    rope_scaling: Optional[dict] = None
    rmsnorm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_weights: bool = True
    use_bias: bool = False
    # Performance
    use_flash_attn: bool = True
    fused_swiglu: bool = True


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (compile-friendly)."""
    
    __slots__ = ("weight", "eps")
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        # Fused RMSNorm: single pass, no intermediate allocations
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x * self.weight


class RoPE(nn.Module):
    """Rotary Positional Embeddings with precomputed complex frequencies."""
    
    def __init__(self, dim: int, max_seq_len: int, theta: float = 500000.0, scaling: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling
        
        # Precompute complex exponential frequencies (more efficient than sin/cos)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if self.scaling and self.scaling.get("type") == "yarn":
            factor = self.scaling.get("factor", 1.0)
            original_max = self.scaling.get("original_max_seq_len", max_seq_len)
            t = t * (original_max / max_seq_len) / factor
        elif self.scaling and self.scaling.get("type") == "linear":
            factor = self.scaling.get("factor", 1.0)
            t = t / factor
        freqs = torch.outer(t, self.inv_freq)
        # Store as complex for fused RoPE
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    
    def forward(self, seq_len: int, start_pos: int = 0) -> Tuple[Tensor, Tensor]:
        end_pos = start_pos + seq_len
        if end_pos > self.max_seq_len:
            new_max = max(end_pos, self.max_seq_len * 2)
            self._build_cache(new_max)
            self.max_seq_len = new_max
        return self.cos_cached[start_pos:end_pos], self.sin_cached[start_pos:end_pos]


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Fused RoPE application using complex multiplication.
    q, k: (bsz, seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim//2)
    """
    # Reshape for complex multiplication
    # Split last dim into real/imag pairs
    q_r, q_i = q.float().chunk(2, dim=-1)
    k_r, k_i = k.float().chunk(2, dim=-1)
    
    # cos/sin: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    q_out_r = q_r * cos - q_i * sin
    q_out_i = q_r * sin + q_i * cos
    k_out_r = k_r * cos - k_i * sin
    k_out_i = k_r * sin + k_i * cos
    
    q = torch.cat([q_out_r, q_out_i], dim=-1).type_as(q)
    k = torch.cat([k_out_r, k_out_i], dim=-1).type_as(k)
    return q, k


class SwiGLU(nn.Module):
    """Fused SwiGLU: single matmul for gate + up projection."""
    
    __slots__ = ("w13", "w2")
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, fused: bool = True):
        super().__init__()
        self.fused = fused
        if fused:
            # Fused: w13 projects to 2*hidden_dim (gate + up combined)
            self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.fused:
            # Single matmul for gate + up
            x13 = self.w13(x)
            gate, up = x13.chunk(2, dim=-1)
            return self.w2(F.silu(gate) * up)
        else:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Attention(nn.Module):
    """Multi-head attention with GQA, RoPE, and flash attention."""
    
    __slots__ = ("q_proj", "k_proj", "v_proj", "o_proj", "rope", 
                 "n_heads", "n_kv_heads", "head_dim", "kv_dim", "use_flash")
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.use_flash = config.use_flash_attn and hasattr(F, "scaled_dot_product_attention")
        
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0
        
        # Fused QKV projections (more efficient than separate)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling)
    
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
        q, k = apply_rope(q, k, cos, sin)
        
        # Handle KV cache for inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv_cache = (k, v) if kv_cache is not None else None
        
        # Transpose for attention: (bsz, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Flash attention (SDPA)
        if self.use_flash:
            try:
                # Native GQA avoids materializing repeated KV heads on newer PyTorch.
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=mask is None and kv_cache is None,
                    enable_gqa=self.n_kv_heads < self.n_heads,
                )
            except TypeError:
                # Compatibility with PyTorch versions before enable_gqa was added.
                if self.n_kv_heads < self.n_heads:
                    repeat_factor = self.n_heads // self.n_kv_heads
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=mask is None and kv_cache is None,
                )
        else:
            if self.n_kv_heads < self.n_heads:
                repeat_factor = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            # Manual fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            if kv_cache is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)
        return output, new_kv_cache


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with fused attention and SwiGLU."""
    
    __slots__ = ("attn_norm", "attn", "ffn_norm", "ffn")
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.use_bias, config.fused_swiglu)
    
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
    """Decoder-only transformer language model (compile-optimized)."""
    
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
        """Initialize with truncated normal (better than normal)."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range, a=-3*self.config.initializer_range, b=3*self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range, a=-3*self.config.initializer_range, b=3*self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def _print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_embed_params = trainable_params
        if self.config.tie_weights:
            non_embed_params -= self.tok_emb.weight.numel()
        print(f"Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-embedding: {non_embed_params:,}")
        print(f"  ~{total_params / 1e6:.1f}M parameters")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for transformer blocks."""
        self._gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
        return_logits: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tuple[Tensor, Tensor]]]]:
        """
        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) with -100 for masked positions
            kv_cache: List of (k, v) per layer for inference
        Returns:
            logits: (batch, seq_len, vocab_size), or None when labels are supplied
                and return_logits=False
            loss: scalar if labels provided
            new_kv_cache: list of (k, v) per layer
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.tok_emb(input_ids)
        
        # RoPE (shared across layers)
        past_len = kv_cache[0][0].shape[1] if kv_cache is not None else 0
        cos, sin = self.blocks[0].attn.rope(seq_len, start_pos=past_len)
        cos = cos.to(device, non_blocking=True)
        sin = sin.to(device, non_blocking=True)
        
        # With right-padded batches, padded query positions have masked labels and
        # valid tokens never attend to padding. Let SDPA use its native causal path.
        # A boolean upper-triangle SDPA mask means "allowed positions" and would
        # accidentally expose future tokens, making training loss meaningless.
        mask = None
        
        # Forward through blocks
        new_kv_cache = [] if kv_cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            
            if getattr(self, '_gradient_checkpointing', False) and self.training and labels is not None:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(x, cos, sin, mask, kv_cache):
                        return module(x, cos, sin, mask, kv_cache)
                    return custom_forward
                
                x, layer_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, cos, sin, mask, layer_kv,
                    use_reentrant=False,
                )
            else:
                x, layer_kv = block(x, cos, sin, mask, layer_kv)
            
            if new_kv_cache is not None:
                new_kv_cache.append(layer_kv)
        
        # Final norm
        x = self.norm(x)
        
        # When training only on the agent span, project only hidden states that
        # predict an unmasked target. This avoids allocating [B, T, vocab] logits.
        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:]
            valid_targets = shift_labels.ne(-100)
            if return_logits:
                logits = self.head(x)
                shift_logits = logits[..., :-1, :].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.contiguous().view(-1),
                    ignore_index=-100,
                )
            else:
                target_states = x[..., :-1, :][valid_targets]
                target_labels = shift_labels[valid_targets]
                target_logits = self.head(target_states)
                loss = F.cross_entropy(target_logits, target_labels)
                logits = None
        else:
            logits = self.head(x)
        
        return logits, loss, new_kv_cache
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
    ) -> Tensor:
        """Generate tokens autoregressively with KV cache."""
        self.eval()
        bsz = input_ids.shape[0]
        kv_cache = None
        
        # Pre-allocate for speed
        current_input = input_ids
        for _ in range(max_new_tokens):
            logits, _, kv_cache = self.forward(current_input, kv_cache=kv_cache)
            next_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty (vectorized)
            if repetition_penalty != 1.0:
                # More efficient: scatter penalty
                for i in range(bsz):
                    unique_tokens = input_ids[i].unique()
                    next_logits[i, unique_tokens] /= repetition_penalty
            
            # Top-k
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumprobs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                next_logits.scatter_(1, sorted_indices, next_logits.gather(1, sorted_indices).masked_fill(sorted_mask, float("-inf")))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            current_input = next_token
            
            if (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    @classmethod
    def from_config(cls, config_dict: dict) -> "SmallLM":
        config = ModelConfig(**config_dict)
        return cls(config)
    
    def save_pretrained(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        import yaml
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config.__dict__, f)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "SmallLM":
        import yaml, os, glob
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location="cpu")
            config_dict = checkpoint.get("model_config", checkpoint.get("config"))
            if not isinstance(config_dict, dict) or "d_model" not in config_dict:
                raise ValueError(f"Checkpoint at {path} does not include a model configuration")
            state_dict = checkpoint.get("model", checkpoint)
        else:
            # Try config.yaml + model.pt first (standard HF-style)
            config_path = os.path.join(path, "config.yaml")
            model_pt_path = os.path.join(path, "model.pt")
            best_path = os.path.join(path, "best_model.pt")
            latest_path = os.path.join(path, "latest.pt")

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
                if os.path.exists(best_path):
                    checkpoint = torch.load(best_path, map_location="cpu")
                    state_dict = checkpoint["model"]
                    config_dict = checkpoint.get("model_config", config_dict)
                elif os.path.exists(model_pt_path):
                    state_dict = torch.load(model_pt_path, map_location="cpu")
                else:
                    raise FileNotFoundError(f"No model weights found in {path}")
            elif os.path.exists(latest_path):
                # Training checkpoints (latest.pt) contain config in the dict
                checkpoint = torch.load(latest_path, map_location="cpu")
                config_dict = checkpoint.get("config", checkpoint.get("model_config"))
                if not isinstance(config_dict, dict) or "d_model" not in config_dict:
                    raise ValueError(f"Checkpoint at {latest_path} does not include a model configuration")
                state_dict = checkpoint["model"]
            else:
                # Fall back to latest step_*.pt
                step_files = sorted(glob.glob(os.path.join(path, "step_*.pt")))
                if step_files:
                    latest_step = step_files[-1]
                    checkpoint = torch.load(latest_step, map_location="cpu")
                    config_dict = checkpoint.get("config", checkpoint.get("model_config"))
                    if not isinstance(config_dict, dict) or "d_model" not in config_dict:
                        raise ValueError(f"Checkpoint at {latest_step} does not include a model configuration")
                    state_dict = checkpoint["model"]
                else:
                    raise FileNotFoundError(
                        f"No config.yaml/model.pt/latest.pt/step_*.pt found in {path}"
                    )
        model = cls.from_config(config_dict)
        model.load_state_dict(state_dict)
        return model.to(device)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
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
    labels[:, :seq_len//2] = -100
    
    logits, loss, _ = model(input_ids, labels)
    print(f"\nForward test:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test generate
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print(f"  Generated: {generated.shape}")
    
    # Test compile
    if hasattr(torch, "compile"):
        print("\nTesting torch.compile...")
        compiled = torch.compile(model, mode="reduce-overhead")
        logits, loss, _ = compiled(input_ids, labels)
        print(f"  Compiled forward works: {logits.shape}")
