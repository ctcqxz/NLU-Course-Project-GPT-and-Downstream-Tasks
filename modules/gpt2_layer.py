from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    This helper applies the output projection + dropout to a sub-layer's output,
    then adds the residual (the sub-layer input). No layer norm here.
    """
    return input + dropout(dense_layer(output))


  def forward(self, hidden_states, attention_mask):
    """
    GPT-2 uses a pre-LayerNorm transformer block:
      - LayerNorm -> self-attention -> dense/dropout -> residual add
      - LayerNorm -> feed-forward (dense + gelu) -> dense/dropout -> residual add
    """
    # Multi-head self-attention sub-layer (pre-LN).
    attn_output = self.self_attention(self.attention_layer_norm(hidden_states), attention_mask)
    hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)

    # Feed-forward sub-layer (pre-LN).
    ff_output = self.interm_af(self.interm_dense(self.out_layer_norm(hidden_states)))
    hidden_states = self.add(hidden_states, ff_output, self.out_dense, self.out_dropout)

    return hidden_states


