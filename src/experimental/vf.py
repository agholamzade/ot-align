
from typing import Callable, Optional, Sequence, Any

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.training import train_state

from ott.neural.networks.layers import time_encoder


Array = Any
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
        x = nn.Dense(self.output_dim, 
                     kernel_init=nn.initializers.constant(0))(x)
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)
    depth: int = 1

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        for _ in range(self.depth):
          x = nn.Dense(
                  features=self.mlp_dim,
                  dtype=self.dtype,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)(inputs)
          x = nn.gelu(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        return output
    

class VFBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    input_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.silu(c)
        c = nn.Dense(3 * self.input_dim, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_mlp= jnp.split(c, 3, axis=-1)
        
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        # MLP Residual.
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size), out_dim= x_modulated.shape[-1], depth=1)(x_modulated)
        x = x + (gate_mlp * mlp_x)
        return x
    

class vf_test(nn.Module):
 
  hidden_dims: Sequence[int]
  output_dims: Sequence[int]
  last_dim: int
  condition_dims: Optional[Sequence[int]] = None
  time_dims: Optional[Sequence[int]] = None
  time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = time_encoder.cyclical_time_encoder
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      train: bool = True,
  ) -> jnp.ndarray:
    """Forward pass through the neural vector field.

    Args:
      t: Time of shape ``[batch, 1]``.
      x: Data of shape ``[batch, ...]``.
      condition: Conditioning vector of shape ``[batch, ...]``.
      train: If `True`, enables dropout for training.

    Returns:
      Output of the neural vector field of shape ``[batch, output_dim]``.
    """
    time_dims = self.hidden_dims if self.time_dims is None else self.time_dims

    t = self.time_encoder(t)
    for time_dim in time_dims:
      t = self.act_fn(nn.Dense(time_dim)(t))

    for hidden_dim in self.hidden_dims:
      x = self.act_fn(nn.Dense(hidden_dim)(x))

    for cond_dim in self.condition_dims:
        condition = self.act_fn(nn.Dense(cond_dim)(condition))
    
    c = condition + t  
    
    for output_dim in self.output_dims:
        x = VFBlock(input_dim = self.hidden_dims[-1], hidden_size=output_dim)(x, c)
    # No activation function for the final layer
    return FinalLayer(hidden_size=self.hidden_dims[-1], output_dim=self.last_dim)(x, c)

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input_dim: int,
      condition_dim: Optional[int] = None,
  ) -> train_state.TrainState:
    """Create the training state.

    Args:
      rng: Random number generator.
      optimizer: Optimizer.
      input_dim: Dimensionality of the velocity field.
      condition_dim: Dimensionality of the condition of the velocity field.

    Returns:
      The training state.
    """
    t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
    if self.condition_dims is None:
      cond = None
    else:
      assert condition_dim > 0, "Condition dimension must be positive."
      cond = jnp.ones((1, condition_dim))

    params = self.init(rng, t, x, cond, train=False)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )