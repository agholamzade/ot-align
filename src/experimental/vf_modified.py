from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.training import train_state

from ott.neural.networks.layers import time_encoder

__all__ = ["VelocityField"]


class MLP_Block(nn.Module):
    hidden_dims: Sequence[int]
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    apply_fn_last: bool = True

    @nn.compact
    def __call__(self,x: jnp.ndarray) -> jnp.ndarray:
        for hidden_dim in self.hidden_dims[:-1]:
            x = nn.Dense(hidden_dim)(x)
            x = self.act_fn(x)
            # x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.hidden_dims[-1])(x)
        if self.apply_fn_last:
            x = self.act_fn(x)
        return x

class VelocityField(nn.Module):

  hidden_dims: Sequence[int]
  output_dims: Sequence[int]
  condition_dims: Sequence[int] 
  time_dims: Sequence[int]
  last_dim: int
  time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = time_encoder.cyclical_time_encoder
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  dropout_rate: float = 0.0

  def setup(self):
    self.time_mlp = MLP_Block(hidden_dims=self.time_dims, act_fn=self.act_fn)
    self.hidden_mlp = MLP_Block(hidden_dims=self.hidden_dims, act_fn=self.act_fn)
    self.condition_mlp = MLP_Block(hidden_dims=self.condition_dims, act_fn=self.act_fn)
    self.output_mlp = MLP_Block(hidden_dims=self.output_dims, act_fn=self.act_fn)

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      train: bool = True,
  ) -> jnp.ndarray:

    t = self.time_encoder(t)
    t = self.time_mlp(t)
    x = self.hidden_mlp(x)
    c = self.condition_mlp(condition)

    feats = jnp.concatenate([t, x, c], axis=-1)
    out = self.output_mlp(feats)
    out = nn.Dense(self.last_dim)(out)

    return out

  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      input_dim: int,
      condition_dim: int,
  ) -> train_state.TrainState:

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
