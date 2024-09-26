import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import optax
from ott.neural.methods.flows import dynamics
from ott.neural.networks import velocity_field
from typing import Callable, Optional, Sequence
from flax.training import train_state


Array = jax.Array

class CustomGRUCell(nn.RNNCellBase):
  features: int
  output_dim: int
  
  @nn.compact
  def __call__(self, carry: tuple[Array], input: Array
               ) -> tuple[tuple[Array, Array], tuple[Array, Array, Array]]:
    carry, y = nn.GRUCell(self.features)(carry, input)
    mean = nn.Dense(self.output_dim)(y)
    logvar = nn.Dense(self.output_dim)(y)
    rng = self.make_rng("gru")
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    z = mean + eps * std 
    return carry, (mean, logvar, z)
  
  @property
  def num_feature_axes(self) -> int:
    return 1


class VelocityFieldRNN(nn.Module):
  vf_hidden_dims: Sequence[int]
  rnn_hidden_dim: int
  output_dim: int

  def setup(self):
    self.gru_cell = CustomGRUCell(features=self.rnn_hidden_dim, output_dim=self.output_dim)
    vf_output_dims = (*self.vf_hidden_dims, self.output_dim)
    self.vf = velocity_field.VelocityField(hidden_dims=self.vf_hidden_dims, output_dims=vf_output_dims, condition_dims= self.vf_hidden_dims)
    # self.rnn = nn.RNN(
    #     gru_cell,
    #     split_rngs={'params': False, 'gru': True},
    #     name='rnn',
    # )

  def __call__(self, t, x_t, src, latent, carry):

    gru_input = jnp.concatenate([latent , src], axis=-1)
    carry, (mean, logvar, z) = self.gru_cell(carry, gru_input)
    
    v_t = self.vf(t, x_t, condition=src, train=True)
    return v_t, carry, (mean, logvar, z)
  
  def rnn_output(self, src, latent, carry):
    gru_input = jnp.concatenate([latent , src], axis=-1)
    carry, (mean, logvar, z) = self.gru_cell(carry, gru_input)
    return carry, (mean, logvar, z)
  
  def vf_output(self, t, x_t, src):
    v_t = self.vf(t, x_t, condition=src, train=False)
    return v_t 
  
  def create_train_state(
      self,
      rng: jax.Array,
      optimizer: optax.OptState,
      src_dim: int,
      tgt_dim: int,
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
    batch_size = 128
    src = jnp.ones((batch_size, src_dim))
    x_t = jnp.ones((batch_size, tgt_dim))
    last_output = jnp.ones((batch_size, tgt_dim))
    t = jnp.ones((batch_size, 1))

    rng, gru_rng = jax.random.split(rng)
    initial_gru_carry = jnp.zeros((1, self.rnn_hidden_dim))
    params = self.init({"params": rng, "gru": gru_rng}, t, x_t, src, last_output, initial_gru_carry)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer
    )


# rng = jax.random.PRNGKey(0)  

# batch_size = 128
# Sequence_length = 10
# src_dim = 3
# tgt_dim = 2 
# rnn_hidden_dim = 10


# model = VelocityFieldRNN(vf_hidden_dims=[10, 10], rnn_hidden_dim=rnn_hidden_dim, output_dim=tgt_dim)

# model_train_state = model.create_train_state(
#     rng=rng,
#     tgt_dim=tgt_dim,
#     src_dim=src_dim,
#     optimizer=optax.adam(1e-3))

# rng, gru_rng = jax.random.split(rng)

# src = jnp.ones((batch_size, src_dim))
# tgt = jnp.ones((batch_size, tgt_dim))
# x_t = jnp.ones((batch_size, tgt_dim))
# t = jnp.ones((batch_size, 1))
# last_output = jnp.ones((batch_size, tgt_dim))
# initial_gru_carry = jnp.zeros((1, rnn_hidden_dim))

# params = model_train_state.params
# outputs = model_train_state.apply_fn({"params": params}, x_t, t, src, tgt, last_output, initial_gru_carry, rngs = {"gru": gru_rng})

# print(outputs[0].shape)  # (128, 2)
# # tabulate_fn = nn.tabulate(model, jax.random.PRNGKey(0))

# # print(outputs[0].shape)  # (128, 10, 2)

# # print(tabulate_fn(t, x, (initial_state, initial_input)))  # (2, 128)