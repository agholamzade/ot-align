# https://github.com/yiyixuxu/denoising-diffusion-flax/blob/main/denoising_diffusion_flax/unet.py

import math
from typing import Any, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

# Port of https://github.com/facebookresearch/DiT/blob/main/models.py into jax.

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, embedding_init=nn.initializers.normal(0.02))

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        # x = nn.Dropout(rate=self.dropout_rate)(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        # output = nn.Dropout(rate=self.dropout_rate)(output)
        return output
    
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


    
################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        print("DiTBlock: x of shape", x.shape, "c of shape", c.shape)
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        print("DiTBlock: x_modulated of shape", x_modulated.shape)

        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, x_modulated)
        print("DiTBlock: attn_x of shape", attn_x.shape)
        x = x + (gate_msa[:, None] * attn_x)
        print("DiTBlock: x of shape", x.shape)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        print("DiTBlock: x_modulated2 of shape", x_modulated2.shape)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        print("DiTBlock: out x of shape: ", x.shape)
        return x
    
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

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    output_size: int
    hidden_size: int
    depth: int
    mlp_depth: int
    num_heads: int
    mlp_ratio: float
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        condition: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        
        print("DiT: Input of shape", x.shape)

        t = TimestepEmbedder(self.hidden_size)(t) # (B, hidden_size)

        for _ in range(self.mlp_depth):
            x = self.act_fn(nn.Dense(self.hidden_size)(x))
        
        for _ in range(self.mlp_depth):
            condition = self.act_fn(nn.Dense(self.hidden_size)(condition))

        c = t + condition

        print("x: noise Embed of shape", x.shape)
        print("DiT: Timestep Embedding of shape", t.shape)
        print("DiT: x Embedding of shape", condition.shape)
        print( "DiT: Condition of shape", c.shape)

        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, c)
            # print("DiT: DiTBlock of shape", x.shape)
        x = FinalLayer(self.hidden_size, self.output_size)(x, c) # (B, num_patches, p*p*c)
        # print("DiT: FinalLayer of shape", x.shape)
        return x
    
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
        cond = jnp.ones((1, condition_dim))

        params = self.init(rng, t, x, cond, train=False)["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )