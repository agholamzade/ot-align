import jax.numpy as jnp
from jax import random
from typing import Any, Dict
from flax import linen as nn
from flax import struct
from functools import reduce

ModuleDef = Any

class MLP(nn.Module):
    config: Dict[str, Any]
    def setup(self):
        self.features = self.config.get('features')
        self.act = get_activation(self.config.get('act'))
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x

class CNN(nn.Module):
    config: Dict[str, Any]
    def setup(self):
        self.features = self.config.get('features')
        self.kernels = self.config.get('kernels')
        self.strides = self.config.get('strides')
        self.padding = self.config.get('padding')
        self.act = get_activation(self.config.get('act'))
        self.conv_module = nn.Conv if self.config.get("module_type") == "conv" else nn.ConvTranspose
        self.layers = [self.conv_module(feat, (kernel, kernel), (stride, stride), self.padding) for
                        feat, kernel, stride in zip(self.features, self.kernels, self.strides)]

    def __call__(self, x):
        for i,lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
        return x 
  

class NormalSampling(nn.Module): 
  
  @nn.compact
  def __call__(self, x, z_rng):
    #flatten input
    split_index = x.shape[1] // 2
    mean_x, logvar_x = jnp.split(x, [split_index], axis=1)
    z = reparameterize(z_rng, mean_x,logvar_x)
    return z, mean_x, logvar_x



class Encoder(nn.Module):
    encoder_config: Dict[str, Any]
    latent_dim: int
    def setup(self):
        if "embedding" in self.encoder_config:
            self.embedding = nn.Embed(self.encoder_config["embedding"].get("num_embeddings"), self.encoder_config["embedding"].get("embedding_dim"))
        if "dense_module" in self.encoder_config:
            self.dense_module = MLP(self.encoder_config.get("dense_module"))
        self.sub_module = create_sub_module(self.encoder_config)
        self.act = get_activation(self.encoder_config.get("act"))
        self.dense = nn.Dense(2*self.latent_dim)
    def __call__(self, x):
        if "embedding" in self.encoder_config:
            x = self.embedding(x)
            x = x.reshape(x.shape[0], -1)
        x = self.sub_module(x)
        x = self.act(x)
        #flatten output
        x = x.reshape((x.shape[0], -1))
        if "dense_module" in self.encoder_config:
            x = self.dense_module(x)
            x = self.act(x)
        x = self.dense(x)
        return x     

class Decoder(nn.Module):
    decoder_config: Dict[str, Any]
    latent_dim: int
    def setup(self):
        self.sub_module = create_sub_module(self.decoder_config)
        self.act = get_activation(self.decoder_config.get("act"))

        if self.decoder_config.get("type") == "cnn":
            self.conv_input = self.decoder_config.get("conv_input")
            self.dense = nn.Dense(reduce(lambda x, y: x * y, self.conv_input))
            if "dense_module" in self.decoder_config:
                self.dense_module = MLP(self.decoder_config.get("dense_module"))
        if "split_indices" in self.decoder_config:
            self.split_indices = self.decoder_config.get("split_indices")

    def __call__(self, x):
        if self.decoder_config.get("type") == "cnn":
           if "dense_module" in self.decoder_config:
               x = self.dense_module(x)
               x = self.act(x)
           x = self.dense(x)
           x = self.act(x)
           x = x.reshape(x.shape[0], *self.conv_input)
        x = self.sub_module(x)
        if "split_indices" in self.decoder_config:
            x = jnp.split(x, self.split_indices, axis=1)
        return x   


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

def create_sub_module(model_params):
  if model_params["type"] == "mlp":
     return MLP(model_params)
  elif model_params["type"] == "cnn":
     return CNN(model_params)
  else:
    raise Exception("Unknown Module")

def get_activation(type):
   if type == "relu":
      return nn.relu
   elif type == "sigmoid":
      return nn.sigmoid
   elif type == "tanh":
      return nn.tanh
   elif type == "gelu":
      return nn.gelu
   elif type == "softmax":
      return nn.softmax
   else:
      raise Exception("Unknown Activation Function")