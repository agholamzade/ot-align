from flax import linen as nn
from typing import Any, Dict
from src.common.networks import *


def get_model(name, model_conf):
    if model_conf["type"] == "vae":
        return VAE(name = name, model_conf = model_conf)
    elif model_conf["type"] == "fused_vae":
        return fVAE(name = name, model_conf = model_conf)
    else:
        raise Exception("Unknown Model")


class VAE(nn.Module):

  """Full VAE model."""
  name: str 
  model_conf: Dict[str, Any]
  sampling_type: str = "normal"

  def setup(self):
      self.encoder_params = self.model_conf.get("encoder")
      self.decoder_params  = self.model_conf.get("decoder")
      self.latent_dim = self.model_conf.get("latent_dim")
      self.output_func = get_activation(self.model_conf.get("output_func"))
      self.encoder = Encoder(self.encoder_params, self.latent_dim)
      self.decoder = Decoder(self.decoder_params, self.latent_dim)
      self.sampling = NormalSampling()
  
  def __call__(self, x, z_rng):
    x = self.encoder(x)
    z, mean_x, logvar_x = self.sampling(x, z_rng)
    recon_x = self.decoder(z)
    return recon_x, mean_x, logvar_x

  def generate(self, z):
    return self.output_func(self.decoder(z))
  
  def encode(self, x):
    return self.encoder(x)
  
  def encode_and_sample(self, x, z_rng):
    x = self.encoder(x)
    z, _, _ = self.sampling(x, z_rng)
    return z
  
  def decode(self, logits, z_rng):
    z, _, _ = self.sampling(logits, z_rng)
    recon_x = self.decoder(z)
    return recon_x
  


class fVAE(nn.Module):

  """Fused VAE model."""
  name: str 
  model_conf: Dict[str, Any]
  sampling_type: str = "normal"

  def setup(self):
      
      self.latent_dim = self.model_conf.get("latent_dim")
      self.encoder_params1 = self.model_conf.get("encoder1")
      self.decoder_params1  = self.model_conf.get("decoder1")
      self.encoder1 = Encoder(self.encoder_params1, self.latent_dim)
      self.decoder1 = Decoder(self.decoder_params1, self.latent_dim)

      self.encoder_params2 = self.model_conf.get("encoder2")
      self.decoder_params2  = self.model_conf.get("decoder2")
      self.encoder2 = Encoder(self.encoder_params2, self.latent_dim)
      self.decoder2 = Decoder(self.decoder_params2, self.latent_dim)
      self.sampling = NormalSampling()

  
  def __call__(self, x, y, z_rng,):
    z_rng1, z_rng2 = random.split(z_rng)
    
    z1 = self.encoder1(x)
    z2 = self.encoder2(y)

    z1, mean_x, logvar_x = self.sampling(z1, z_rng1)
    z2, mean_y, logvar_y = self.sampling(z2, z_rng2)

    recon_x1 = self.decoder1(z1)
    recon_y1 = self.decoder2(z1)

    recon_x2 = self.decoder1(z2)
    recon_y2 = self.decoder2(z2)

    return recon_x1, recon_x2, recon_y1, recon_y2, mean_x, logvar_x, mean_y, logvar_y
  
  def encode_and_sample(self, x, z_rng, encoder = 1):
    if encoder == 1:
        x = self.encoder1(x)
        z, _, _ = self.sampling(x, z_rng)
    else:
       x = self.encoder2(x)
       z, _, _ = self.sampling(x, z_rng)
    return z
      
  def decode(self, z, decoder = 1):
    if decoder == 1:
        recon = self.decoder1(z)
    else:
        recon = self.decoder2(z)
    return recon
  

class Classification_head(nn.Module):
  """Classification head for the VAE model."""
  num_classes: int
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.num_classes)(x)
    return x