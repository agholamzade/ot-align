from src.experiments.swirl import swiss_spiral_morph
from src.experiments.rotated_mnist import train_vae_mnist, discrete_moph, morph_genot, train_fused_vae , latent_fused_vae
from src.experiments.imagenet import morph_features_genot, classification_head

import wandb

     

def rotated_mnist(exp_config, mode):

    if mode == "train":
        train_vae_mnist(exp_config)
    elif "train_fused" in mode:
       wandb.agent(exp_config.sweep_id, function=lambda: train_fused_vae(exp_config), project=exp_config.wandb.project, count=1)
    elif "train_latent" in mode:
       wandb.agent(exp_config.sweep_id, function=lambda: latent_fused_vae(exp_config), project=exp_config.wandb.project, count=1)
    elif "morph_discrete" in mode:
        discrete_moph(exp_config)
    elif "morph_genot" in mode:
        wandb.agent(exp_config.sweep_id, function=lambda: morph_genot(exp_config), project=exp_config.wandb.project, count=1)
    else:
        raise ValueError("Invalid mode")
        

def swiss_roll(exp_config, mode):
    if mode == "train":
        raise ValueError("Swiss roll does not have a train mode")
    elif mode == "morph":
        swiss_spiral_morph(exp_config)
    elif mode == "eval":
        raise ValueError("Swiss roll does not have an eval mode")

def morph_features(exp_config, mode):
  if mode == "morph_discrete":
     raise ValueError("Morph features does not have a discrete mode")
  elif mode == "morph_genot":
    wandb.agent(exp_config.sweep_id, function=lambda: morph_features_genot(exp_config), project=exp_config.wandb.project, count=1)
  elif mode == "classification_head":
     wandb.agent(exp_config.sweep_id, function=lambda: classification_head(exp_config), project=exp_config.wandb.project, count=1)
  else:
    raise ValueError("Invalid mode")