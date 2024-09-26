from ml_collections import ConfigDict

def get_config():

    config = ConfigDict()
    config.sweep_id = "saz8asjv"
    config.logdir = "./logdir/feature/"
    config.cost_type = "bridge"
    config.epsilon = 5e-3
    config.scale_cost = "mean"

    config.morph = morph = ConfigDict({"n_iters": 10000 + 1, "batch_size": 512, "tau_a": 1, "tau_b": 1, "fused_penalty": 0, "scale_cost": config.scale_cost})

    morph.vf_small = ConfigDict({"hidden_dims": [128]*4, "time_dims": [128]*4, "condition_dims": [128]*4, "output_dims": 4*[256]}) #723K
    morph.vf_medium = ConfigDict({"hidden_dims": [256]*6, "time_dims": [256]*6, "condition_dims": [256]*6, "output_dims": 6*[512]})#3.2M
    morph.vf_large = ConfigDict({"hidden_dims": [1024]*8, "time_dims": [1024]*8, "condition_dims": [1024]*8, "output_dims": 8*[1680]})#49M

    morph.adanl_small = ConfigDict({"hidden_dims": [128]*2, "time_dims": [128]*2, "condition_dims": [128]*2, "output_dims": 5*[128]})#725K
    morph.adanl_medium = ConfigDict({"hidden_dims": [256]*2, "time_dims": [256]*2, "condition_dims": [256]*2, "output_dims": 7*[256]})#3M
    morph.adanl_large = ConfigDict({"hidden_dims": [1024]*2, "time_dims": [1024]*2, "condition_dims": [1024]*2, "output_dims": 8*[1024]})#49M
    
    morph.optimizer = morph_optimizer = ConfigDict()
    morph_optimizer.lr_schedule = ConfigDict({"type": "const", "value": 1e-4})

        # latent Fusion VAE (LatentFusedVAE)
    config.latentfvae = latent_fvae = ConfigDict({"modality": "fused"}) #30M
    latent_fvae.train = train_latentvae = ConfigDict({"num_epochs": 800, "learning_rate": 1e-4, "batch_size": 256, "opt": "adam"})
    train_latentvae.beta_schedule  = ConfigDict({"type": "linear",  "init_value": 1, "end_value": 1, "n_epochs": 1000, "transition_begin": 20})
    latent_fvae.model = model_latentfvae  = ConfigDict({"type": "fused_vae", "latent_dim": 128, "reconstruction_loss1": "mean_squared_error", "reconstruction_loss2": "mean_squared_error", "kl_divergence": "kl_divergence_normal"})
    model_latentfvae.encoder1 =  ConfigDict({"type": "mlp", "features": [1024]*8 , "act": "relu"})
    model_latentfvae.decoder1 =  ConfigDict({"type": "mlp", "features": [1024]*8 + [model_fn["latent_dim"]], "act": "relu"})
    model_latentfvae.encoder2 = ConfigDict({"type": "mlp", "features": [1024]*8 , "act": "relu"})
    model_latentfvae.decoder2 = ConfigDict({"type": "mlp", "features": [1024]*8 + [model_sn["latent_dim"]] , "act": "relu"})



    config.wandb = wandb = ConfigDict()
    wandb.project = "imagenet"
    wandb.run_name = "feature_morph_genot"

    return config