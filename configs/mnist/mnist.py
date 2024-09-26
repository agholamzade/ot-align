
from ml_collections import ConfigDict

def get_config():

    config = ConfigDict()

    config.logdir = "./logdir/mnist/"
    config.n_angles = 1
    config.seed = 0
    config.load_id = "u8z8djci"
    config.num_epochs = 101
    config.epsilon = 1e-3
    config.fused_penalty = 1
    config.scale_cost = "mean"
    config.sweep_id = "frynxcox"

    config.vf =  ConfigDict({"hidden_dims": [1024]*2, "time_dims": [1024]*2, "condition_dims": [1024]*2, "output_dims": 8*[1024]})#49M

    # First Network (FN)
    config.FN = FN = ConfigDict({"modality": "image"})
    FN.train = train_fn = ConfigDict({"num_epochs": config.num_epochs, "opt": "adam", "learning_rate": 0.001, "batch_size": 1024})
    train_fn.beta_schedule  = ConfigDict({"type": "const",  "value": 1})
    FN.model = model_fn = ConfigDict({"type": "vae", "latent_dim": 16, "output_func": "sigmoid", "reconstruction_loss": "binary_cross_entropy", "kl_divergence": "kl_divergence_normal"})
    model_fn.encoder = ConfigDict({"type": "cnn", "module_type": "conv", "features": [128, 256, 512], "kernels": [3, 3, 3], "strides": [2, 2,2], "padding": "SAME", "act": "relu"})
    model_fn.decoder = ConfigDict({"type": "cnn", "module_type": "convT", "features": [256, 128,1], "kernels": [3, 3, 3], "strides": [2, 2, 1], "conv_input": [7, 7, 512], "padding": "SAME", "act": "relu"})

    # Second Network (SN)
    config.SN = SN = ConfigDict({"modality": "text"})
    SN.train = train_sn = ConfigDict({"num_epochs": config.num_epochs, "learning_rate": 0.01, "opt": "adam", "batch_size": 1024})
    train_sn.beta_schedule = ConfigDict({"type": "const",  "value":1})
    SN.model = model_sn = ConfigDict({"type": "vae", "latent_dim": 4, "output_func": "softmax", "reconstruction_loss": "softmax_cross_entropy", "kl_divergence": "kl_divergence_normal"})
    model_sn.encoder  = ConfigDict({"type": "mlp", "features": [64,32], "act": "relu"})
    model_sn.decoder  = ConfigDict({"type": "mlp", "features": [32,16, 10*config.n_angles], "act": "relu"})

    # end-to-end Fusion VAE (FusedVAE)
    config.FusedVAE = FusedVAE = ConfigDict({"modality": "fused"}) # text_image 36M image_text 32M
    FusedVAE.train = train_fusedvae = ConfigDict({"num_epochs": 400+1, "learning_rate": 1e-4, "batch_size": 256, "opt": "adam"})
    train_fusedvae.beta_schedule  = ConfigDict({"type": "linear",  "init_value": 1, "end_value": 1, "n_epochs": 100, "transition_begin": 60})
    FusedVAE.model = model_fusedvae  = ConfigDict({"type": "fused_vae", "latent_dim": 16, "reconstruction_loss1": "binary_cross_entropy", "reconstruction_loss2": "softmax_cross_entropy", "kl_divergence": "kl_divergence_normal"})
    model_fusedvae.encoder1 = ConfigDict({"type": "cnn", "module_type": "conv", "features": [512, 1024, 2048], "kernels": [3, 3, 3], "strides": [2, 2,2], "padding": "SAME", "act": "relu"})
    model_fusedvae.decoder1 =  ConfigDict({"type": "cnn", "module_type": "convT", "features": [1024, 512,1], "kernels": [3, 3, 3], "strides": [2, 2, 1], "conv_input": [7, 7, 2048], "padding": "SAME", "act": "relu"})
    model_fusedvae.encoder2 = ConfigDict({"type": "mlp", "features": [64,32], "act": "relu"})
    model_fusedvae.decoder2 = ConfigDict({"type": "mlp", "features": [32,16, 10*1], "act": "relu"})

    # latent Fusion VAE (LatentFusedVAE)
    config.latentfvae = latent_fvae = ConfigDict({"modality": "fused"}) #30M
    latent_fvae.train = train_latentvae = ConfigDict({"num_epochs": 800, "learning_rate": 1e-4, "batch_size": 256, "opt": "adam"})
    train_latentvae.beta_schedule  = ConfigDict({"type": "linear",  "init_value": 1, "end_value": 1, "n_epochs": 1000, "transition_begin": 20})
    latent_fvae.model = model_latentfvae  = ConfigDict({"type": "fused_vae", "latent_dim": 128, "reconstruction_loss1": "mean_squared_error", "reconstruction_loss2": "mean_squared_error", "kl_divergence": "kl_divergence_normal"})
    model_latentfvae.encoder1 =  ConfigDict({"type": "mlp", "features": [1024]*8 , "act": "relu"})
    model_latentfvae.decoder1 =  ConfigDict({"type": "mlp", "features": [1024]*8 + [model_fn["latent_dim"]], "act": "relu"})
    model_latentfvae.encoder2 = ConfigDict({"type": "mlp", "features": [1024]*8 , "act": "relu"})
    model_latentfvae.decoder2 = ConfigDict({"type": "mlp", "features": [1024]*8 + [model_sn["latent_dim"]] , "act": "relu"})
    



    config.net_type = "vf_adanl"
    config.net_size = "large"
    
    config.morph = morph_config =ConfigDict({"scale_cost": config.scale_cost, "epsilon": config.epsilon, "fused_penalty": config.fused_penalty, "n_iters": 3e3+1, "batch_size": 256})
    morph_config.optimizer = ConfigDict()
    morph_config.optimizer.lr_schedule = ConfigDict({"type": "const", "value": 1e-4})

    morph_config.vf_small = ConfigDict({"hidden_dims": [128]*4, "time_dims": [128]*4, "condition_dims": [128]*4, "output_dims": 4*[256]}) #723K
    morph_config.vf_medium = ConfigDict({"hidden_dims": [256]*6, "time_dims": [256]*6, "condition_dims": [256]*6, "output_dims": 6*[512]})#3.2M
    morph_config.vf_large = ConfigDict({"hidden_dims": [1024]*8, "time_dims": [1024]*8, "condition_dims": [1024]*8, "output_dims": 8*[1680]})#49M

    morph_config.adanl_small = ConfigDict({"hidden_dims": [128]*2, "time_dims": [128]*2, "condition_dims": [128]*2, "output_dims": 5*[128]})#725K
    morph_config.adanl_medium = ConfigDict({"hidden_dims": [256]*2, "time_dims": [256]*2, "condition_dims": [256]*2, "output_dims": 7*[256]})#3M
    morph_config.adanl_large = ConfigDict({"hidden_dims": [1024]*2, "time_dims": [1024]*2, "condition_dims": [1024]*2, "output_dims": 8*[1024]})#49M


    # WandB configurationkld_loss_y
    config.wandb = wandb = ConfigDict()
    wandb.project = "mnist_sweeps"



    return config
