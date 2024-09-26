
from ml_collections import ConfigDict

def get_config():

    config = ConfigDict()

    config.logdir = "./logdir/swiss_roll/"
    config.seed = 0

    # Morph configuration
    config.morph = morph = ConfigDict({"n_iters": 10000, "batch_size": 1024,
                                        "cost_fn": "euclidean", "tau_a": 1, "tau_b": 1, "fused_penalty": 0, "scale_cost": "mean"})

    morph.vf = ConfigDict({"hidden_dims": [128]*8, "time_dims": [128]*8, "condition_dims": [128]*8, "output_dims": 8*[256] + [2]})
    morph.optimizer = morph_optimizer = ConfigDict()
    morph_optimizer.lr_schedule = ConfigDict({"type": "const", "value": 1e-4})

    morph.solver = solver = ConfigDict({"problem_type": "quadratic"})
    solver.sinkhorn = ConfigDict({"max_iterations": 6000})
    solver.gw = ConfigDict({"epsilon": 5e-3, "max_iterations": 1000, "store_inner_errors": True})

    # WandB configuration
    config.wandb = wandb = ConfigDict()
    wandb.project = "swiss_roll"
    wandb.run_name = "swiss_roll_to_spiral"
   
    return config
