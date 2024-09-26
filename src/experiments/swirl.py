import os

from functools import partial

import jax 
from src.common.utils import sample_spiral, sample_swiss_roll
from src.common.data import OTDataExtended, OTDatasetExtended
from src.experimental.vf import vf_test
import wandb
from src.common.morph import MorphUsingGenot
from ott.neural.networks import velocity_field
from matplotlib import pyplot as plt

def split_data(key, n_samples, train_ratio=0.6):    
    indices = jax.random.permutation(key, n_samples)    
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
        
    return train_indices, test_indices

def swiss_spiral_morph(exp_config):

    rng = jax.random.PRNGKey(exp_config.seed)
    rng, *subrngs = jax.random.split(rng, 4)
    # Generation parameters
    n_spiral = 2000
    n_swiss_roll = 2000
    length = 10
    min_radius = 3
    max_radius = 10
    noise = 0.4
    min_angle = 0
    max_angle = 9
    angle_shift = 3

    spiral = sample_spiral(
        n_spiral,
        min_radius,
        max_radius,
        key=subrngs[0],
        min_angle=min_angle + angle_shift,
        max_angle=max_angle + angle_shift,
        noise=noise,
    )

    swiss_roll = sample_swiss_roll(
        n_swiss_roll,
        min_radius,
        max_radius,
        key=subrngs[1:],
        length=length,
        min_angle=min_angle,
        max_angle=max_angle,
    )   
    
    run = wandb.init(project= exp_config.wandb.project,
                group= "swiss_spiral", 
                job_type= "morph",
                name = exp_config.wandb.run_name,
                config= exp_config.to_dict())


    rng, *subrngs = jax.random.split(rng, 3)

    train_swiss_roll_indices, test_swiss_roll_indices = split_data(subrngs[0], swiss_roll.shape[0])
    test_spiral_indices, train_spiral_indices = split_data(subrngs[1], spiral.shape[0])

    train_swiss_roll, test_swiss_roll = swiss_roll[train_swiss_roll_indices], swiss_roll[test_swiss_roll_indices]
    train_spiral, test_spiral = spiral[train_spiral_indices], spiral[test_spiral_indices]
    
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(test_spiral[:, 0], test_spiral[:, 1], c= test_spiral_indices)
    wandb.log({"target": wandb.Image(fig)}, commit = False)

    def callback_func(transform_func, step,eval_data, eval_labels):
            fig = plt.figure(figsize=(5, 5))
            trasnformed_data = transform_func(eval_data)
            plt.scatter(trasnformed_data[:, 0], trasnformed_data[:, 1], c= eval_labels)
            wandb.log({"morphed": wandb.Image(fig)}, step = step)
    
    callback_func = partial(callback_func, eval_data = test_swiss_roll, eval_labels = test_swiss_roll_indices)

    src_data = OTDataExtended(quad = train_swiss_roll) 
    tgt_data = OTDataExtended(quad = train_spiral)
    train_ds = OTDatasetExtended(src_data, tgt_data)

    save_path = os.path.abspath(os.path.join(exp_config.logdir, "morph", "mlp_vf", wandb.run.id))


    # vf = velocity_field.VelocityField(**exp_config.morph.vf)
    vf = vf_test(**exp_config.morph.vf)
    morphing = MorphUsingGenot(rng,
                                exp_config,
                                train_ds,
                                vf,
                                train_data = train_swiss_roll,
                                eval_data = test_swiss_roll,
                                use_true_alignment = False,
                                save_path= save_path,
                                callback_func= callback_func)
    model = morphing.morph()

    print("Model saved at: ", save_path)
    

    
