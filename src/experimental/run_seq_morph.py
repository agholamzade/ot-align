import sys
import os

import optax
import wandb
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project directory in this case)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import jax
from src.common.data import GenotDataLoader, OTDataExtended, OTDatasetExtended
from src.common.ott import get_gromov_match_fn
from rnn_vf import VelocityFieldRNN
from rnn_genot import RNNGENOT
from ott.neural.methods.flows import dynamics
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ott.solvers.quadratic import gromov_wasserstein
from ott.geometry.costs import SqEuclidean

def sample_spiral(
    n, min_radius, max_radius, key, min_angle=0, max_angle=10, noise=1.0
):
    radius = jnp.linspace(min_radius, max_radius, n)
    angles = jnp.linspace(min_angle, max_angle, n)
    data = []
    noise = jax.random.normal(key, (2, n)) * noise
    for i in range(n):
        x = (radius[i] + noise[0, i]) * jnp.cos(angles[i])
        y = (radius[i] + noise[1, i]) * jnp.sin(angles[i])
        data.append([x, y])
    data = jnp.array(data)
    return data


def sample_swiss_roll(
    n, min_radius, max_radius, length, key, min_angle=0, max_angle=10, noise=0.1
):
    spiral = sample_spiral(
        n, min_radius, max_radius, key[0], min_angle, max_angle, noise
    )
    third_axis = jax.random.uniform(key[1], (n, 1)) * length
    swiss_roll = jnp.hstack((spiral[:, 0:1], third_axis, spiral[:, 1:]))
    return swiss_roll


# Generation parameters
n_spiral = 1
n_swiss_roll = 1
length = 1
min_radius = 3
max_radius = 10
noise = 0
min_angle = 0
angle_shift = 0
max_angle = 9

# Generate the data
# Seed
rng = jax.random.PRNGKey(0)
rng, *subrngs = jax.random.split(rng, 4)


# spiral = sample_spiral(
#     n_spiral,
#     min_radius,
#     max_radius,
#     key=subrngs[0],
#     min_angle=min_angle ,
#     max_angle=max_angle ,
#     noise=noise,
# )
# swiss_roll = sample_swiss_roll(
#     n_swiss_roll,
#     min_radius,
#     max_radius,
#     key=subrngs[1:],
#     length=length,
#     min_angle=min_angle,
#     max_angle=max_angle,
# )

rng, key1, key2 = jax.random.split(rng,3)

# Number of samples to generate for each mean
n_samples = 200

# # Function to generate samples for a given mean
# def generate_samples(key, mean, n_samples):
#     subkeys = jax.random.split(key, mean.shape[0])
#     samples = jax.vmap(lambda subkey, mu: mu + .4*jax.random.normal(subkey, shape=(n_samples, mu.shape[-1])))(subkeys, mean)
#     return samples

def generate_samples(key, mean, n_samples):
    subkeys = jax.random.split(key, mean.shape[0])
    samples = jax.vmap(lambda subkey, mu: mu + .4*jax.random.normal(subkey, shape=(n_samples, mu.shape[-1])))(subkeys, mean)
    return samples

spirals = generate_samples(key1, spiral, n_samples)
swiss_rolls = generate_samples(key2, swiss_roll, n_samples)
spirals = jnp.transpose(spirals, (1, 0, 2))
swiss_rolls = jnp.transpose(swiss_rolls, (1, 0, 2))



src_data = OTDataExtended(quad = swiss_rolls) 
tgt_data = OTDataExtended(quad = spirals)
train_ds = OTDatasetExtended(src_data, tgt_data)

rng, dl_key = jax.random.split(rng)
data_loader = GenotDataLoader(dl_key, train_ds, batch_size=128)

vf_hidden_dims = [256,256,256,256,256,256,256,256]
output_dim = 2
rnn_hidden_dim = 128
batch_size = 128
Sequence_length = spirals.shape[1]

wandb.init(project="experimental", name="rnn_vf_with_rnn")

fig = plt.figure()
for i in range(spirals.shape[1]):
    plt.scatter(spirals[:, i, 0], spirals[:, i, 1], label=f"sample {i}")
wandb.log({"trarget": wandb.Image(fig)})

# Instantiate the CustomRNN model
vf = VelocityFieldRNN(rnn_hidden_dim=rnn_hidden_dim, vf_hidden_dims=vf_hidden_dims, output_dim=output_dim)
solver = gromov_wasserstein.GromovWasserstein(epsilon = .01)

match_fn = get_gromov_match_fn(
                ot_solver=solver,
                cost_fn= SqEuclidean(),
                scale_cost= "mean",
                tau_a= 1,
                tau_b=1,
                fused_penalty= 0)

src_dim = train_ds.src_dim[-1]
tgt_dim = train_ds.tgt_dim[-1]
sequence_length = train_ds.src_dim[0]
print(src_dim, tgt_dim, sequence_length)

rng, rng_init, rng_call = jax.random.split(rng, 3)
optimizer = optax.adam(1e-3)    

genot_model = RNNGENOT(
            vf,
            flow=dynamics.ConstantNoiseFlow(0.0),
            data_match_fn=match_fn,
            source_dim=src_dim,
            target_dim=tgt_dim,
            sequence_length=sequence_length,
            rng=rng_init,
            optimizer=optimizer,
            n_samples_per_src= 1,
            eval_data=swiss_rolls,
        )

genot_model(data_loader, 2000, rng=rng_call)
transported = genot_model.transport(swiss_rolls, rng=rng_call)
print(transported.shape)

