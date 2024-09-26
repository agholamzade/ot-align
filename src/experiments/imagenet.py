import wandb
import jax
import jax.numpy as jnp
from src.common.data import OTDataExtended, OTDatasetExtended
from src.common.morph import MorphUsingGenot
from src.common.models import Classification_head
from src.experimental.vf_modified import VelocityField
from src.experimental.vf import vf_test
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import partial
import numpy as np
import umap
from collections import defaultdict
from ott.geometry import pointcloud, costs, geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from src.experiments.match_helpers import get_match_sinhorn
import optax
from flax.training import train_state
from src.common.data import FusedDsBuilder
from src.common.models import get_model


def get_umap_transform(features, n_components=2, num_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=num_neighbors, min_dist=min_dist, metric="cosine")
    reducer = reducer.fit(features)
    return reducer

def plot_umap(reducer, features, labels, fig= None):
    embedding = reducer.transform(features)
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["label"] = labels
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette=sns.color_palette("hsv", len(np.unique(labels))) )
    return fig

def split_data(key ,n_samples, train_ratio=0.6):
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_samples))
    split_index = int(train_ratio * n_samples)
    return shuffled_indices[:split_index], shuffled_indices[split_index:]


def select_vf(last_dim, config):
    print(config.morph)
    if config.net_type == "vf_baseline":
        if config.net_size == "small":
            vf = VelocityField(last_dim= last_dim, **config.morph["vf_small"])
        elif config.net_size == "medium":
            vf = VelocityField(last_dim= last_dim, **config.morph["vf_medium"])
        else:
            vf = VelocityField(last_dim= last_dim, **config.morph["vf_large"])
    elif config.net_type == "vf_adanl":
        if config.net_size == "small":
            vf = vf_test(last_dim= last_dim, **config.morph["adanl_small"])
        elif config.net_size == "medium":
            vf = vf_test(last_dim= last_dim, **config.morph["adanl_medium"])
        else:    
            vf = vf_test(last_dim= last_dim, **config.morph["adanl_large"])
    else:
        raise ValueError("Invalid net type")
    return vf


def calculate_bridge_cost(x_feat, x_paired_feat, x_paired_labels, y_feat, selected_labels):
    n = x_feat.shape[0]
    y_pairwise = pointcloud.PointCloud(y_feat, y_feat, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
    cost_xp = np.zeros((n, selected_labels.shape[0]))
    cost_xy = np.full((n, selected_labels.shape[0]), np.inf)
    for label_ind, label in enumerate(selected_labels):
        print(f"Calculating cost for label {label}")
        x_p_label = x_paired_feat[x_paired_labels == label]
        if x_p_label.shape[0] == 0:
            continue
        distance_to_paired_x = pointcloud.PointCloud(x_feat, x_p_label, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
        cost_xp[:, label_ind] = jnp.min(distance_to_paired_x, axis=1)
        cost_xy = jnp.minimum(cost_xy, cost_xp[:, label_ind][:, jnp.newaxis] + y_pairwise[label_ind, :])
    assert cost_xy.shape == (n, selected_labels.shape[0])
    return cost_xy

def plot_cost(cost_xy, labels, ax):
    sorted_indices = np.argsort(labels)
    sorted_cost = cost_xy[jnp.ix_(sorted_indices, sorted_indices)]
#     sorted_cost = cost_xy[jnp.ix_(sorted_indices)]
    im = ax.imshow(sorted_cost, cmap='viridis')
    ax.axis('off')
     # Adding the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Cost')  # Optional: Add a label to the colorbar


def find_match(rng, tmat, true_labels, selected_labels):
    n,m = tmat.shape
    rngs = jax.random.split(rng, n)
    tgt_ixs = jax.vmap(
        lambda rng, row: jax.random.choice(rng, a=m, p=row, shape=(1,)),
        in_axes=[0, 0],
    )(rngs, tmat)  # (m, k)
    tgt_ixs = tgt_ixs.reshape(-1, *tgt_ixs.shape[2:])  # (m * k, ...)
    tgt = selected_labels[tgt_ixs]
    acc = jnp.mean(selected_labels[tgt_ixs]== true_labels)
    return acc, tgt

def solve_sinkhorn(cost_xy, epsilon, scale_cost = "mean"):
    geom_xy = geometry.Geometry(cost_matrix= cost_xy, epsilon= epsilon, scale_cost = scale_cost)

    linear_ot_solver = sinkhorn.Sinkhorn(max_iterations=6000)
    out = linear_ot_solver(linear_problem.LinearProblem(geom_xy))
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The sinkhorn has converged: {out.converged}")

    transport = out.matrix
    return out, transport

def solve_alignment(rng,
                    x_features,
                    x_labels,
                    y_features,
                    portion,
                    selected_labels,
                    chunk_size = 1000):
    n = x_features.shape[0]
    y_feat= y_features[selected_labels]
    acc_lin_list = []
    tgt_lin = np.zeros((n,), dtype=np.int32)
    
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        x_features_chunk, x_labels_chunk = x_features[i:end_i,:], x_labels[i:end_i]
        n_points = int(x_features_chunk.shape[0]*portion)
        n_points = max(1, n_points)
        
        rng, key1, key2 = jax.random.split(rng, 3)
        selected_points = jax.random.choice(key1, x_features_chunk.shape[0], shape=(n_points,), replace=False)
        x_paired_feat, x_paired_labels=  x_features_chunk[selected_points], x_labels_chunk[selected_points]
        
        cost_xy = calculate_bridge_cost(x_features_chunk, x_paired_feat, x_paired_labels, y_feat, selected_labels)
        
        lin_out, transport = solve_sinkhorn(cost_xy, 5e-3)
        acc_lin, tgt_lin[i:end_i] = find_match(key2, transport, x_labels_chunk, selected_labels)
        acc_lin_list.append(acc_lin)
        
        print("acc_lin", acc_lin)

    return np.mean(acc_lin_list), tgt_lin


def compute_overlap():

    def calculate_percentages(arr):
        flattened_arr = arr.flatten()
        unique_elements, counts = jnp.unique(flattened_arr, return_counts=True)
        total_count = jnp.sum(counts)
        percentages = (counts / total_count) * 100
        return unique_elements, percentages
    # Load data only when needed
    x_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/all_features.npy")
    x_labels = jnp.load("/ptmp/agholamzadeh/imagenet_cache/labels.npy")
    n_classes = [2, 10, 50, 100, 500, 1000]
    all_overlaps = defaultdict(list)
    for n_class in n_classes:
        for seed in range(5):
            print(f"Computing overlap for {n_class} classes with seed {seed}") 
            rng = jax.random.PRNGKey(seed)
            rng, key = jax.random.split(rng)
            selected_labels = jax.random.choice(key, jnp.arange(1000), shape=(n_class,), replace=False)
            selected_x_features = x_features[jnp.isin(x_labels, selected_labels)]
            selected_x_labels = x_labels[jnp.isin(x_labels, selected_labels)]
            n_points = int(min(selected_x_features.shape[0], 1e5))
            
            rng, key = jax.random.split(rng)
            indices = jax.random.choice(key, selected_x_features.shape[0], shape=(n_points,), replace=False)
            x_features_sample = selected_x_features[indices, :]
            x_labels_sample = selected_x_labels[indices]
            
            overlap_dict = defaultdict(list)
            for label in np.array(selected_labels):
                label_features = x_features_sample[x_labels_sample == label]
                cost_matrix = pointcloud.PointCloud(label_features, x_features_sample, cost_fn=costs.Cosine()).cost_matrix
                
                # Approximate nearest neighbors
                dists, neighbours = jax.lax.approx_min_k(cost_matrix, k=10)
                
                # Calculate percentages
                unique_elements, percentages = calculate_percentages(x_labels_sample[neighbours])
                
                # Convert to dictionary and store in overlap_dict
                percentage_dict = dict(zip(np.array(unique_elements), np.array(percentages)))
                overlap_dict[label].append(percentage_dict)     
            overlap= []
            for k,v in overlap_dict.items(): 
                overlap.append(100 - v[0][k]) 
            all_overlaps["seed"].append(seed)
            all_overlaps["n_class"].append(n_class)
            all_overlaps["overlap"].append(np.mean(overlap))      
    return all_overlaps



def morph_features_genot(exp_config):
    print("num of devices:" , jax.local_device_count())
    run = wandb.init(project=exp_config.wandb.project,
                    name=exp_config.wandb.run_name,
                    config=exp_config.to_dict(),
                    group="features",
                    job_type="morph")
    
    config = run.config
    rng = jax.random.PRNGKey(config.seed)
    rng, key = jax.random.split(rng)

    x_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/all_features.npy")
    x_labels = jnp.load("/ptmp/agholamzadeh/imagenet_cache/labels.npy")

    y_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/prompt_features.npy")
    str_labels = jnp.load("/ptmp/agholamzadeh/imagenet_cache/imagenet_classes.npy")
    
    rng, key = jax.random.split(rng)
    selected_labels = jax.random.choice(key, jnp.arange(1000), shape=(config.n_classes,), replace=False)
    selected_labels = jnp.sort(selected_labels)
    selected_labels_str = str_labels[selected_labels]
    selected_x_features = x_features[jnp.isin(x_labels, selected_labels)]
    selected_x_labels = x_labels[jnp.isin(x_labels, selected_labels)]

    rng, key = jax.random.split(rng)
    x_features_train_indices, x_features_test_indices = split_data(key, selected_x_features.shape[0])

    x_features_train, x_features_test = selected_x_features[x_features_train_indices], selected_x_features[x_features_test_indices]
    x_labels_train, x_labels_test = selected_x_labels[x_features_train_indices], selected_x_labels[x_features_test_indices]

    y_features_train, y_features_test = y_features[x_labels_train], y_features[x_labels_test]

    rng, key = jax.random.split(rng)
    if config.n_classes > 5:
        vis_labels = jax.random.choice(key, selected_labels, shape=(5,), replace=False)
    else:
        vis_labels = selected_labels
    print(vis_labels)
    x_features_vis = x_features_test[jnp.isin(x_labels_test, vis_labels)]
    x_labels_vis = x_labels_test[jnp.isin(x_labels_test, vis_labels)]
    y_features_vis = y_features[x_labels_vis]

    x_labels_vis_str = str_labels[x_labels_vis]
    x_labels_train_str = str_labels[x_labels_train]
    x_labels_test_str = str_labels[x_labels_test]


    x_umap = get_umap_transform(x_features_vis)
    x_fig = plot_umap(x_umap, x_features_vis, x_labels_vis_str)
    
    y_umap = get_umap_transform(y_features_vis)
    y_fig = plot_umap(y_umap, y_features_vis, x_labels_vis_str)

    wandb.log({"x_umap": wandb.Image(x_fig)}, commit=False)
    wandb.log({"y_umap": wandb.Image(y_fig)}, commit=False)

    x_paired, y_paired = None, None
    match_fn = None
    if config.align == "true":
        n_paired = int(config.percent_paired * x_features_train.shape[0])
        rng, key = jax.random.split(rng)
        paired_indices = jax.random.choice(key, x_features_train.shape[0], replace=False, shape=(n_paired,))
        x_features_train, x_labels_train =  x_features_train[paired_indices], x_labels_train[paired_indices]
        y_features_train = y_features[x_labels_train]
        is_aligned = True
    elif config.align == "global":
        acc, y_labels_paired = solve_alignment(rng, x_features_train, x_labels_train, y_features, config.percent_paired, selected_labels, chunk_size=int(1e5))
        y_features_train = y_features[y_labels_paired]
        print(f"Accuracy global: {acc}")
        is_aligned = True
    else:
        rng, key = jax.random.split(rng)
        n_paired = int(config.percent_paired * x_features_train.shape[0])
        paired_indices = jax.random.choice(key, x_features_train.shape[0], replace=False, shape=(n_paired,))
        x_paired, x_labes_paired =  x_features_train[paired_indices], x_labels_train[paired_indices]
        y_paired = y_features[x_labes_paired]
        match_fn = get_match_sinhorn(1e-1, config.scale_cost)
        is_aligned = False

    src_data = OTDataExtended(lin= x_features_train, labels = x_labels_train) 
    tgt_data = OTDataExtended(lin = y_features_train, labels =  x_labels_train)
    train_ds = OTDatasetExtended(src_data, tgt_data, is_aligned=is_aligned)


    def callback_func(transform_func, step,
                     x_features_vis,
                     x_labels_vis,
                     train_data,  
                     train_labels,
                     train_target,
                     test_data,
                     test_target,
                     test_labels, 
                     y_umap):
        # fig = plt.figure(figsize=(5, 5))
        transformed_data = transform_func(x_features_vis)
        reduced_transformed = y_umap.transform(transformed_data)
        df = pd.DataFrame(reduced_transformed, columns=["x", "y"])
        df["label"] = x_labels_vis
        # sns.scatterplot(data=df, x="x", y="y", hue="label", palette=sns.color_palette("hsv", len(np.unique(x_labels_vis))) )
        from ott.geometry import pointcloud, costs
        transformed_train = transform_func(train_data)
        cost_matrix_train = pointcloud.PointCloud(train_target, transformed_train, cost_fn=costs.Cosine(), scale_cost="mean").cost_matrix
        paired_train = train_labels[cost_matrix_train.argmin(axis=0)]
        test_transformed = transform_func(test_data)
        cost_matrix_test = pointcloud.PointCloud(test_target, test_transformed, cost_fn=costs.Cosine(), scale_cost="mean").cost_matrix
        paired_test = test_labels[cost_matrix_test.argmin(axis=0)]
        wandb.log({"accuracy_test": (paired_test == test_labels).mean(), 
                   "accuracy_train": (paired_train == train_labels).mean()}, step = step, commit=False)

    rng, key1, key2 = jax.random.split(rng, 3)
    train_example_indices, _ = split_data(key1, x_features_train.shape[0], train_ratio=0.01)
    x_features_train, x_labels_train_str, y_features_train = x_features_train[train_example_indices], x_labels_train_str[train_example_indices], y_features_train[train_example_indices]
    test_example_indices, _ = split_data(key2, x_features_test.shape[0], train_ratio=0.01)
    x_features_test, x_labels_test_str, y_features_test = x_features_test[test_example_indices], x_labels_test_str[test_example_indices], y_features_test[test_example_indices]
    


    callback_func = partial(callback_func,
                            x_features_vis = x_features_vis,
                            x_labels_vis = x_labels_vis_str,
                            train_data = x_features_train, 
                            train_labels = x_labels_train_str,
                            train_target = y_features_train,
                            test_target = y_features_test,
                            test_labels = x_labels_test_str,
                            test_data = x_features_test,
                            y_umap = y_umap)
        

    rng, rng_genot = jax.random.split(rng)
    
    vf = select_vf(y_features_train.shape[1], config)

    save_path = os.path.abspath(os.path.join(exp_config.logdir, "morph", "mlp_vf", wandb.run.id))

    morph = MorphUsingGenot(rng_genot, 
                            config,
                            train_ds,
                            vf,
                            is_aligned=is_aligned,
                            save_path=save_path,
                            callback_func=callback_func,
                            x_paired=x_paired,
                            y_paired=y_paired,
                            match_fn=match_fn)

    morphed_model = morph.morph()


    run.finish()
    print("Finished")

def classification_head(exp_config):

    run = wandb.init(project=exp_config.wandb.project,
                    name="classification",
                    config=exp_config.to_dict(),
                    group="classification",
                    job_type="classification")
    
    config = run.config
    x_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/all_features.npy")
    x_labels = jnp.load("/ptmp/agholamzadeh/imagenet_cache/labels.npy")
    
    rng = jax.random.PRNGKey(config.seed)
    rng, key = jax.random.split(rng)

    x_features_train_indices, x_features_test_indices = split_data(key, x_features.shape[0])

    x_features_train, x_features_test = x_features[x_features_train_indices], x_features[x_features_test_indices]
    x_labels_train, x_labels_test = x_labels[x_features_train_indices], x_labels[x_features_test_indices]

    n_paired = int(config.percent_paired * x_features_train.shape[0])
    rng, key = jax.random.split(rng)
    paired_indices = jax.random.choice(key, x_features_train.shape[0], replace=False, shape=(n_paired,))
    x_features_train, x_labels_train =  x_features_train[paired_indices], x_labels_train[paired_indices]

    model = Classification_head(num_classes=1000)

    def train_step(state, batch, labels):
            def loss_fn(params):
                pred = model.apply(
                    {'params': params}, batch)
                loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, labels))
                metrics = {'loss': loss, 'train_accuracy': jnp.mean(jnp.argmax(pred, axis=-1) == labels)}
                return loss, metrics
            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, metrics
    
    train_step = jax.jit(train_step)

    optimizer = optax.adam(1e-4)
    params = model.init(key, x_features_train[0])["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    rng, key = jax.random.split(rng)
    batch_size = 256
    for epoch in range(300):
        train_batch_metrics = []
        test_metrics = []
        for i in range(0, x_features_train.shape[0], batch_size):
            end = min(i+batch_size, x_features_train.shape[0])
            x_batch = x_features_train[i:end]
            x_labels_batch = x_labels_train[i:end]
            state, train_metrics = train_step(state, x_batch, x_labels_batch)
            train_batch_metrics.append(train_metrics)
        for i in range(0, x_features_test.shape[0], 10000):
            end = min(i+10000, x_features_test.shape[0])
            x_batch = x_features_test[i:end]
            x_labels_batch = x_labels_test[i:end]
            pred = model.apply({'params': state.params}, x_batch)
            test_accuracy = jnp.mean(jnp.argmax(pred, axis=-1) == x_labels_batch)
            test_metrics.append(test_accuracy)
        metrics = {k: np.mean([m[k] for m in train_batch_metrics]) for k in train_batch_metrics[0]}
        metrics["test_accuracy"] = np.mean(test_metrics)
        wandb.log(metrics, step=epoch)
    


def latent_fvae(exp_config):

    run = wandb.init(project=exp_config.wandb.project,
                    name="classification",
                    config=exp_config.to_dict(),
                    group="classification",
                    job_type="classification")
    
    config = run.config
    x_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/all_features.npy")
    x_labels = jnp.load("/ptmp/agholamzadeh/imagenet_cache/labels.npy")
    y_features = jnp.load("/ptmp/agholamzadeh/imagenet_cache/prompt_features.npy")

    rng = jax.random.PRNGKey(config.seed)
    rng, key = jax.random.split(rng)

    x_features_train_indices, x_features_test_indices = split_data(key, x_features.shape[0])

    x_features_train, x_features_test = x_features[x_features_train_indices], x_features[x_features_test_indices]
    x_labels_train, x_labels_test = x_labels[x_features_train_indices], x_labels[x_features_test_indices]

    n_paired = int(config.percent_paired * x_features_train.shape[0])
    rng, key = jax.random.split(rng)
    paired_indices = jax.random.choice(key, x_features_train.shape[0], replace=False, shape=(n_paired,))
    x_features_train, x_labels_train =  x_features_train[paired_indices], x_labels_train[paired_indices]
    fused_ds = FusedDsBuilder(x_features_train, y_features[x_labels_train], x_labels_train, exp_config)

    model_name = "latentfvae"
    model_conf = exp_config[model_name]
    fused_model = get_model(model_name, model_conf["model"])
    fused_model.train(fused_ds, exp_config)