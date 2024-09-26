from collections import defaultdict
import jax
import jax.numpy as jnp
import os 
import orbax.checkpoint as ocp
from src.common.utils import WandbLogger, plot_latent_umap2
from src.common.morph import MorphUsingGenot
from src.common.data import FusedDsBuilder, MnistDsBuilder, OTDataExtended, OTDatasetExtended 
from src.common.models import get_model
from src.common.train import get_trainer
from src.common.ott import semi_supervised_cost
import wandb
from ott.geometry import pointcloud, costs, graph, geometry
from ott.problems.quadratic import quadratic_problem
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
import ott.solvers.utils as solver_utils
from mvlearn.embed import KMCCA
import pandas as pd
from configs.vf import get_config
from src.experimental.vf import vf_test
from src.experimental.vf_modified import VelocityField

from sklearn.metrics import roc_auc_score
import numpy as np
from functools import partial
from src.experiments.match_helpers import get_match_sinhorn, get_match_gw


def find_match_accuracy(rng_sample, transport, labels):
    src_ixs, tgt_ixs = solver_utils.sample_conditional(rng_sample, transport)
    labels = jnp.argmax(labels, axis = 1)
    src, tgt = labels[src_ixs], labels[tgt_ixs]
    src = src.reshape(-1, *src.shape[2:])  # (n * k, ...)
    tgt = tgt.reshape(-1, *tgt.shape[2:])  # (m * k, ...)
    acc = jnp.mean(src == tgt)
    return acc

def get_kcca_cost(x_samples, y_samples, x_paired_indices, y_paired_indices):
    n_components = min([x_samples.shape[1], y_samples.shape[1]])
    kcca = KMCCA(n_components=n_components, kernel= "rbf", regs = .01)
    x_samples_paired, y_samples_paired = x_samples[x_paired_indices], y_samples[y_paired_indices]
    kcca.fit([x_samples_paired, y_samples_paired])
    x_samples_transformed, y_samples_transformed = kcca.transform([x_samples, y_samples])
    cost_xy = pointcloud.PointCloud(x_samples_transformed, y_samples_transformed, scale_cost = "mean").cost_matrix
    return cost_xy

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



def get_knn_cost(pairwise_distances_x, pairwise_distances_y, x_paired_indices, y_paired_indices, k=15, batch_size=500):
    n_x = pairwise_distances_x.shape[0]
    n_y = pairwise_distances_y.shape[0]

    max_distance = jnp.max(jnp.concatenate([pairwise_distances_x, pairwise_distances_y]))
    pairwise_distances_xy = 10 * max_distance * jnp.ones([n_x, n_y], dtype=jnp.float16)
    pairwise_distances_xy = pairwise_distances_xy.at[x_paired_indices, y_paired_indices].set(1e-20)
    
    first_cost_part = jnp.concatenate((pairwise_distances_x, pairwise_distances_xy), axis=1)
    second_cost_part = jnp.concatenate((pairwise_distances_xy.T, pairwise_distances_y), axis=1)
    pairwise_distances = jnp.concatenate((first_cost_part, second_cost_part), axis=0)
    
    # Initialize empty arrays to hold the results
    all_distances = []
    all_indices = []

    # Compute KNN in batches
    for i in range(0, pairwise_distances.shape[0], batch_size):
        batch_distances = pairwise_distances[i:i+batch_size, :]
        distances, indices = jax.lax.approx_min_k(
            batch_distances, k=k, recall_target=0.95, aggregate_to_topk=True)
        
        all_distances.append(distances)
        all_indices.append(indices)

    # Concatenate the batch results
    distances = jnp.concatenate(all_distances, axis=0)
    indices = jnp.concatenate(all_indices, axis=0)

    a = jnp.zeros((n_x + n_y, n_x + n_y), dtype=jnp.float16)
    adj_matrix = a.at[
        jnp.repeat(jnp.arange(n_x + n_y), repeats=k).flatten(), indices.flatten()
    ].set(1)
    
    cost = graph.Graph.from_graph(adj_matrix, normalize=True).cost_matrix[
        :n_x, n_x:]
    
    return cost

def solve_sinkhorn(cost_xy, epsilon, scale_cost = "mean"):
    geom_xy = geometry.Geometry(cost_matrix= cost_xy, epsilon= epsilon, scale_cost = scale_cost)

    linear_ot_solver = sinkhorn.Sinkhorn(max_iterations=6000)
    out = linear_ot_solver(linear_problem.LinearProblem(geom_xy))
    transport = out.matrix
    return out, transport

def solve_gw(cost_xx, cost_yy, cost_xy, epsilon, scale_cost ,fused_penalty = 1):
    geom_xx = geometry.Geometry(cost_matrix= cost_xx, epsilon= epsilon, scale_cost= scale_cost)
    geom_yy = geometry.Geometry(cost_matrix= cost_yy, epsilon= epsilon, scale_cost= scale_cost)
    if cost_xy is None:
        geom_xy = None
    else:
        geom_xy = geometry.Geometry(cost_matrix= cost_xy, epsilon= epsilon, scale_cost= scale_cost)

    prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty)
    linear_ot_solver = sinkhorn.Sinkhorn(max_iterations=6000)
    solver = gromov_wasserstein.GromovWasserstein(linear_ot_solver = linear_ot_solver, store_inner_errors=True, epsilon= epsilon, max_iterations= 200)

    out = solver(prob)
    transport = out.matrix
    has_converged = bool(out.linear_convergence[out.n_iters - 1])
    print(f"{out.n_iters} outer iterations were needed.")
    print(f"The last Sinkhorn iteration has converged: {has_converged}")
    print(f"The outer loop of Gromov Wasserstein has converged: {out.converged}")
    print(f"The final regularized GW cost is: {out.reg_gw_cost:.3f}")
    return out, transport

def calc_mse_per_class(true_recon, recon, test_labels, n_classes = 10):
    mse = []
    print("True recon shape", true_recon.shape, "recon shape", recon.shape, "test_labels shape", test_labels)
    for i in range(n_classes):
        if jnp.any(test_labels == i) == False:
            continue
        mean_true = jnp.mean(true_recon[test_labels==i], axis =0)
        print("Mean true shape", mean_true.shape)
        mean_pred = jnp.mean(recon[test_labels==i], axis =0)
        print("Mean pred shape", mean_pred.shape)
        mse.append(np.mean((mean_true - mean_pred) ** 2))
        print("MSE", mse[-1])
    return np.mean(mse)


def restore_model(path):
    abspath = os.path.abspath(path)
    ckpt_mgr =ocp.CheckpointManager(abspath, item_names=["state", "metadata"])
    step = ckpt_mgr.latest_step()
    print(f"Restoring from {abspath} at step {step}")
    return ckpt_mgr.restore(step, args=ocp.args.Composite(
            state=ocp.args.StandardRestore(), metadata=ocp.args.JsonRestore()))

def bridge_cost(x_feat, y_feat, x_paired_feat, y_paired_feat, chunk_size= 500):
    n, m = x_feat.shape[0], y_feat.shape[0]
    distance_to_paired_x = pointcloud.PointCloud(x_feat, x_paired_feat, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
    distance_to_paired_y = pointcloud.PointCloud(y_feat, y_paired_feat, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
        # Process in chunks
    cost_xy = jnp.full((n, m), jnp.inf, dtype=jnp.float16)
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(0, m, chunk_size):
            end_j = min(j + chunk_size, m)
            # Compute the combined distances for the current chunk
            distance_xy_chunk = distance_to_paired_x[i:end_i, None, :] + distance_to_paired_y[None, j:end_j, :]
            cost_xy_chunk = jnp.min(distance_xy_chunk, axis=-1)
            
            # Update the corresponding part of the final cost matrix
            cost_xy = cost_xy.at[i:end_i, j:end_j].set(cost_xy_chunk)

    return cost_xy


def model_apply_dict(model, model_state, data, **kwargs):
    batch_size = 1000
    n = data.shape[0]
    results = []
    for i in range(0, n, batch_size):
        s = slice(i, min(i+batch_size, n))
        batch = data[s]
        results.append(model.apply({"params": model_state["params"]}, batch, **kwargs))

    return jnp.concatenate(results, axis = 0)
def apply_transform(transform_func, data):
    batch_size = 1000
    n = data.shape[0]
    results = []
    for i in range(0, n, batch_size):
        s = slice(i, min(i+batch_size, n))
        batch = data[s]
        results.append(transform_func(batch))

    return jnp.concatenate(results, axis = 0)

def to_text_callback(transform_func, step, train_data, train_labels, test_data, test_labels, sn_model, sn_state):
    transformed_train = apply_transform(transform_func, train_data)
    transformed_test = apply_transform(transform_func, test_data)
    recon_pred = model_apply_dict(sn_model, sn_state, transformed_train,method = "generate")
    recon_test = model_apply_dict(sn_model, sn_state, transformed_test, method = "generate")
    train_acc = jnp.mean(jnp.argmax(recon_pred, axis = 1) == jnp.argmax(train_labels, axis = 1))
    test_acc = jnp.mean(jnp.argmax(recon_test, axis = 1) == jnp.argmax(test_labels, axis = 1))
    print("Train acc", train_labels)
    print("Test acc", test_labels)
    # train_auc = roc_auc_score(train_labels, recon_pred, multi_class='ovr')
    # test_auc = roc_auc_score(test_labels, recon_test, multi_class='ovr')
    wandb.log({"train_acc": train_acc, "test_acc": test_acc,
                # "train_auc": train_auc, "test_auc": test_auc,
                  "step": step})

def to_img_callback(transform_func, step, train_data, train_input ,train_labels, test_data, test_input ,test_labels, fn_model, fn_state):
    transformed_train = apply_transform(transform_func, train_data)
    transformed_test = apply_transform(transform_func, test_data)
    recon_pred = model_apply_dict(fn_model, fn_state, transformed_train,method = "generate")
    recon_test = model_apply_dict(fn_model, fn_state, transformed_test, method = "generate")
    train_mse = calc_mse_per_class(train_input, recon_pred, train_labels, n_classes= 10)
    test_mse = calc_mse_per_class(test_input, recon_test, test_labels, n_classes= 10)
    wandb.log({"train_mse": train_mse, "test_mse": test_mse, "step": step})


def train_vae_mnist(exp_config):
    rng = jax.random.PRNGKey(exp_config.seed)

    run = wandb.init(project= exp_config.wandb.project,
                group= "mnist", 
                job_type= "train",
                name = "train_mnist",
                config= exp_config)

    rng, fn_ds_rng, sn_ds_rng = jax.random.split(rng, 3)
    rng , fn_trainer_rng, sn_trainer_rng = jax.random.split(rng, 3)


    logdir = exp_config.logdir

    model_name = "FN"
    model_conf = exp_config[model_name]
    fn_model = get_model(model_name, model_conf["model"])
    fn_ds = MnistDsBuilder(exp_config, fn_ds_rng, "image")
    trainer = get_trainer(fn_trainer_rng, "vae", model_conf, fn_model, fn_ds, logdir)
    trainer.create_functions()
    trainer.train_model()
    fn_state = trainer.state


    model_name = "SN"
    model_conf = exp_config[model_name]
    sn_model = get_model(model_name, model_conf["model"])
    sn_ds = MnistDsBuilder(exp_config, sn_ds_rng, "text")
    trainer = get_trainer(sn_trainer_rng, "vae", model_conf, sn_model, sn_ds, logdir)
    trainer.create_functions()
    trainer.train_model()
    sn_state = trainer.state



    # ds = MnistDsBuilder(exp_config, rng, "fused")

    # save_path = os.path.join(exp_config.save_path, wandb.run.id)
    # os.makedirs(save_path, exist_ok=True)
    # rng = process_and_save(ds, "val", fn_model, fn_state, sn_model, sn_state, rng, save_path, "val_res.npz")
    # rng = process_and_save(ds, "test", fn_model, fn_state, sn_model, sn_state, rng, save_path, "test_res.npz")


    run.finish()

def train_fused_vae(exp_config):


    rng = jax.random.PRNGKey(exp_config.seed)
    rng, fused_ds_rng, fused_trainer_rng, rng_callback = jax.random.split(rng, 4)

    run = wandb.init(project= exp_config.wandb.project,
                group= "mnist", 
                job_type= "train",
                name = "end_to_end_fvae",
                config= exp_config.to_dict())

    exp_config = run.config  
    logdir = exp_config.logdir
    model_name = "FusedVAE"
    model_conf = exp_config[model_name]
    fused_model = get_model(model_name, model_conf["model"])
    fused_ds = MnistDsBuilder(exp_config, fused_ds_rng, "fused")

    num_val_samples = 1000
    (img_sample, text_sample), labels =next(fused_ds.get_n_samples("test", num_val_samples))

    def callback(model, params, img_sample, text_sample, labels,rng):
        recon_x1, recon_x2, recon_y1, recon_y2, mean_x, logvar_x, mean_y, logvar_y = model.apply(
            {'params': params}, img_sample, text_sample, rng)
        recon_x2 = jax.nn.sigmoid(recon_x2)
        recon_y1 = jax.nn.softmax(recon_y1)
        images = jnp.concat((img_sample[:8], recon_x2[:8]))
        image_grid = WandbLogger.image_grid(images)
        acc_y1 = jnp.mean(jnp.argmax(recon_y1, axis = 1) == jnp.argmax(text_sample, axis = 1))
        test_auc = roc_auc_score(jnp.argmax(text_sample, axis = 1), recon_y1, multi_class='ovr')
        mse_x2 = calc_mse_per_class(img_sample, recon_x2, labels, n_classes=10)
        wandb.log({"test_acc": acc_y1, "test_mse": mse_x2, "test_auc": test_auc,"image_grid": wandb.Image(image_grid)},commit=False)

    callback_func = partial(callback, img_sample = img_sample, text_sample = text_sample, labels = labels, rng = rng_callback)
    
    trainer = get_trainer(rng= fused_trainer_rng, train_type="fused_vae", confs=model_conf, model=fused_model, ds_builder=fused_ds, logdir=logdir, callback=callback_func)
    trainer.create_functions(type=exp_config.train_type)
    trainer.train_model(exp_config.n_paired)    

    
def latent_fused_vae(exp_config):
    rng = jax.random.PRNGKey(exp_config.seed)
    rng, fused_ds_rng, fused_trainer_rng, rng_callback = jax.random.split(rng, 4)

    run = wandb.init(project= exp_config.wandb.project,
                group= "mnist", 
                job_type= "train",
                name = "latent_fvae",
                config= exp_config.to_dict())

    exp_config = wandb.config
    
    rng = jax.random.PRNGKey(exp_config.seed)
    

    rng, ds_key = jax.random.split(rng)
    rng, fn_key, sn_key = jax.random.split(rng, 3)
    ds = MnistDsBuilder(exp_config, ds_key, "fused")
    n_paired = exp_config.n_paired

    (val_img, val_text), val_label = next(ds.get_n_samples("val", n_paired))
    (test_img, test_text), test_label = next(ds.get_n_samples("test", 1000))
    
    name = "FN"
    fn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
    fn_model_conf = fn_restore["metadata"]["model_conf"]
    fn_model = get_model(name, fn_model_conf)
    fn_state = fn_restore["state"]
    img_feat_train = model_apply_dict(fn_model, fn_state, val_img, z_rng = fn_key, method = "encode_and_sample")   
    img_feat_test = model_apply_dict(fn_model, fn_state, test_img, z_rng = fn_key, method = "encode_and_sample") 

    name = "SN"
    sn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
    sn_model_conf = sn_restore["metadata"]["model_conf"]
    sn_model = get_model(name, sn_model_conf)
    sn_state = sn_restore["state"]
    text_feat_train = model_apply_dict(sn_model, sn_state, val_text, z_rng = sn_key, method = "encode_and_sample")
    text_feat_test = model_apply_dict(sn_model, sn_state, test_text, z_rng = sn_key, method = "encode_and_sample")

    print("Img feat shape", img_feat_train.shape, "Text feat shape", text_feat_train.shape)

    def callback(model, params, img_sample, text_sample, img_feat, text_feat, labels, rng, fn_model, fn_state, sn_model, sn_state):
        recon_x1, recon_x2, recon_y1, recon_y2, mean_x, logvar_x, mean_y, logvar_y = model.apply(
            {'params': params}, img_feat, text_feat, rng)
        
        
        recon_img_1 = model_apply_dict(fn_model, fn_state, recon_x1, method = "generate")
        recon_img_2 = model_apply_dict(fn_model, fn_state, recon_x2, method = "generate")

        recon_text_1 = model_apply_dict(sn_model, sn_state, recon_y1, method = "generate")
        recon_text_2 = model_apply_dict(sn_model, sn_state, recon_y2, method = "generate")

        images = jnp.concat((img_sample[:8], recon_img_1[:8], recon_img_2[:8]))
        image_grid = WandbLogger.image_grid(images)

        print("Recon text shape", recon_text_1[:8])
        print("Text sample shape", text_sample[:8])

        test_acc_y1 = jnp.mean(jnp.argmax(recon_text_1, axis = 1) == jnp.argmax(text_sample, axis = 1))
        test_auc_1 = roc_auc_score(jnp.argmax(text_sample, axis = 1), recon_text_1, multi_class='ovr')

        test_acc_y2 = jnp.mean(jnp.argmax(recon_text_2, axis = 1) == jnp.argmax(text_sample, axis = 1))
        test_auc_2 = roc_auc_score(jnp.argmax(text_sample, axis = 1), recon_text_2, multi_class='ovr')

        test_mse_x1 = calc_mse_per_class(img_sample, recon_img_1, labels, n_classes=10)
        test_mse_x2 = calc_mse_per_class(img_sample, recon_img_2, labels, n_classes=10)

        z_x = model.apply({"params": params}, img_feat[:5000], rng, 1, method= "encode_and_sample")
        z_y = model.apply({"params": params}, text_feat[:5000], rng, 2, method= "encode_and_sample")

        fig = plot_latent_umap2(z_x, z_y, labels[:5000])

        wandb.log({
                    "test_acc_y1": test_acc_y1, "test_auc_1": test_auc_1,
                    "test_acc_y2": test_acc_y2, "test_auc_2": test_auc_2,
                    "test_mse_x1": test_mse_x1, "test_mse_x2": test_mse_x2,
                    "latent_umap": wandb.Image(fig),
                   "image_grid": wandb.Image(image_grid)},commit=False)

    callback_func = partial(callback, img_sample= test_img, text_sample = test_text,
                            img_feat = img_feat_test, text_feat = text_feat_test, labels = test_label,
                             rng = rng_callback, fn_model = fn_model, fn_state = fn_state, sn_model = sn_model, sn_state = sn_state)

    logdir = exp_config.logdir
    model_name = "latentfvae"
    model_conf = exp_config[model_name]
    fused_model = get_model(model_name, model_conf["model"])

    fused_ds = FusedDsBuilder(img_feat_train, text_feat_train, val_label, exp_config)

    trainer = get_trainer(rng= fused_trainer_rng, train_type="fused_vae", confs=model_conf, model=fused_model, ds_builder=fused_ds, logdir=logdir, callback=callback_func)
    trainer.create_functions("fused")
    trainer.train_model(-1)    





def discrete_moph(exp_config):
    
    wandb.init(project= exp_config.wandb.project,
                group= "mnist", 
                job_type= "morph_discrete",
                name = "mnist-morph-discrete",
                config= exp_config)
    
    for seed in range(5):
        rng = jax.random.PRNGKey(seed)

        rng, ds_key = jax.random.split(rng)
        rng, fn_key, sn_key = jax.random.split(rng, 3)
        
        ds = MnistDsBuilder(exp_config, ds_key, "fused")
        n_repeats = 2
        (img, text), label = next(ds.get_n_samples("val", n_repeats*ds.get_num_examples("val"),n_repeats = n_repeats))
        print("img_shape", img.shape, "text_shape", text.shape, "label_shape", label.shape)
        n_data = img.shape[0]

        name = "FN"
        fn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
        fn_model_conf = fn_restore["metadata"]["model_conf"]
        fn_model = get_model(name, fn_model_conf)
        fn_state = fn_restore["state"]
        img_feat = model_apply_dict(fn_model, fn_state, img, z_rng = fn_key, method = "encode_and_sample")    

        name = "SN"
        sn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
        sn_model_conf = sn_restore["metadata"]["model_conf"]
        sn_model = get_model(name, sn_model_conf)
        sn_state = sn_restore["state"]
        text_feat = model_apply_dict(sn_model, sn_state, text, z_rng = sn_key, method = "encode_and_sample")

        morph_config = exp_config.morph

        cost_xx = pointcloud.PointCloud(img_feat, img_feat, epsilon=morph_config.epsilon, cost_fn= costs.Cosine(),
                                        scale_cost=morph_config.scale_cost).cost_matrix
        
        cost_yy = pointcloud.PointCloud(text_feat, text_feat, epsilon=morph_config.epsilon,cost_fn= costs.Cosine(),
                                        scale_cost=morph_config.scale_cost).cost_matrix
            

        n_paired_set = [int(n_data*i) for i in np.linspace(0, 1, 11)]
        wandb.define_metric("n_paired")
        fuse_costs = ["knn", "kcca", "bridge"]
        problem_types = ["sinkhorn", "gw"]
        # fuse_costs= ["bridge"]
        # problem_types = ["sinkhorn"]
        res = defaultdict(list)
        for n_paired in n_paired_set:
            for fuse_cost in fuse_costs:
                for problem_type in problem_types:
                    if problem_type == "sinkhorn" and n_paired == 0:
                        continue
                    rng, key, key_sample = jax.random.split(rng,3)
                    if n_paired != 0:
                        paired_indices = jax.random.choice(key, cost_xx.shape[0], replace=False, shape=(n_paired,))
                        if fuse_cost == "kcca":
                            cost_xy = get_kcca_cost(img_feat, text_feat, paired_indices, paired_indices)
                        elif fuse_cost == "knn":
                            cost_xy = get_knn_cost(cost_xx, cost_yy, paired_indices, paired_indices, k = 15)
                        elif fuse_cost == "bridge":
                            cost_xy = semi_supervised_cost(cost_xx, cost_yy, paired_indices, paired_indices, chunk_size=256)
                        else:
                            raise ValueError("Invalid fuse cost")
                    else:
                        cost_xy = None
                    if problem_type == "sinkhorn":
                        lin_out, lin_transport = solve_sinkhorn(cost_xy, morph_config.epsilon, morph_config.scale_cost)
                        match_acc = find_match_accuracy(key_sample, lin_transport, text)
                        print("Match accuracy: ", match_acc, "n_paired: ", n_paired)
                        del lin_out, lin_transport, cost_xy
                    else:
                        gw_out, gw_transport = solve_gw(cost_xx, cost_yy, cost_xy, morph_config.epsilon, morph_config.scale_cost, morph_config.fused_penalty)
                        match_acc = find_match_accuracy(key_sample, gw_transport, text)
                        wandb.log({"gw_match_accuracy": match_acc, "n_paired": n_paired})
                    res["n_paired"].append(n_paired)
                    res["fuse_cost"].append(fuse_cost)
                    res["problem_type"].append(problem_type)
                    res["match_accuracy"].append(match_acc)
                    
        
        df = pd.DataFrame(res)
        df["seed"] = exp_config.seed
        save_path = os.path.join(exp_config.logdir, "morph_discrete", wandb.run.id)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "results_{}.csv".format(seed)))
    wandb.finish()    
    print("Finished")


def morph_genot(exp_config):

    wandb.init(project= exp_config.wandb.project,
            group= "mnist", 
            job_type= "morph_genot",
            name = "morph_genot",
            config= exp_config.to_dict())
    
    exp_config = wandb.config
    
    rng = jax.random.PRNGKey(exp_config.seed)
    

    rng, ds_key = jax.random.split(rng)
    rng, fn_key, sn_key = jax.random.split(rng, 3)
        
    ds = MnistDsBuilder(exp_config, ds_key, "fused")
    (val_img, val_text), val_label = next(ds.get_n_samples("val", ds.get_num_examples("val")))
    (test_img, test_text), test_label = next(ds.get_n_samples("test", ds.get_num_examples("test")))

    if "n_classes" in exp_config:
        rng, key = jax.random.split(rng)
        selected_classes = jax.random.choice(key, 10, shape=(exp_config.n_classes,), replace=False)
        print("Selected classes", selected_classes)
        val_label , test_label= jnp.argmax(val_text, axis = 1), jnp.argmax(test_text, axis = 1)
        val_img, val_text, val_label = val_img[jnp.isin(val_label, selected_classes)], val_text[jnp.isin(val_label, selected_classes)], val_label[jnp.isin(val_label, selected_classes)]
        test_img, test_text, test_label = test_img[jnp.isin(test_label, selected_classes)], test_text[jnp.isin(test_label, selected_classes)], test_label[jnp.isin(test_label, selected_classes)]
        print("Val img shape", val_img.shape, "Val text shape", val_text.shape, "Val label shape", val_label.shape)
    
    
    name = "FN"
    fn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
    fn_model_conf = fn_restore["metadata"]["model_conf"]
    fn_model = get_model(name, fn_model_conf)
    fn_state = fn_restore["state"]
    img_feat_train = model_apply_dict(fn_model, fn_state, val_img, z_rng = fn_key, method = "encode_and_sample")   
    img_feat_test = model_apply_dict(fn_model, fn_state, test_img, z_rng = fn_key, method = "encode_and_sample") 

    name = "SN"
    sn_restore = restore_model(os.path.join(exp_config.logdir, "train", exp_config.load_id, name))
    sn_model_conf = sn_restore["metadata"]["model_conf"]
    sn_model = get_model(name, sn_model_conf)
    sn_state = sn_restore["state"]
    text_feat_train = model_apply_dict(sn_model, sn_state, val_text, z_rng = sn_key, method = "encode_and_sample")
    text_feat_test = model_apply_dict(sn_model, sn_state, test_text, z_rng = sn_key, method = "encode_and_sample")
    if "n_classes" in exp_config:
        if exp_config.n_classes < 10:
            n_paired = img_feat_train.shape[0]
        else:
            n_paired = exp_config.n_paired
    else:
        n_paired = exp_config.n_paired

    rng, key = jax.random.split(rng)
    paired_indices = jax.random.choice(key, img_feat_train.shape[0], replace=False, shape=(n_paired,))
    x_paired, y_paired = None, None
    match_fn = None
    if exp_config.align == "true":
        img_train, text_train = img_feat_train[paired_indices], text_feat_train[paired_indices]
        is_aligned = True
    elif exp_config.align == "global":
        cost_xx = pointcloud.PointCloud(img_feat_train, img_feat_train, epsilon=exp_config.epsilon, cost_fn= costs.Cosine()).cost_matrix 
        cost_yy = pointcloud.PointCloud(text_feat_train, text_feat_train, epsilon=exp_config.epsilon,cost_fn= costs.Cosine()).cost_matrix
        cost_xy = semi_supervised_cost(cost_xx, cost_yy, paired_indices, paired_indices)
        if exp_config.problem_type == "sinkhorn":
            lin_out, lin_transport = solve_sinkhorn(cost_xy, exp_config.epsilon, exp_config.scale_cost)
            src_train_indices, tgt_train_indices = solver_utils.sample_conditional(rng, lin_transport)
            del lin_out, lin_transport
        else:
            gw_out, gw_transport = solve_gw(cost_xx, cost_yy, cost_xy, exp_config.epsilon, exp_config.scale_cost, exp_config.fused_penalty)
            src_train_indices, tgt_train_indices = solver_utils.sample_conditional(rng, gw_transport)
            del gw_out, gw_transport
        img_train, text_train = img_feat_train[src_train_indices], text_feat_train[tgt_train_indices]
        del cost_xx, cost_yy, cost_xy
        is_aligned = True
    elif exp_config.align == "local":
        img_train, text_train = img_feat_train, text_feat_train
        img_paired, text_paired = img_feat_train[paired_indices], text_feat_train[paired_indices]
        if exp_config.target == "text":
            x_paired, y_paired = img_paired, text_paired
        else:
            x_paired, y_paired = text_paired, img_paired
        if exp_config.problem_type == "sinkhorn":
            print("Using sinkhorn")
            print("Epsilon: ", exp_config.epsilon)
            print("Scale cost: ", exp_config.scale_cost)
            match_fn = get_match_sinhorn(exp_config.epsilon, exp_config.scale_cost)
        else:
            match_fn = get_match_gw(exp_config.epsilon, exp_config.scale_cost)
        is_aligned = False
    else:
        raise ValueError("Invalid alignment type")
    
    if exp_config.target == "text":
        src_data = OTDataExtended(lin= img_train)
        tgt_data = OTDataExtended(lin =text_train)
        callback_func = partial(to_text_callback, 
                                train_data = img_feat_train, 
                                train_labels = val_text,
                                test_data = img_feat_test,
                                test_labels = test_text, 
                                sn_model = sn_model,
                                sn_state = sn_state)
    else:
        src_data = OTDataExtended(lin= text_train)
        tgt_data = OTDataExtended(lin = img_train)
        callback_func = partial(to_img_callback, 
                                train_data = text_feat_train, 
                                train_input = val_img,
                                train_labels = val_label,
                                test_data = text_feat_test,
                                test_input = test_img,
                                test_labels = test_label,
                                fn_model = fn_model,
                                fn_state = fn_state)

    train_ds = OTDatasetExtended(src_data, tgt_data, is_aligned=is_aligned)    

    rng, rng_genot = jax.random.split(rng)
    
    save_path = os.path.abspath(os.path.join(exp_config.logdir, "morph", "genot", wandb.run.id))

    # vf = vf_test(last_dim=train_ds.tgt_dim, **exp_config.vf)

    vf = select_vf(train_ds.tgt_dim, exp_config)

    
    morph_genot = MorphUsingGenot(rng_genot, 
                        exp_config,
                        train_ds,
                        vf,
                        is_aligned=is_aligned,
                        save_path=save_path,
                        callback_func=callback_func,
                        match_fn=match_fn,
                        x_paired = x_paired,
                        y_paired = y_paired)

    morph_genot.morph()
    wandb.finish()

