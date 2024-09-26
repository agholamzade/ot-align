import jax
import jax.numpy as jnp
from sklearn.calibration import label_binarize
import yaml
from flax import linen as nn
import optax
import wandb
import numpy as np
import math
from PIL import Image
import copy
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import orbax
from ml_collections import ConfigDict

def read_config(conf_path, config):
    with open(conf_path,'r') as f:
            recursive_update(config, yaml.safe_load(f))
    return config

def get_optimization(train_conf):
    if train_conf["opt"] == "adam":
        return optax.adam(train_conf["learning_rate"])
    elif train_conf["opt"] == "sgd":
        return optax.sgd(train_conf["learning_rate"])
    else:
        raise Exception("Unknown Optimization Function")
    

def recursive_update(original_dict, new_dict):
    for key, value in new_dict.items():
        if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
            recursive_update(original_dict[key], value)
        else:
            original_dict[key] = value
    
def update_default_dict(default_dict, new_dict):
    dict_copy =  copy.deepcopy(default_dict)
    recursive_update(dict_copy, new_dict)
    return dict_copy


def create_wandb_run(file_conf, group, job_type):
    run = wandb.init(project=file_conf["wandb"]["project"],
                group= group, 
                job_type= job_type,
                config= file_conf,
                name = file_conf["wandb"]["run_name"])
    return run


@jax.vmap
def kl_divergence_normal(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
  logits = nn.log_sigmoid(logits)
  return -jnp.sum(
      labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
  )

@jax.vmap
def accuracy(predictions, labels):
    """
    Computes the accuracy of softmax predictions.
    """
    predicted_class = jnp.argmax(predictions)
    true_class = jnp.argmax(labels)
    return jnp.equal(predicted_class, true_class)


@jax.vmap
def l2_loss(x):
    return (x ** 2).mean()


def find_params_by_node_name(params, node_name):
    from typing import Iterable

    def _is_leaf_fun(x):
        if isinstance(x, Iterable) and jax.tree_util.all_leaves(x.values()):
            return True
        return False

    def _get_key_finder(key):
        def _finder(x):
            value = x.get(key)
            return None if value is None else {key: value}
        return _finder

    filtered_params = jax.tree_map(_get_key_finder(node_name), params, is_leaf=_is_leaf_fun)
    filtered_params = [x for x in jax.tree_leaves(filtered_params) if x is not None]

    return filtered_params

@jax.vmap
def multi_softmax_cross_entropy(splits, labels):
    return jnp.sum(jnp.array([optax.softmax_cross_entropy_with_integer_labels(logits, label) for logits, label in zip(splits, labels)]))

@jax.vmap
def mean_squared_error(predictions, labels):
    return optax.squared_error(predictions, labels).mean()

def get_loss_function(type):
    if type == "kl_divergence_normal":
        return kl_divergence_normal
    elif type == "binary_cross_entropy":
        return binary_cross_entropy_with_logits
    elif type == "mean_squared_error":
        return mean_squared_error
    elif type == "softmax_cross_entropy":
        return optax.softmax_cross_entropy
    elif type == "multi_softmax_cross_entropy":
        return multi_softmax_cross_entropy
    else:
        raise Exception("Unknown Activation Function")


def svm_accuracy(X,y,model):
    model.fit(X, y)
    return model.score(X, y)


def plot_latent_umap(X, labels, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    fig = plt.figure()
    fit = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    u = fit.fit_transform(X)
    sns.scatterplot(x = u[:,0],y=  u[:,1], hue=labels, palette = "tab10", s = 5) 
    plt.title('UMAP, n_neighbors = {}, min_dist = {}, metric = {}'.format(n_neighbors, min_dist, metric))
    return fig

def plot_latent_umap2(x,y, labels, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    fig = plt.figure()
    stacked = jnp.vstack([x,y])
    fit = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    u = fit.fit_transform(stacked)
    u1 = u[:x.shape[0]]
    u2 = u[x.shape[0]:]
    print(u1.shape)
    print(labels.shape)
    sns.scatterplot(x = u1[:,0],y=  u1[:,1], hue=labels, palette = "tab10", s = 5,) 
    sns.scatterplot(x = u2[:,0],y=  u2[:,1], hue=labels, palette = "tab10", s = 5, marker = "x") 
    plt.title('UMAP, n_neighbors = {}, min_dist = {}, metric = {}'.format(n_neighbors, min_dist, metric))
    return fig


def plot_multiclass_roc(y_true, y_score, classes):
    """
    Plots the ROC curve for a multi-class classification problem.

    Parameters:
    y_true (array-like): True labels.
    y_score (array-like):  estimates of the positive class 
    classes (array-like): List of all classes.

    Returns:
    None
    """
    # Binarize the output
    y_true = label_binarize(y_true, classes=classes)
    n_classes = y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for each class
    fig = plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
    #          label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
    plt.legend(loc="lower right")
    return fig

class WandbLogger:

    def __init__(self, name, every_n_epochs = 20):
        self.name = name  
        self.every_n_epochs = every_n_epochs  
        wandb.define_metric(f'{self.name}/epoch')
        wandb.define_metric(f'{self.name}/*', step_metric = f'{self.name}/epoch')

    def log_metrics(self, data):
        dict = {f'{self.name}/{k}': v for k, v in data.items()}
        wandb.log(dict)

   
    @staticmethod
    def accumulate_metrics(prefix, metrics):
        metrics = jax.device_get(metrics)
        return {
            prefix+" "+k: np.mean([metric[k] for metric in metrics])
            for k in metrics[0]
        }
    
    @staticmethod
    def image_grid(images ,nrow=8, padding=2, pad_value=0.0):
        """Make a grid of images."""
            
        ndarray = jnp.asarray(images)

        if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
            ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

        # make the mini-batch of images into a grid
        nmaps = ndarray.shape[0]
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = (
            int(ndarray.shape[1] + padding),
            int(ndarray.shape[2] + padding),
        )
        num_channels = ndarray.shape[3]
        grid = jnp.full(
            (height * ymaps + padding, width * xmaps + padding, num_channels),
            pad_value,
        ).astype(jnp.float32)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                grid = grid.at[
                    y * height + padding : (y + 1) * height,
                    x * width + padding : (x + 1) * width,
                ].set(ndarray[k])
                k = k + 1

        ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
        return Image.fromarray(ndarr.copy())


    
    def log_image(self, images, label, epoch):
        if epoch % self.every_n_epochs == 0: 
            image_grid = WandbLogger.image_grid(images)
            wandb.log({self.name + "/" + label: wandb.Image(image_grid)})
    
    def log_plot_image(self, fig, label):
        wandb.log({self.name + "/" + label: wandb.Image(fig)}, commit=False)


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


def plot_swiss_spiral(
    swiss_roll, spiral, colormap_angles_swiss_roll, colormap_angles_spiral
):
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(spiral[:, 0], spiral[:, 1], c=colormap_angles_spiral)
    ax.grid()
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.view_init(7, -80)
    ax.scatter(
        swiss_roll[:, 0],
        swiss_roll[:, 1],
        swiss_roll[:, 2],
        c=colormap_angles_swiss_roll,
    )
    ax.set_adjustable("box")
    return fig

def restore_orbax(rel_path, sub_folder, net_name):
    abspath = os.path.abspath(rel_path)
    saved_path = os.path.join(abspath, sub_folder, net_name)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(saved_path)

def to_wandb_config(d: ConfigDict, parent_key: str = '', sep: str ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def create_wandb_run(project, group, job_type, run_name, config= None):
    
    run = wandb.init(project= project,
                group= group, 
                job_type= job_type,
                name = run_name,
                config= config)
    return run

def cosine_similarity(x, y):
    dot_product = jnp.dot(x, y)
    norm_x = jnp.linalg.norm(x)
    norm_y = jnp.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

# Step 2: Calculate the cosine distance
def cosine_distance(x, y):
    return 1.0 - cosine_similarity(x, y)

# Step 3: Compute the mean cosine distance between two arrays
def mean_cosine_distance(arr1, arr2):
    distances = jax.vmap(cosine_distance)(arr1, arr2)
    return jnp.mean(distances)


def get_param_schedule(schedule_conf):
    if schedule_conf["type"] == "const":
        return optax.constant_schedule(schedule_conf["value"])
    elif schedule_conf["type"] == "linear":
        return optax.linear_schedule(schedule_conf["init_value"], schedule_conf["end_value"], 
                                    schedule_conf["n_epochs"], schedule_conf["transition_begin"])
    else:
        raise Exception("Unknown Schedule Type")
    


# Example function to calculate gradient norm
def calculate_grad_norm(grads):
    from jax.tree_util import tree_map
    
    total_norm = jnp.sqrt(sum(tree_map(lambda x: jnp.sum(x**2), grads).values()))
    return total_norm