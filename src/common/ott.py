from functools import partial

import jax
import jax.numpy as jnp

from ott.geometry import pointcloud, costs, geometry, graph
from ott.problems.quadratic import quadratic_problem
from ott.problems.linear import linear_problem

from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from flax.linen import initializers

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

LinTerm = Tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                 Optional[jnp.ndarray]]


he_normal_init = initializers.he_normal()



def get_cost_fn(type: str) -> costs.CostFn:
     if type == "sqeuclidean":
          return costs.SqEuclidean()
     elif type == "euclidean":
          return costs.Euclidean()
     elif type == "cosine":
          return costs.Cosine()

def sinkhorn_match_cost(
  ot_solver: Any,
  epsilon: float,
  tau_a: float,
  tau_b: float,
  scale_cost: Any,
  ) -> Callable:
    @partial(
    jax.jit,
    static_argnames=["ot_solver", "epsilon", "tau_a", "tau_b", "scale_cost"],
    )

    def match_pairs(
      x_indices: jnp.ndarray,
      y_indices: jnp.ndarray,
      cost: jnp.ndarray, 
      ot_solver: Any,
      tau_a: float,
      tau_b: float,
      epsilon: float,
      scale_cost: Any,
      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cost_matrix = cost[x_indices.flatten(), :][:, y_indices.flatten()]
        geom = geometry.Geometry(cost_matrix=cost_matrix, epsilon=epsilon, scale_cost=scale_cost)
        out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
        transport = out.matrix
        return transport, out
    
    return jax.tree_util.Partial(
    match_pairs,
    ot_solver=ot_solver,
    epsilon=epsilon,
    tau_a=tau_a,
    tau_b=tau_b,
    scale_cost=scale_cost)


def gromov_match_fn(
    ot_solver: Any,
    tau_a: float,
    tau_b: float,
    scale_cost: Any,
    fused_penalty: float,
    ) -> Callable:
         
        @partial(
            jax.jit,
            static_argnames=[
            "ot_solver",
            "scale_cost",
            "fused_penalty",
            "tau_a",
            "tau_b",
            ],
        )
        def match_pairs(
            x_indices: jnp.ndarray,
            y_indices: jnp.ndarray,
            cost_xx: jnp.ndarray, 
            cost_yy: jnp.ndarray, 
            cost_xy: Optional[jnp.ndarray],
            ot_solver: Any,
            tau_a: float,
            tau_b: float,
            scale_cost,
            fused_penalty: float,
            ) -> Tuple[jnp.ndarray, Tuple[jnp.array, jnp.array]]:

            cost_matrix_xx = cost_xx[x_indices.flatten(), :][:, x_indices.flatten()]
            cost_matrix_yy = cost_yy[y_indices.flatten(), :][:, y_indices.flatten()]

            geom_xx = geometry.Geometry(cost_matrix=cost_matrix_xx, scale_cost=scale_cost)
            geom_yy = geometry.Geometry(cost_matrix=cost_matrix_yy, scale_cost=scale_cost)
           
            if cost_matrix_yy is None:
                geom_xy = None
            else:
                cost_matrix_xy = cost_xy[x_indices.flatten(), :][:, y_indices.flatten()]
                geom_xy = geometry.Geometry(cost_matrix=cost_matrix_xy, scale_cost=scale_cost)

            prob = quadratic_problem.QuadraticProblem(
            geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b)
            
            out = ot_solver(prob)
            transport = out.matrix
            # a, b = transport.sum(axis=1), out.matrix.sum(axis=0)

            return transport, out
        
        return jax.tree_util.Partial(
            match_pairs,
            ot_solver=ot_solver,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            fused_penalty=fused_penalty)


def get_sinkhorn_match_fn(
  ot_solver: Any,
  epsilon: float,
  cost_fn: str,
  tau_a: float,
  tau_b: float,
  scale_cost: Any,
  ) -> Callable:
    @partial(
    jax.jit,
    static_argnames=["ot_solver", "epsilon", "cost_fn", "scale_cost", "tau_a", "tau_b", "k_samples_per_x"],
    )

    def match_pairs(
      x: jnp.ndarray,
      y: jnp.ndarray,
      ot_solver: Any,
      tau_a: float,
      tau_b: float,
      epsilon: float,
      cost_fn: str,
      scale_cost: Any,
      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        geom = pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn)
        out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
        transport = out.matrix

        return transport, out
    
    return jax.tree_util.Partial(
    match_pairs,
    ot_solver=ot_solver,
    epsilon=epsilon,
    cost_fn=cost_fn,
    tau_a=tau_a,
    tau_b=tau_b,
    scale_cost=scale_cost)
  
def get_gromov_match_fn(
    ot_solver: Any,
    cost_fn: Optional[costs.CostFn],
    tau_a: float,
    tau_b: float,
    scale_cost: Any,
    fused_penalty: float,
    ) -> Callable:
         
        @partial(
            jax.jit,
            static_argnames=[
            "ot_solver",
            "cost_fn",
            "scale_cost",
            "fused_penalty",
            "tau_a",
            "tau_b",
            ],
        )
        def match_pairs(
            xx: jnp.ndarray,
            yy: jnp.ndarray,
            x: Optional[jnp.ndarray],
            y: Optional[jnp.ndarray],
            ot_solver: Any,
            tau_a: float,
            tau_b: float,
            cost_fn: Optional[costs.CostFn],
            scale_cost,
            fused_penalty: float,
            ) -> Tuple[jnp.ndarray, Tuple[jnp.array, jnp.array]]:

            geom_xx = pointcloud.PointCloud(x=xx, y=xx, cost_fn=cost_fn, scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=yy, y=yy, cost_fn=cost_fn, scale_cost=scale_cost)

            if x is None:
                geom_xy = None
            else:
                geom_xy = pointcloud.PointCloud(x, y, cost_fn=cost_fn, scale_cost=scale_cost)

            prob = quadratic_problem.QuadraticProblem(
            geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b)
            
            out = ot_solver(prob)
            transport = out.matrix
            # a, b = transport.sum(axis=1), out.matrix.sum(axis=0)

            return transport, out
        
        return jax.tree_util.Partial(
            match_pairs,
            ot_solver=ot_solver,
            cost_fn=cost_fn,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            fused_penalty=fused_penalty)

def _get_match_fn_graph(
  ot_solver: Any,
  epsilon: float,
  k_neighbors: int,
  tau_a: float,
  tau_b: float,
  scale_cost: Any,
  fused_penalty: float,
  **kwargs,
  ) -> Callable:
  
  def get_nearest_neighbors( X: jnp.ndarray, Y: jnp.ndarray, k: int = 30) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    concat = jnp.concatenate((X, Y), axis=0)
    pairwise_euclidean_distances = pointcloud.PointCloud(concat, concat).cost_matrix
    distances, indices = jax.lax.approx_min_k(
    pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True)
    return distances, indices
  
  def create_cost_matrix(X: jnp.array, Y: jnp.array, k_neighbors: int, **kwargs: Any) -> jnp.array:
    distances, indices = get_nearest_neighbors(X, Y, k_neighbors)
    a = jnp.zeros((len(X) + len(Y), len(X) + len(Y)))
    adj_matrix = a.at[
    jnp.repeat(jnp.arange(len(X) + len(Y)), repeats=k_neighbors).flatten(), indices.flatten()
    ].set(distances.flatten())
    return graph.Graph.from_graph(adj_matrix, normalize=kwargs.pop("normalize", True), **kwargs).cost_matrix[
    : len(X), len(X) :
    ]
  @partial(
    jax.jit,
    static_argnames=[
    "ot_solver",
    "problem_type",
    "epsilon",
    "k_neighbors",
    "tau_a",
    "tau_b",
    "k_samples_per_x",
    "fused_penalty",
    "split_dim",
    ],
  )

  def match_pairs(
    xx: Optional[jnp.ndarray],
    yy: Optional[jnp.ndarray],
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    ot_solver: Any,
    epsilon: float,
    tau_a: float,
    tau_b: float,
    fused_penalty: float,
    k_neighbors: int,
    **kwargs,
    ) -> Tuple[jnp.array, jnp.array, jnp.ndarray, jnp.ndarray]:

    if xx is None :
      cm = create_cost_matrix(x, y, k_neighbors, **kwargs)
      geom = geometry.Geometry(cost_matrix=cm, epsilon=epsilon, scale_cost=scale_cost)
      out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
    else:
      cm_xx = create_cost_matrix(xx, xx, k_neighbors, **kwargs)
      cm_yy = create_cost_matrix(yy, yy, k_neighbors, **kwargs)
      geom_xx = geometry.Geometry(cost_matrix=cm_xx, epsilon=epsilon, scale_cost=scale_cost)
      geom_yy = geometry.Geometry(cost_matrix=cm_yy, epsilon=epsilon, scale_cost=scale_cost)
      if x is None :
        geom_xy = None
      else:
        cm_xy = create_cost_matrix(x, y, k_neighbors, **kwargs)
        geom_xy = geometry.Geometry(cost_matrix=cm_xy, epsilon=epsilon, scale_cost=scale_cost)
      prob = quadratic_problem.QuadraticProblem(
      geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b)
      out = ot_solver(prob)
      transport = out.matrix

      return transport, out
    
  return jax.tree_util.Partial(
    match_pairs,
    ot_solver=ot_solver,
    epsilon=epsilon,
    k_neighbors=k_neighbors,
    tau_a=tau_a,
    tau_b=tau_b,
    fused_penalty=fused_penalty,
    **kwargs,
    )
      
  

      
def get_solver(config) -> Any:
    problem_type = config["problem_type"]
    if problem_type == "linear":
        config = config["sinkhorn"]
        return sinkhorn.Sinkhorn(**config)
    elif problem_type == "quadratic":
        sinkhorn_conf = config["sinkhorn"]
        linear_ot_solver = sinkhorn.Sinkhorn(**sinkhorn_conf)
        gw_config = config["gw"]
        return gromov_wasserstein.GromovWasserstein(linear_ot_solver=linear_ot_solver,**gw_config)
    else:
        raise ValueError(f"Unknown solver type: {type}")


@jax.tree_util.register_pytree_node_class
class WassersteinCost(costs.CostFn):
    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        return wasserstein_distance(x, y)

def wasserstein_distance(dist_x, dist_y):
    mean_x, logvar_x = jnp.split(dist_x, [dist_x.shape[0] // 2])
    mean_y, logvar_y = jnp.split(dist_y, [dist_y.shape[0] // 2])

    var_x = jnp.exp(logvar_x)
    var_y = jnp.exp(logvar_y)

    mean_x, mean_y = pad_to_match_length(mean_x, mean_y, 0)
    var_x, var_y = pad_to_match_length(var_x, var_y, 1e-12)

    # Compute the squared Euclidean distance between the means
    mean_diff_squared = jnp.sum((mean_x - mean_y) ** 2)

    # Compute the element-wise square roots of the variances
    sqrt_var_x = jnp.sqrt(var_x)
    sqrt_var_y = jnp.sqrt(var_y)

    # Compute the trace term for diagonal matrices
    trace_term = jnp.sum(var_x + var_y - 2 * sqrt_var_x * sqrt_var_y)

    wasserstein_distance_squared = mean_diff_squared + trace_term

    # Compute the Wasserstein distance
    return wasserstein_distance_squared

def pad_to_match_length(array1, array2, value):
    len1 = array1.shape[0]
    len2 = array2.shape[0]

    if len1 > len2:
        padding = len1 - len2
        array2_padded = jnp.pad(array2, (0, padding), mode='constant', constant_values=value)
        return array1, array2_padded
    elif len2 > len1:
        padding = len2 - len1
        array1_padded = jnp.pad(array1, (0, padding), mode='constant', constant_values=value)
        return array1_padded, array2
    else:
        return array1, array2
    
def semi_supervised_cost(pairwise_distances_x, pairwise_distances_y, x_paired_indices, y_paired_indices, chunk_size=256):
    distance_to_paired_x = pairwise_distances_x[:, x_paired_indices]
    distance_to_paired_y = pairwise_distances_y[:, y_paired_indices]
    # Initialize the final cost matrix with large values
    cost_xy = jnp.full((pairwise_distances_x.shape[0], pairwise_distances_y.shape[0]), jnp.inf)
    # Process in chunks
    for i in range(0, pairwise_distances_x.shape[0], chunk_size):
        end_i = min(i + chunk_size, pairwise_distances_x.shape[0])
        for j in range(0, pairwise_distances_y.shape[0], chunk_size):
            end_j = min(j + chunk_size, pairwise_distances_y.shape[0])
            
            # Compute the combined distances for the current chunk
            distance_xy_chunk = distance_to_paired_x[i:end_i, None, :] + distance_to_paired_y[None, j:end_j, :]
            cost_xy_chunk = jnp.min(distance_xy_chunk, axis=-1)
            
            # Update the corresponding part of the final cost matrix
            cost_xy = cost_xy.at[i:end_i, j:end_j].set(cost_xy_chunk)

    return cost_xy

def get_pairwise_cost(x_samples, y_samples, cost_fn = "cosine", scale_cost="mean"):
  if cost_fn == "cosine":
     cost_fn = costs.Cosine()
  elif cost_fn == "euclidean":
     cost_fn = costs.Euclidean()
  elif cost_fn == "wasserstein":
     cost_fn = WassersteinCost()
  else:
     raise ValueError(f"Unknown cost function: {cost_fn}")
  return pointcloud.PointCloud(x_samples, y_samples, scale_cost= scale_cost, cost_fn= cost_fn).cost_matrix



def get_nearest_neighbors( X: jnp.ndarray, k: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]: 
    pairwise_distances = pointcloud.PointCloud(X, X, cost_fn=costs.Cosine()).cost_matrix
    distances, indices = jax.lax.approx_min_k(
    pairwise_distances, k=k, recall_target=0.95, aggregate_to_topk=True)
    return distances, indices



def mutual_knn(feats_A, feats_B, k=10):
    """
    Computes the mutual KNN accuracy.

    Args:
        feats_A: A JAX array of shape N x feat_dim
        feats_B: A JAX array of shape N x feat_dim

    Returns:
        A float representing the mutual KNN accuracy
    """
    _, knn_A = get_nearest_neighbors(feats_A, k)
    _, knn_B = get_nearest_neighbors(feats_B, k)

    n = knn_A.shape[0]
    topk = knn_A.shape[1]

    # Create binary masks for knn_A and knn_B
    lvm_mask = jnp.zeros((n, n), dtype=jnp.float32).at[jnp.arange(n)[:, None], knn_A].set(1.0)
    llm_mask = jnp.zeros((n, n), dtype=jnp.float32).at[jnp.arange(n)[:, None], knn_B].set(1.0)
    
    acc = jnp.sum(lvm_mask * llm_mask, axis=1) / topk
    
    return jnp.mean(acc).item()
    
def compute_score(x_feats, y_feats, k=10):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a JAX array of shape N x L x D
        y_feats: a JAX array of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    best_alignment_indices = None
    best_alignment_score = 0

    for i in range(-1, x_feats.shape[1]):
        x = x_feats.reshape(x_feats.shape[0], -1) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.reshape(y_feats.shape[0], -1) if j == -1 else y_feats[:, j, :]
                                            
            score = mutual_knn(x, y, k=k)

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)
    
    return best_alignment_score, best_alignment_indices