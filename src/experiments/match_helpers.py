from functools import partial
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud, costs, geometry
from ott.problems.quadratic import quadratic_problem
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal



@jax.jit
def bridge_cost(x_feat, y_feat, x_paired_feat, y_paired_feat, chunk_size= 512):
    distance_to_paired_x = pointcloud.PointCloud(x_feat, x_paired_feat, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
    distance_to_paired_y = pointcloud.PointCloud(y_paired_feat, y_feat, scale_cost = "mean", cost_fn=costs.Cosine()).cost_matrix
    cost_xy = jnp.min(distance_to_paired_x[:, :, jnp.newaxis] + distance_to_paired_y[jnp.newaxis, :, :], axis=1)
    return cost_xy

def get_match_sinhorn(
    epsilon: float,
    scale_cost: Any,
    ) -> Callable:
         
        @partial(
            jax.jit,
            static_argnames=[
            "epsilon",
            "scale_cost",
            ],
        )
        def match_pairs(
            xx: jnp.ndarray,
            yy: jnp.ndarray,
            x_paired: jnp.ndarray,
            y_paired: jnp.ndarray,
            epsilon: float,
            scale_cost: Any,
            ) -> Tuple[jnp.ndarray, Tuple[jnp.array, jnp.array]]:

            cost_xy =  bridge_cost(xx, yy, x_paired, y_paired)
            geom_xy = geometry.Geometry(cost_matrix= cost_xy, epsilon= epsilon, scale_cost = scale_cost)
            linear_ot_solver = sinkhorn.Sinkhorn(max_iterations=6000)
            out = linear_ot_solver(linear_problem.LinearProblem(geom_xy))
            transport = out.matrix

            return transport, out
        return jax.tree_util.Partial(
            match_pairs,
            epsilon = epsilon,
            scale_cost = scale_cost,)


def get_match_gw(
    epsilon: float,
    scale_cost: Any,
    ) -> Callable:
         
        @partial(
            jax.jit,
            static_argnames=[
            "epsilon",
            "scale_cost",
            ],
        )
        def match_pairs(
            xx: jnp.ndarray,
            yy: jnp.ndarray,
            x_paired: jnp.ndarray,
            y_paired: jnp.ndarray,
            epsilon: float,
            scale_cost: Any,
            ) -> Tuple[jnp.ndarray, Tuple[jnp.array, jnp.array]]:

            cost_xy =  bridge_cost(xx, yy, x_paired, y_paired)
            geom_xy = geometry.Geometry(cost_matrix= cost_xy, epsilon= epsilon, scale_cost = scale_cost)
            geom_xx = pointcloud.PointCloud(x=xx, y=xx, cost_fn=costs.Cosine(), scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=yy, y=yy, cost_fn=costs.Cosine(), scale_cost=scale_cost)
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=1)
            linear_ot_solver = sinkhorn.Sinkhorn(max_iterations=6000)
            solver = gromov_wasserstein.GromovWasserstein(linear_ot_solver = linear_ot_solver, store_inner_errors=True, epsilon= epsilon, max_iterations= 200)
            out = solver(prob)
            transport = out.matrix
            return transport, out
        
        return jax.tree_util.Partial(
            match_pairs,
            epsilon = epsilon,
            scale_cost = scale_cost,)
