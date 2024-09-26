import functools

import flax.jax_utils
import flax.jax_utils
import flax.jax_utils
import flax.jax_utils
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import flax 

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from ott.neural.methods.flows import genot
from ott.neural.methods.flows import dynamics
from src.experimental.diffusion_transformer import DiT

from ott.neural.methods.flows import genot
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils
from ott import utils
import wandb
import orbax.checkpoint as ocp


from flax.training import orbax_utils, train_state

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

LinTerm = Tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                 Optional[jnp.ndarray]]

DataMatchFn = Union[Callable[[LinTerm], jnp.ndarray], Callable[[QuadTerm],
                                                               jnp.ndarray]]


class GenotModified(genot.GENOT):
    def __init__(
        self,
        vf: DiT,
        flow: dynamics.BaseFlow,
        *,
        source_dim: int,
        target_dim: int,
        condition_dim: Optional[int] = None,
        time_sampler: Callable[[jax.Array, int],
                                jnp.ndarray] = solver_utils.uniform_sampler,
        latent_noise_fn: Optional[Callable[[jax.Array, Tuple[int, ...]],
                                            jnp.ndarray]] = None,
        latent_match_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                            jnp.ndarray]] = None,
        n_samples_per_src: int = 1,
        data_match_fn: Optional[DataMatchFn] = None,
        callback: Optional[Callable] = None,
        callback_freq: int = 500,
        log_freq: int = 30,
        use_true_alignment: bool = False,
        save_path: str,
        **kwargs: Any):
        
        self.vf = vf
        self.flow = flow
        self.data_match_fn = data_match_fn
        self.time_sampler = time_sampler
        
        if latent_noise_fn is None:
            latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim, cov=5e-1)
        
        self.latent_noise_fn = latent_noise_fn
        self.latent_match_fn = latent_match_fn
        self.n_samples_per_src = n_samples_per_src

        self.vf_state_repl = flax.jax_utils.replicate(self.vf.create_train_state(
            input_dim=target_dim,
            condition_dim=source_dim + (condition_dim or 0),
            **kwargs))
        

        self.step_fn = self._get_step_fn()
        self.callback = callback
        self.callbak_freq = callback_freq
        self.log_freq = log_freq
        self.use_true_alignment = use_true_alignment

        self.n_devices = jax.local_device_count()
        
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=500,
            max_to_keep=2,
            # other options
        )
        self.checkpointer = ocp.CheckpointManager(directory=save_path, options= options, item_names=["state", "metadata"])

    
    def _get_step_fn(self) -> Callable:

      def step_fn(
          vf_state: train_state.TrainState,
          rng: jax.Array,
          time: jnp.ndarray,
          source: jnp.ndarray,
          target: jnp.ndarray,
          latent: jnp.ndarray,
          source_conditions: Optional[jnp.ndarray],
      ):

        def loss_fn(
            params: jnp.ndarray, time: jnp.ndarray, source: jnp.ndarray,
            target: jnp.ndarray, latent: jnp.ndarray,
            source_conditions: Optional[jnp.ndarray], rng: jax.Array
        ) -> jnp.ndarray:
          rng_flow, rng_dropout = jax.random.split(rng, 2)
          x_t = self.flow.compute_xt(rng_flow, time, latent, target)
          if source_conditions is None:
            cond = source
          else:
            cond = jnp.concatenate([source, source_conditions], axis=-1)

          v_t = vf_state.apply_fn({"params": params},
                                  time,
                                  x_t,
                                  cond,
                                  rngs={"dropout": rng_dropout})
          u_t = self.flow.compute_ut(time, latent, target)

          return jnp.mean((v_t - u_t) ** 2)

        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(
            vf_state.params, time, source, target, latent, source_conditions, rng
        )
        # Average gradients across all devices
        grads = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), grads)
        loss = jax.lax.pmean(loss, axis_name='batch')

        new_vf_state = vf_state.apply_gradients(grads=grads)

        return loss, new_vf_state, grads

      return jax.pmap(step_fn, axis_name='batch', donate_argnums=(0,)) 

    def __call__(
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      n_iters: int,
      rng: Optional[jax.Array] = None,
      x_paired: Optional[jnp.ndarray] = None,
      y_paired: Optional[jnp.ndarray] = None
      ) -> Dict[str, List[float]]:
      """Train the GENOT model.

      Args:
        loader: Data loader returning a dictionary with possible keys
          `src_lin`, `tgt_lin`, `src_quad`, `tgt_quad`, `src_conditions`.
        n_iters: Number of iterations to train the model.
        rng: Random key for seeding.

      Returns:
        Training logs.
      """

      def prepare_data(batch: Dict[str, jnp.ndarray]):

        src_lin, src_quad, src_labels = batch.get("src_lin"), batch.get("src_quad"), batch.get("src_labels")
        tgt_lin, tgt_quad, tgt_labels = batch.get("tgt_lin"), batch.get("tgt_quad"), batch.get("tgt_labels")
        arrs = src_lin, tgt_lin
        if src_quad is None and tgt_quad is None:  # lin
          src, tgt = src_lin, tgt_lin
        #   if self.use_precomputed_cost:               
        #     arrs = batch.get("src_ix"), batch.get("tgt_ix"), self.cost_xy
        elif src_lin is None and tgt_lin is None:  # quad
          src, tgt = src_quad, tgt_quad
        #   if self.use_precomputed_cost:               
        #     arrs = batch.get("src_ix"), batch.get("tgt_ix"), self.cost_xx, self.cost_yy, self.cost_xy
        elif all(
            arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)
        ):  # fused quad
          src, tgt = src_quad, tgt_quad
        #   if self.use_precomputed_cost:               
        #     arrs = batch.get("src_ix"), batch.get("tgt_ix"), self.cost_xx, self.cost_yy, self.cost_xy
        else:
          raise RuntimeError("Cannot infer OT problem type from data.")
        return (src, batch.get("src_condition"), tgt), arrs, (src_labels, tgt_labels)

      rng = utils.default_prng_key(rng)
      for step, batch in enumerate(loader):
        rng = jax.random.split(rng, 5)
        rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

        batch = jtu.tree_map(jnp.asarray, batch)
        (src, src_cond, tgt), matching_data, labels = prepare_data(batch)
        n = src.shape[0]
        time = self.time_sampler(rng_time, n * self.n_samples_per_src)
        latent = self.latent_noise_fn(rng_noise, (n, self.n_samples_per_src))

        if self.use_true_alignment:
           tmat = jnp.eye(n)/n
        else:
          tmat, matching_out = self.data_match_fn(xx= matching_data[0], yy= matching_data[1],x_paired = x_paired, y_paired=y_paired )  # (n, m)        
        
        src_ixs, tgt_ixs = solver_utils.sample_conditional(  # (n, k), (m, k)
            rng_resample,
            tmat,
            k=self.n_samples_per_src,
        )

        src, tgt = src[src_ixs], tgt[tgt_ixs]  # (n, k, ...),  # (m, k, ...)
        if src_cond is not None:
          src_cond = src_cond[src_ixs]

        if self.latent_match_fn is not None:
          src, src_cond, tgt = self._match_latent(rng, src, src_cond, latent, tgt)

        src = src.reshape(-1, *src.shape[2:])  # (n * k, ...)
        tgt = tgt.reshape(-1, *tgt.shape[2:])  # (m * k, ...)
        latent = latent.reshape(-1, *latent.shape[2:])
        if src_cond is not None:
          src_cond = src_cond.reshape(-1, *src_cond.shape[2:])

        rng_step_fn = flax.jax_utils.replicate(rng_step_fn)

        ## reshapes to data parallel
        # Split the input data across devices (along the first axis)
        src = jnp.reshape(src, (self.n_devices, -1, *src.shape[1:]))
        tgt = jnp.reshape(tgt, (self.n_devices, -1, *tgt.shape[1:]))
        latent = jnp.reshape(latent, (self.n_devices, -1, *latent.shape[1:]))
        if src_cond is not None:
            src_cond = jnp.reshape(src_cond, (self.n_devices, -1, *src_cond.shape[1:]))
        time = jnp.reshape(time, (self.n_devices, -1, *time.shape[1:]))

        loss, self.vf_state_repl, grads = self.step_fn(
          self.vf_state_repl, rng_step_fn,time, src, tgt, latent, src_cond
        )

        self.save(step)
        
        if step % self.log_freq == 0:
          metrics = {"genot_loss": flax.jax_utils.unreplicate(loss),
                      "grad_norm": optax.global_norm(grads), "step": step}
          
          if not self.use_true_alignment:
            print("Matching")
            print(matching_out)
            print(matching_out.converged)
            metrics["converged"] = float(matching_out.converged)
            metrics["n_iters_matching"] = matching_out.n_iters
            if hasattr(matching_out, "reg_gw_cost"):
              metrics["gw_cost"] = matching_out.reg_gw_cost
            else:
              metrics["sinkhorn_cost"] = matching_out.reg_ot_cost
          wandb.log(metrics , step = step)
          
        if self.callback is not None and step % self.callbak_freq == 0:
          self.vf_state = flax.jax_utils.unreplicate(self.vf_state_repl)
          self.callback(transform_func = self.transport, step = step)

        if step >= n_iters:
          break
      return

    def save(self, step):
      """Save the model to a file.

      Args:
        path: Path to save the model.
      """
      metadata = {"wandb_run_id": wandb.run.id, "config": wandb.config.as_dict()}
      self.checkpointer.save(step=step,
                             args= ocp.args.Composite(
                               state = ocp.args.StandardSave(flax.jax_utils.unreplicate(self.vf_state_repl)),
                               metadata = ocp.args.JsonSave(metadata)
                             ))

    
    # def load(self, path: str): 
    #   """Load the model from a file
    #    Args:
    #        path: Path to load the model from.
    #   """

    #   ckpt = self.orbax_checkpointer.restore(path)
    #   self.vf_state = train_state.TrainState.create(
    #     apply_fn=self.vf.apply, params=ckpt["vf_state"]["params"], tx=optax.adam(learning_rate=1e-4)
    # )


def _multivariate_normal(
    rng: jax.Array,
    shape: Tuple[int, ...],
    dim: int,
    mean: float = 0.0,
    cov: float = 1.0
) -> jnp.ndarray:
  mean = jnp.full(dim, fill_value=mean)
  cov = jnp.diag(jnp.full(dim, fill_value=cov))
  return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)
