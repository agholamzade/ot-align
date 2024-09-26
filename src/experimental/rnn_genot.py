
import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import wandb
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax

import matplotlib.pyplot as plt

import diffrax
from flax.training import train_state, orbax_utils

from ott import utils
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils
from rnn_vf import VelocityFieldRNN

__all__ = ["GENOT"]

LinTerm = Tuple[jnp.ndarray, jnp.ndarray]
QuadTerm = Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                 Optional[jnp.ndarray]]
DataMatchFn = Union[Callable[[LinTerm], jnp.ndarray], Callable[[QuadTerm],
                                                               jnp.ndarray]]


class RNNGENOT:
  def __init__(
      self,
      vf: VelocityFieldRNN,
      flow: dynamics.BaseFlow,
      data_match_fn: DataMatchFn,
      *,
      source_dim: int,
      target_dim: int,
      sequence_length: int,
      condition_dim: Optional[int] = None,
      time_sampler: Callable[[jax.Array, int],
                             jnp.ndarray] = solver_utils.uniform_sampler,
      latent_noise_fn: Optional[Callable[[jax.Array, Tuple[int, ...]],
                                         jnp.ndarray]] = None,
      latent_match_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray],
                                         jnp.ndarray]] = None,
      n_samples_per_src: int = 1,
      eval_data: jnp.ndarray = None, 
      **kwargs: Any,
  ):
    self.vf = vf
    self.flow = flow
    self.data_match_fn = data_match_fn
    self.time_sampler = time_sampler
    if latent_noise_fn is None:
      latent_noise_fn = functools.partial(_multivariate_normal, dim=target_dim)
    self.latent_noise_fn = latent_noise_fn
    self.latent_match_fn = latent_match_fn
    self.n_samples_per_src = n_samples_per_src
    self.sequence_length = sequence_length
    self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    self.eval_data = eval_data

    self.beta_schedule = optax.linear_schedule(1.0, 0.0, 1000, 200)
    self.vf_state = self.vf.create_train_state(
        src_dim=source_dim,
        tgt_dim=target_dim,
        **kwargs
    )
    self.step_fn = self._get_step_fn()

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        rng: jax.Array,
        vf_state: train_state.TrainState,
        time: jnp.ndarray,
        source: jnp.ndarray,
        target: jnp.ndarray,
        latent: jnp.ndarray,
        beta: float = 1.0,
    ):

      def loss_fn(
          params: jnp.ndarray, time: jnp.ndarray, source: jnp.ndarray,
          target: jnp.ndarray, latent: jnp.ndarray , beta,rng: jax.Array
      ) -> jnp.ndarray:
        # time: (SL, BS), source: (SL, BS,SD), target: (SL, BS,TD), latent: (SL, BS, TD)
        print("time: ", time.shape, "source: ", source.shape, "target: ", target.shape, "latent: ", latent.shape)


        seq_len = source.shape[0]
        carry = jnp.zeros((source.shape[1], self.vf.rnn_hidden_dim))

        last_mean, last_logvar = jnp.zeros_like(latent), jnp.zeros_like(latent)
        matching_losses = []
        kl_losses1 = []
        kl_losses2 = []
        losses = []
        for i in range(seq_len):
          rng_flow, rng_dropout, rng_gru = jax.random.split(rng, 3)
          t = time[i]
          src = source[i]
          tgt = target[i]

          x_t = self.flow.compute_xt(rng_flow, t, latent, tgt)
          vf_out = self.vf_state.apply_fn({"params": params}, t, x_t, src, rngs={"dropout": rng_dropout}, method = "vf_output")

          carry, (mean, logvar, latent) = self.vf_state.apply_fn({"params": params}, src, latent, carry, rngs={"gru": rng_gru}, method = "rnn_output")

          u_t = self.flow.compute_ut(time, latent, target)
          matching_loss = jnp.mean(jnp.square(vf_out - u_t))
          # kl_loss1 = _kl_divergence_diag(mean, logvar, jnp.zeros_like(mean), jnp.zeros_like(logvar))
          # kl_loss2 = _kl_divergence_diag(mean, logvar, last_mean, last_logvar)
          # losses.append((matching_loss + beta * (kl_loss1) -  (1-beta) * kl_loss2))
          losses.append(matching_loss)
          last_mean, last_logvar = mean, logvar
          # kl_losses1.append(kl_loss1)
          # kl_losses2.append(kl_loss2)
          matching_losses.append(matching_loss)
        loss = jnp.mean(jnp.array(losses))
        metrics = {
            "loss": loss,
            "matching_loss": jnp.mean(jnp.array(matching_losses)),
            # "kl_loss1": jnp.mean(jnp.array(kl_losses1)),
            # "kl_loss2": jnp.mean(jnp.array(kl_losses2)),
        }
        return loss, metrics
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, metrics), grads = grad_fn(vf_state.params, time, source, target, latent, beta, rng)
      return loss, vf_state.apply_gradients(grads=grads), metrics

    return step_fn

  def __call__(
      self,
      loader: Iterable[Dict[str, np.ndarray]],
      n_iters: int,
      rng: Optional[jax.Array] = None
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

    def prepare_data(
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray],
               Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray],
                     Optional[jnp.ndarray]]]:
      src_lin, src_quad = batch.get("src_lin"), batch.get("src_quad")
      tgt_lin, tgt_quad = batch.get("tgt_lin"), batch.get("tgt_quad")

      if src_quad is None and tgt_quad is None:  # lin
        src, tgt = src_lin, tgt_lin
      elif src_lin is None and tgt_lin is None:  # quad
        src, tgt = src_quad, tgt_quad
      elif all(
          arr is not None for arr in (src_lin, tgt_lin, src_quad, tgt_quad)
      ):  # fused quad
        src, tgt = src_quad, tgt_quad
      else:
        raise RuntimeError("Cannot infer OT problem type from data.")
      arrs = src_quad, tgt_quad, src_lin, tgt_lin
      return (src, batch.get("src_condition"), tgt), arrs

    rng = utils.default_prng_key(rng)
    training_logs = {"loss": []}
    for batch in loader:
      rng = jax.random.split(rng, 5)
      rng, rng_resample, rng_noise, rng_time, rng_step_fn = rng

      batch = jtu.tree_map(jnp.asarray, batch)
      (src, src_cond, tgt), matching_data_seq = prepare_data(batch) # (BS, SL,..), (BS, SL,..)
      
      src_perm ,tgt_perm = [], []
      for si in range(self.sequence_length):
        matching_data = [jnp.squeeze(arr[:, si, :]) if arr is not None else None for arr in matching_data_seq]
        tmat, out = self.data_match_fn(*matching_data)  # (n, m)
        src_ixs, tgt_ixs = solver_utils.sample_conditional(  # (n, k), (m, k)
          rng_resample,
          tmat,
          k=self.n_samples_per_src,)
        src_perm.append(src_ixs[:,0])
        tgt_perm.append(tgt_ixs[:,0])

      n = src.shape[0]
      src = jnp.array([src[src_perm[i], i, :] for i in range(self.sequence_length)])
      tgt = jnp.array([tgt[tgt_perm[i], i, :] for i in range(self.sequence_length)])

      latent = self.latent_noise_fn(rng_noise, (n, self.n_samples_per_src))
      latent = latent.reshape(-1, *latent.shape[2:])

      time = jax.random.uniform(rng_time, (self.sequence_length, n, 1))

      loss, self.vf_state, metrics = self.step_fn(
          rng_step_fn, self.vf_state, time, src, tgt, latent, beta = self.beta_schedule(len(training_logs["loss"]))
      )
      wandb.log(metrics)
      transported = self.transport(source = self.eval_data)

      if len(training_logs["loss"]) % 10 == 0:
        fig = plt.figure()
        for i in range(transported.shape[1]):
          plt.scatter(transported[:, i, 0], transported[:, i, 1], label=f"sample {i}")
        wandb.log({"transported": wandb.Image(fig)})
        
      training_logs["loss"].append(loss)
      if len(training_logs["loss"]) >= n_iters: 
        break
    return training_logs

  def _match_latent(
      self, rng: jax.Array, src: jnp.ndarray, src_cond: Optional[jnp.ndarray],
      latent: jnp.ndarray, tgt: jnp.ndarray
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:

    def resample(
        rng: jax.Array, src: jnp.ndarray, src_cond: Optional[jnp.ndarray],
        tgt: jnp.ndarray, latent: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
      tmat = self.latent_match_fn(latent, tgt)  # (n, k)

      src_ixs, tgt_ixs = solver_utils.sample_joint(rng, tmat)  # (n,), (m,)
      src, tgt = src[src_ixs], tgt[tgt_ixs]
      if src_cond is not None:
        src_cond = src_cond[src_ixs]

      return src, src_cond, tgt

    cond_axis = None if src_cond is None else 1
    in_axes, out_axes = (0, 1, cond_axis, 1, 1), (1, cond_axis, 1)
    resample_fn = jax.jit(jax.vmap(resample, in_axes, out_axes))

    rngs = jax.random.split(rng, self.n_samples_per_src)
    return resample_fn(rngs, src, src_cond, tgt, latent)

  def transport(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      t0: float = 0.0,
      t1: float = 1.0,
      rng: Optional[jax.Array] = None,
      **kwargs: Any,
  ) -> jnp.ndarray:
    """Transport data with the learned plan.
    Args:
      source: Data to transport. shape=(n,sl,d)
      condition: Condition of the input data.
      t0: Starting time of integration of neural ODE.
      t1: End time of integration of neural ODE.
      rng: Random generate used to sample from the latent distribution.
      kwargs: Keyword arguments for :func:`~diffrax.odesolve`.

    Returns:
      The push-forward defined by the learned transport plan.
    """

    def vf(t: jnp.ndarray, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      params = self.vf_state.params
      return self.vf_state.apply_fn({"params": params}, t, x, cond, method = "vf_output")

    def solve_ode(x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
      ode_term = diffrax.ODETerm(vf)
      sol = diffrax.diffeqsolve(
          ode_term,
          t0=t0,
          t1=t1,
          y0=x,
          args=cond,
          **kwargs,
      )
      return sol.ys[0]

    kwargs.setdefault("dt0", None)
    kwargs.setdefault("solver", diffrax.Tsit5())
    kwargs.setdefault(
        "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
    )

    jitted_solve = jax.jit(jax.vmap(solve_ode))
    batch_size, seq_len, _ = source.shape
    source = source.transpose(1, 0, 2)
    
    rng = utils.default_prng_key(rng)
    latent = self.latent_noise_fn(rng, (batch_size,))
    carry = jnp.zeros((batch_size, self.vf.rnn_hidden_dim))
    params = self.vf_state.params
    transported_seq = []
    for seq in range(seq_len):
      src = source[seq]
      rng, rng_gru = jax.random.split(rng, 2)
      transported = jitted_solve(latent, src)
      transported_seq.append(transported)
      carry, (mean, logvar, latent) = self.vf_state.apply_fn({"params": params}, method = "rnn_output", src=src, latent=latent, carry=carry, rngs={"gru": rng_gru})

    return jnp.array(transported_seq).transpose(1, 0, 2)
  
  def save(self, path: str):
      """Save the model to a file.

      Args:
        path: Path to save the model.
      """
      ckpt = {'vf_state': self.vf_state}
      save_args = orbax_utils.save_args_from_target(ckpt)
      self.orbax_checkpointer.save(path ,ckpt, save_args= save_args, force = True)

    
  def load(self, path: str): 
      """Load the model from a file
       Args:
           path: Path to load the model from.
      """

      ckpt = self.orbax_checkpointer.restore(path)
      self.vf_state = train_state.TrainState.create(
        apply_fn=self.vf.apply, params=ckpt["vf_state"]["params"], tx=optax.adam(learning_rate=1e-4)
    )

  

@jax.vmap
def _kl_divergence_diag(mu0, logvar0, mu1, logvar1):
    # Reshape to ensure inputs are 2D
   
    var0 = jnp.exp(logvar0)
    var1 = jnp.exp(logvar1)
    
    k = mu0.shape[0]
    
    term1 = jnp.sum(var0 / var1)
    term2 = jnp.sum((mu1 - mu0)**2 / var1)
    term3 = jnp.sum(logvar1 - logvar0)
    
    # KL divergence
    kl_div = 0.5 * (term1 + term2 - k + term3)
    
    return kl_div

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
