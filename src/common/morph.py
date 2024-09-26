import jax
import optax

from ott.neural.methods.flows import dynamics

from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from ott.neural.networks import velocity_field

from src.common.utils import *
from src.common.data import GenotDataLoader
from src.common.ott import *
from src.common.genot import GenotModified

class MorphUsingGenot:
    def __init__(self,
                 rng, 
                 confs, 
                 train_ds,
                 vf, 
                 is_aligned = False,
                 save_path = None,
                 callback_func = None,
                 match_fn = None,
                 x_paired=None,
                 y_paired=None,
                 ):

        self.rng = rng
        self.vf = vf
        self.morph_confs = confs["morph"]
        self.train_ds = train_ds
        self.is_aligned = is_aligned
        self.save_path = save_path
        self.callback_func = callback_func
        self.match_fn = match_fn
        self.x_paired = x_paired
        self.y_paired = y_paired

    def morph(self):
            
        self.rng, rng_init, rng_call, rng_dl = jax.random.split(self.rng, 4)

        src_dim = self.train_ds.src_dim
        tgt_dim = self.train_ds.tgt_dim

        print(f"Source dimension: {src_dim} Target dimension: {tgt_dim}")
        
        param_schedule = get_param_schedule(self.morph_confs["optimizer"]["lr_schedule"])
        optimizer = optax.adam(learning_rate=param_schedule)

        model = GenotModified(
            vf= self.vf,
            flow=dynamics.ConstantNoiseFlow(0.0),
            data_match_fn=self.match_fn,
            source_dim=src_dim,
            target_dim=tgt_dim,
            rng=rng_init,
            optimizer=optimizer,
            n_samples_per_src= 1,
            use_true_alignment=self.is_aligned,
            save_path=self.save_path,
            callback=self.callback_func,
            callback_freq=100)
                

        n_iters = self.morph_confs["n_iters"]
        train_dl = GenotDataLoader(rng= rng_dl, dataset= self.train_ds, batch_size= self.morph_confs["batch_size"])

        model(train_dl, n_iters, rng=rng_call, x_paired= self.x_paired, y_paired=  self.y_paired)
    
        return model