from sklearn import svm
from flax import linen as nn
from typing import Any, Dict
from jax import random
import jax.numpy as jnp
from flax.training import train_state
from src.common.models import *
from src.common.utils import *
import orbax.checkpoint as ocp
import os 
from flax.training.orbax_utils import save_args_from_target
def get_trainer(rng, train_type, confs, model, ds_builder, logdir, callback = None):

    if train_type == "vae":
        return TrainerModuleVAE(rng, confs["train"], model, ds_builder, logdir, callback)
    elif train_type == "fused_vae":
        return TrainerModulefVAE(rng, confs["train"], model, ds_builder, logdir, callback)

class TrainerModule:
    def __init__(self, rng, train_conf, model, ds_builder, logdir, callback = None):
        self.name = model.name
        self.modality = ds_builder.modality
        self.model_conf = model.model_conf
        self.latent_dim = self.model_conf["latent_dim"]
        self.model = model
        self.rng = rng
        self.ds_builder = ds_builder
        self.train_conf = train_conf
        self.num_epochs = train_conf["num_epochs"]
        self.logger  = WandbLogger(self.name, 20)
        self.callback = callback


        print(self.model_conf)
        abs_path = os.path.abspath(logdir)
        save_path = os.path.join(abs_path, "train", wandb.run.id, self.name)

        # At the top level
        mgr_options = ocp.CheckpointManagerOptions(
        create=True, max_to_keep=1, save_interval_steps=50)

        self.ckpt_mgr =ocp.CheckpointManager(save_path, options=mgr_options, item_names=["state", "metadata"])

    
    def save_model(self, step):
        if isinstance(self.model_conf, dict):
            metadata = {"model_conf": self.model_conf, "train_conf": self.train_conf, "modality": self.modality}
        else:
            metadata = {"model_conf": self.model_conf.to_dict(), "train_conf": self.train_conf.to_dict(), "modality": self.modality}
        self.ckpt_mgr.save(step, args=ocp.args.Composite(
            state=ocp.args.StandardSave(self.state), metadata=ocp.args.JsonSave(metadata)))

    def get_parm_schedule(self, schedule_conf):
        if schedule_conf["type"] == "const":
            return optax.constant_schedule(schedule_conf["value"])
        else:
            return optax.linear_schedule(schedule_conf["init_value"], schedule_conf["end_value"], 
                                         schedule_conf["n_epochs"], schedule_conf["transition_begin"])
    

class TrainerModuleVAE(TrainerModule):

    def __init__(self, rng, train_conf, model, ds_builder, logdir, callback = None):
        super().__init__(rng, train_conf, model, ds_builder, logdir, callback)
        self.beta_schedule_conf = train_conf["beta_schedule"]
        self.reconstruction_loss = get_loss_function(self.model_conf["reconstruction_loss"])
        self.kl_divergence=  get_loss_function(self.model_conf["kl_divergence"])
        self.svm_kernels = ["linear", "rbf"]

    
    def create_functions(self):
        # train function
        if self.modality == "text":
            kl_divergence_normaliztion = self.latent_dim
        else:    
            kl_divergence_normaliztion = 1
        def train_step(state, batch, z_rng, beta):
            def loss_fn(params):
                recon_x, mean, logvar = self.model.apply(
                    {'params': params}, batch, z_rng)

                rec_loss = self.reconstruction_loss(recon_x, batch).mean()
                kld_loss = self.kl_divergence(mean, logvar).mean()/kl_divergence_normaliztion
                loss = rec_loss + beta*kld_loss
                metrics = self.compute_metrics(recon_x, batch, mean, logvar, beta)
                return loss, metrics
            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, metrics
        
        self.train_step = jax.jit(train_step)

        def eval_f(params, x, z_rng, beta):
            def eval_model(vae):
                recon_x, mean, logvar = vae(x, z_rng)
                metrics = self.compute_metrics(recon_x, x, mean, logvar, beta)
                return metrics 
            return nn.apply(eval_model, self.model)({'params': params})            
        
        self.eval_f = jax.jit(eval_f)
        
        self.beta_schedule = self.get_parm_schedule(self.beta_schedule_conf)

    def train_model(self):
        # Initialize model
        self.rng, key = random.split(self.rng)

        self.ds_builder.set_train_vars(self.train_conf["batch_size"])

        dummy_input = self.ds_builder.get_dummy()
        print(dummy_input.shape)
        params = self.model.init(key, dummy_input, self.rng)['params']
        optimizer = get_optimization(self.train_conf)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

        self.rng, z_key, eval_rng = random.split(self.rng, 3)
        
        z = random.normal(z_key, (64, self.latent_dim))
        num_train_samples = 8
        train_samples, _ = next(self.ds_builder.get_n_samples("train", num_train_samples))
    
        num_val_samples = 8
        val_samples, _ =next(self.ds_builder.get_n_samples("val", num_val_samples))

        train_ds = self.ds_builder.build_split("train")
        train_steps_per_epoch = self.ds_builder.get_n_steps("train")

        val_ds, labels = next(self.ds_builder.get_n_samples("val", 1000))
        
        for epoch in range(self.num_epochs):
            train_batch_metrics = []
            beta = self.beta_schedule(epoch)
            for _ in range(train_steps_per_epoch):
                batch,_ = next(train_ds)
                self.rng, key = random.split(self.rng)
                self.state, metrics = self.train_step(self.state, batch, key, beta)
                train_batch_metrics.append(metrics)
            
            train_batch_metrics = WandbLogger.accumulate_metrics("Train", train_batch_metrics)

            val_batch_metrics = [self.eval_f(self.state.params, val_ds, eval_rng, beta)]      
            val_batch_metrics = WandbLogger.accumulate_metrics("Val", val_batch_metrics)

            self.logger.log_metrics({**train_batch_metrics, **val_batch_metrics ,"epoch": epoch})
            self.save_model(epoch)
            if self.modality == "image":
                 # log images
                 generated_images = self.model.apply({"params": self.state.params}, z, method= "generate")
                 self.logger.log_image(generated_images, "sample", epoch)
                 recon_samples, _, _ =  self.model.apply({"params": self.state.params}, val_samples, eval_rng)
                 comparison = jnp.concat((val_samples, recon_samples))
                 self.logger.log_image(comparison, "test-reconstruction", epoch)
                 recon_samples, _, _ =  self.model.apply({"params": self.state.params}, train_samples, eval_rng)
                 comparison = jnp.concat((train_samples, recon_samples))
                 self.logger.log_image(comparison, "train-reconstruction", epoch)

        self.ckpt_mgr.wait_until_finished()
        
            
    def compute_metrics(self,recon_x, x, mean, logvar, beta):
        rec_loss = self.reconstruction_loss(recon_x, x).mean()
        kld_loss = self.kl_divergence(mean, logvar).mean()
        metrics = {'rec': rec_loss, 'kld': kld_loss, 'loss': rec_loss + beta*kld_loss, 'beta': beta}
        if self.modality == 'text':
            metrics["acc"] = accuracy(recon_x, x).mean()
        return metrics

    
    def draw_latent(self, rng):
        total_n_val = self.ds_builder.get_num_examples("val")
        val_X, labels = self.ds_builder.get_n_samples("val", total_n_val)
        z = self.model.apply({"params": self.state.params}, val_X, rng, method= "encode_and_sample")
        fig = plot_latent_umap(z, labels, metric='cosine')
        self.logger.log_plot_image(fig, "umap")
        fig = plot_latent_umap(z, labels, metric='euclidean')
        self.logger.log_plot_image(fig, "umap")


class TrainerModulefVAE(TrainerModule):
    
        def __init__(self, rng, train_conf, model, ds_builder, logdir, callback = None):
            super().__init__(rng, train_conf, model, ds_builder, logdir,callback)
            self.beta_schedule_conf = train_conf["beta_schedule"]
            self.reconstruction_loss1 = get_loss_function(self.model_conf["reconstruction_loss1"])
            self.reconstruction_loss2 = get_loss_function(self.model_conf["reconstruction_loss2"])
            self.kl_divergence=  get_loss_function(self.model_conf["kl_divergence"])
            self.kl_normalization = self.latent_dim
                
        def create_functions(self, type):
            # train function

            def train_step(state, batch_x, batch_y, z_rng, beta):
                def loss_fn(params):
                    recon_x1, recon_x2, recon_y1, recon_y2, mean_x, logvar_x, mean_y, logvar_y = self.model.apply(
                        {'params': params}, batch_x, batch_y, z_rng)

                    if type=="text_image":
                        rec_loss_x1 = 0
                        rec_loss_y1 = 0
                        rec_loss_y2 = 0
                        kld_loss_x = 0
                        rec_loss_x2 = self.reconstruction_loss1(recon_x2, batch_x).mean()
                        kld_loss_y = self.kl_divergence(mean_y, logvar_y).mean()
                        loss = rec_loss_x2 + beta*(kld_loss_y)
                    elif type=="image_text":
                        rec_loss_x1 = 0
                        rec_loss_x2 = 0
                        rec_loss_y2 = 0
                        kld_loss_y = 0
                        rec_loss_y1 = self.reconstruction_loss2(recon_y1, batch_y).mean()
                        kld_loss_x = self.kl_divergence(mean_x, logvar_x).mean()/self.kl_normalization
                        loss = rec_loss_y1 + beta*(kld_loss_x)
                    elif type== "fused":
                        rec_loss_x1 = self.reconstruction_loss1(recon_x1, batch_x).mean()
                        rec_loss_x2 = self.reconstruction_loss1(recon_x2, batch_x).mean()
                        rec_loss_y1 = self.reconstruction_loss2(recon_y1, batch_y).mean()
                        rec_loss_y2 = self.reconstruction_loss2(recon_y2, batch_y).mean()
                        kld_loss_y = self.kl_divergence(mean_y, logvar_y).mean()/self.kl_normalization
                        kld_loss_x = self.kl_divergence(mean_x, logvar_x).mean()/self.kl_normalization
                        loss = (rec_loss_x1 + rec_loss_x2)+ (rec_loss_y1 + rec_loss_y2) + beta*(kld_loss_x + kld_loss_y)
                    else:
                        raise ValueError("Invalid type")

                    metrics = self.compute_metrics(rec_loss_x1, rec_loss_x2, rec_loss_y1, rec_loss_y2, kld_loss_x, kld_loss_y, loss, beta)
                    return loss, metrics
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                state = state.apply_gradients(grads=grads)
                return state, metrics
            
            self.train_step = jax.jit(train_step)
                
            self.beta_schedule = self.get_parm_schedule(self.beta_schedule_conf)
        
        def train_model(self, take):
            # Initialize model
            rng, key = random.split(self.rng)
    
            self.ds_builder.set_train_vars(self.train_conf["batch_size"])
    
            dummy_input_x, dummy_input_y = self.ds_builder.get_dummy()
            params = self.model.init(key, dummy_input_x, dummy_input_y, rng)['params']
            optimizer = get_optimization(self.train_conf)
            self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
        
            train_ds = self.ds_builder.build_split("val", take)
            train_steps_per_epoch = self.ds_builder.get_n_steps()
            
            for epoch in range(self.num_epochs):
                train_batch_metrics = []
                beta = self.beta_schedule(epoch)
                for _ in range(train_steps_per_epoch):
                    rng, key = random.split(rng)
                    (batch_x, batch_y), _ = next(train_ds)
                    self.state, metrics = self.train_step(self.state, batch_x, batch_y, key, beta)
                    train_batch_metrics.append(metrics)
                
                train_batch_metrics = WandbLogger.accumulate_metrics("Train", train_batch_metrics)
                         
                print({**train_batch_metrics, "epoch": epoch})
    
                self.logger.log_metrics({**train_batch_metrics, "epoch": epoch})
                self.save_model(epoch)
                if self.callback is not None and (epoch +1) % 10 == 0:
                    self.callback(model = self.model, params = self.state.params, )
            
            # self.draw_latent(rng)
            self.ckpt_mgr.wait_until_finished()

        def compute_metrics(self, rec_loss_x1, rec_loss_x2, rec_loss_y1, rec_loss_y2, kld_loss_x, kld_loss_y, loss, beta):
            metrics = {'rec_x1': rec_loss_x1, "rec_x2": rec_loss_x2,
                        "rec_y1": rec_loss_y1, "rec_y2": rec_loss_y2,
                          'kld_x': kld_loss_x, 'kld_y': kld_loss_y,
                            'loss': loss, "beta": beta}
            return metrics
        
        def draw_latent(self, rng):
            x = self.ds_builder.x
            y = self.ds_builder.y
            labels = self.ds_builder.labels
            z_x = self.model.apply({"params": self.state.params}, x, rng, 1, method= "encode_and_sample")
            z_y = self.model.apply({"params": self.state.params}, y, rng, 2, method= "encode_and_sample")
            fig = plot_latent_umap2(z_x, z_y, labels, metric='euclidean')
            self.logger.log_plot_image(fig, "umap")


        