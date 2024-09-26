
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import math
import jax
import dataclasses
from ott.neural.datasets import OTData, OTDataset
from typing import Any, Dict, Optional, Sequence 
import numpy as np
from collections import defaultdict

Item_t = Dict[str, np.ndarray]


class DsBuilder:
    buffer_size = 60000
    train_split = "train[:60%]"
    val_split = "train[60%:80%]"
    test_split = "train[80%:]"
    def __init__(self, conf, rng, modality):
        """Initialize the dataset builder with a dataset name."""
        self.modality = modality
        self.seed = conf["seed"]
        self.conf = conf
        self.rng = rng
        self.cache = True 


    def set_train_vars(self, batch_size):
        self.batch_size = batch_size
    
    def build_dataset(self, split , cache, n_repeats = 1):
        ds_split = self.get_split(split)
        ds = self.ds_builder.as_dataset(split=ds_split, shuffle_files=False)
        ds = ds.enumerate()
        ds = ds.repeat(n_repeats)
        ds = ds.map(self.preprocess)
        if cache:
            ds = ds.cache()
        return ds
    
    def build_split(self, split, take=-1):
        ds = self.build_dataset(split=split, cache= self.cache)
        if take != -1:
            ds = ds.take(take)
        ds = ds.repeat()
        ds = ds.shuffle(self.buffer_size, seed = self.seed)
        ds = ds.batch(self.batch_size)
        ds = iter(tfds.as_numpy(ds))
        return ds

    def get_split(self, split):
        if split == "train":
            return self.train_split
        elif split == "test":
            return self.test_split
        elif split == "val":
            return self.val_split
        else:
            return split
    
    def get_num_examples(self, split):
        ds_split = self.get_split(split)
        return self.ds_builder.info.splits[ds_split].num_examples
    
    def get_n_steps(self, split = "train"):
        return self.get_num_examples(split) // self.batch_size
        
    def get_n_samples(self, split, n_samples, batch_size = -1, n_repeats = 1):
        """Fetch a specified number of samples from the dataset."""
        if batch_size == -1:
            batch_size = n_samples
        ds = self.build_dataset(split=split, cache=False, n_repeats=n_repeats)
        samples = iter(tfds.as_numpy(ds.take(n_samples).batch(batch_size)))
        return samples   
    
    def preprocess(self, index, x):
        if self.modality == "image":
            return self.prepare_image(index,x)
        elif self.modality == "fused":
            return self.prepare_fused(index, x)
        else:
            return self.prepare_label(index, x)


class MnistDsBuilder(DsBuilder):
    
    train_split = "train[:50000]"
    val_split = "train[50000:]"
    test_split = "test"
    
    def __init__(self, conf, rng, modality):
        """Initialize the dataset builder with a dataset name."""
        super().__init__(conf, rng, modality)
        self.ds_builder = tfds.builder("mnist")
        self.seed = conf["seed"]
        self.angles = jnp.linspace(0,180, conf["n_angles"])
        self.angles_tensor = tf.constant(self.angles, dtype=tf.float32)
        self.encoding_depth = len(self.angles)*10
        self.ds_builder.download_and_prepare()
        self.n_train = self.ds_builder.info.splits["train"].num_examples
        
        self.rot_indices = tf.constant(jax.random.choice(rng, conf["n_angles"], (self.n_train, )), dtype=tf.int32)

    def get_dummy(self):
        if self.modality == "image":
            return jnp.ones((self.batch_size, 28, 28, 1), jnp.float32)
        if self.modality == "fused":
            return (jnp.ones((self.batch_size, 28, 28, 1), jnp.float32), jnp.ones((self.batch_size, self.encoding_depth), jnp.float32))
        else:
            return jnp.ones((self.batch_size, self.encoding_depth), jnp.float32)
        
    def prepare_image(self,index,x):
        """Prepares an image for training or testing."""
        y = tf.cast(x['label'], tf.int32)
        x = tf.cast(x['image'], tf.float32)
        x = x / 255.0
        x = tf.where(x > 0.5, 1.0, 0.0)  

        rot_index = tf.gather(self.rot_indices, index)
        selected_angle = tf.gather(self.angles_tensor, rot_index)
        radians = selected_angle * (math.pi / 180)  
        x = tfa.image.rotate(x, radians)
        encoded_label = MnistDsBuilder.encode_labels(y, rot_index)
        return x, encoded_label
    
    def prepare_fused(self, index, x):
        """Prepares an image for training or testing."""
        img, label = self.prepare_image(index, x)
        text, _ = self.prepare_label(index, x)

        return (img, text), label
    
    def prepare_label(self, index, x):
        """Prepares a label for training or testing."""
        y = tf.cast(x['label'], tf.int32)
        x = tf.cast(x['image'], tf.float32)

        rot_index = tf.gather(self.rot_indices, index)
        encoded_label = MnistDsBuilder.encode_labels(y, rot_index)
        return self.one_hot_encoding(encoded_label), encoded_label
    
    def one_hot_encoding(self, encoded_label):
        one_hot_encode = tf.one_hot(encoded_label, self.encoding_depth)
        return one_hot_encode 
    
    @staticmethod
    def encode_labels(label, rot_index):
        return rot_index*10 + label
    

class FusedDsBuilder():
    buffer_size = 60000
    modality = "fused"
    def __init__(self, x, y, labels, conf):
        """Initialize the dataset builder with."""
        self.x = x
        self.y = y
        self.labels = labels
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        self.seed = conf["seed"]

    
    def set_train_vars(self, batch_size):
        self.batch_size = batch_size

    def get_dummy(self):
            return (jnp.ones((1,self.x.shape[-1]), jnp.float32), jnp.ones((1,self.y.shape[-1]), jnp.float32))
   
    def build_split(self, split, take=-1):
        ds = self.train_dataset
        ds = ds.map(self.prepare)
        ds = ds.repeat()
        ds = ds.shuffle(self.buffer_size, seed = self.seed)
        ds = ds.batch(self.batch_size)
        ds = iter(tfds.as_numpy(ds))
        return ds
    
    def prepare(self, x,y):
        return (x, y), 1
    
    def get_n_steps(self):
        n_steps = self.x.shape[0]// self.batch_size
        return max(n_steps, 1)
    
    def get_n_examples(self):
        return self.x.shape[0]
    

@dataclasses.dataclass(repr=False, frozen=True)
class OTDataExtended(OTData):
    labels: Optional[np.ndarray] = None

    def __getitem__(self, ix: int) -> Item_t:
        data = super().__getitem__(ix)
        if self.labels is not None:
            data['labels'] = self.labels[ix]
        data['ix'] = ix
        return data
    
    def get_dimension(self):
        if self.quad is not None:
            return  self.quad.shape[-1]
        return self.lin.shape[-1]
        
         

class OTDatasetExtended(OTDataset):
    def __init__(
        self,
        src_data: OTDataExtended,
        tgt_data: OTDataExtended,
        src_conditions: Optional[Sequence[Any]] = None,
        tgt_conditions: Optional[Sequence[Any]] = None,
        is_aligned: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(src_data, tgt_data, src_conditions, tgt_conditions, is_aligned, seed)
        self.src_dim = src_data.get_dimension()
        self.tgt_dim = tgt_data.get_dimension()

    
    # def _sample_from_target(self, src_ix: int) -> Item_t:
    #     src_cond = self.src_conditions[src_ix]
    #     tgt_ixs = self._tgt_cond_to_ix[src_cond]
    #     ix = self._rng.choice(tgt_ixs)
    #     return ix

    # def __getitem__(self, ix: int) -> Item_t:
    #     # src = self.src_data[ix]
    #     # src = {f"{self.SRC_PREFIX}_{k}": v for k, v in src.items()}
    #     # src_ix = ix
    #     # if self.is_aligned:
    #     #     tgt_ix = src_ix
    #     # else:
    #     #     tgt_ix = self._sample_from_target(ix)
    #     # tgt = self.tgt_data[tgt_ix]
    #     # tgt = {f"{self.TGT_PREFIX}_{k}": v for k, v in tgt.items()}

    #     item = super().__getitem__(ix)
    #     if 'labels' in self.src_data.__dict__:
    #         if self.src_data.labels is not None:
    #             item[f"{self.SRC_PREFIX}_labels"] = self.src_data.labels[ix]
    #         if self.tgt_data.labels is not None:
    #             item[f"{self.TGT_PREFIX}_labels"] = self.tgt_data.labels[ix]
    #     return item    


class GenotDataLoader:
    def __init__(self, rng, dataset: OTDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self._rng = rng

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, jnp.ndarray]:
        data = defaultdict(list)
        self._rng, key = jax.random.split(self._rng)
        indices = jax.random.randint(key, (self.batch_size,), 0, len(self.dataset))
        for ix in indices:
          for k, v in self.dataset[ix].items():
              data[k].append(v)
        return {k: jnp.stack(v) for k, v in data.items()}