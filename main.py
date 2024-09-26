from absl import app
from absl import flags
import wandb
from src.common.utils import read_config
from src.run_experiments import *
import tensorflow as tf
import jax
from ml_collections import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path hyperparameter configuration.',
    lock_config=True)

flags.DEFINE_string("mode", 'train', "train - morph - eval / morph - eval / train / morph / eval")
flags.DEFINE_string("exp", 'rotated_mnist', "which experiment to run")


def main(_):


      # Make sure tf does not allocate gpu memory.
    
    tf.config.experimental.set_visible_devices([], 'GPU')

    print('JAX process: %d / %d', jax.process_index(), jax.process_count())
    print('JAX local devices: %r', jax.local_devices())


    wandb.login()  

    if FLAGS.exp == "rotated_mnist":
       rotated_mnist(FLAGS.config, FLAGS.mode)
    elif FLAGS.exp == "swiss_roll":
       swiss_roll(FLAGS.config, FLAGS.mode)
    elif FLAGS.exp == "feature_morph": 
       print(FLAGS.config)
       morph_features(FLAGS.config, FLAGS.mode)
    else:
       raise ValueError("Experiment not found")
    

if __name__ == '__main__':
  app.run(main) 