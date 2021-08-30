import logging

import jax
import tensorflow as tf


import hydra
from omegaconf import DictConfig

from model.model import Perceiver
from data import get_stead
from train_stead import train_and_evaluate

from utils import make_absolute

@hydra.main(config_path='configs',config_name='default')
def main(config):

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    
    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    #   platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
    #                                        f'process_count: {jax.process_count()}')
    #   platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
    #                                        FLAGS.workdir, 'workdir')

    model = Perceiver(**config.model)

    train_ds,test_ds = get_stead(batch_size=config.train.batch_size)

    workdir = make_absolute(config.logging.workdir)

    state = train_and_evaluate(config, model, train_ds, test_ds, workdir)


if __name__ == '__main__':
    main()