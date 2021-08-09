import logging
import click

from config import get_config, DEFAULT_CONFIG_FILE
from train import train_and_evaluate
import jax
import tensorflow as tf


@click.group()
def cli():
    pass

@cli.command()
@click.option('--config_file', default=DEFAULT_CONFIG_FILE, help='Config file.', type=str)
@click.option('--workdir', default='../')
def train(config_file,workdir):

    config = get_config(config_file)

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

    train_and_evaluate(config, workdir)
#

if __name__ == '__main__':
    cli()