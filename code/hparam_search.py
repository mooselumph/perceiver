import logging

import optuna

from train import train_and_evaluate
import jax
import tensorflow as tf

import hydra
from omegaconf import OmegaConf

from model.model import Perceiver
from data import get_stead
from train_stead import train_and_evaluate

import copy


@hydra.main(config_path='configs/default.yaml')
def main(config: OmegaConf):

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())


    def train(config: OmegaConf):
    
        model = Perceiver(**config.model)

        train_ds,test_ds = get_stead(batch_size=config.batch_size)

        state = train_and_evaluate(config, model, train_ds, test_ds, config.logging.workdir)

        return state


    def objective(trial):

        local_config = copy.deepcopy(config)

        local_config.param = trial.suggest_int('n_layers', 1, 3)

        accuracy = train(local_config)

        return accuracy


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

