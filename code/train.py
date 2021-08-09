import numpy as np

import logging

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

from tqdm.auto import tqdm

import optax

from config import ConfigSchema
from data import get_datasets

from functools import partial


@partial(jax.jit, static_argnums=(0,))
def apply_model(model, state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = model.apply({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def create_train_state(model, rng, config):
  """Creates initial `TrainState`."""
  params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.adam(config.learning_rate, config.momentum)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)




def train_epoch(model, state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  pbar = tqdm(perms)
  for perm in pbar:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]

    grads, loss, accuracy = apply_model(model, state, batch_images, batch_labels)

    pbar.set_description(f'Loss {loss}, Accuracy {accuracy}')

    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)


  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def train_and_evaluate(
                      model: nn.Module,
                      train_ds,
                      test_ds,
                      config: ConfigSchema,
                      workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.
  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.
  Returns:
    The train state (which includes the `.params`).
  """

  logging.basicConfig(level=logging.INFO)

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  

  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(model, init_rng, config)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(model, state, train_ds,
                                                    config.batch_size,
                                                    input_rng)
    _, test_loss, test_accuracy = apply_model(model, state, test_ds['image'][:config.batch_size],
                                              test_ds['label'][:config.batch_size])

    print(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state