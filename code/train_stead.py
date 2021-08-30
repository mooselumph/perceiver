
import logging

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint

from tqdm.auto import tqdm

import optax

from functools import partial

from omegaconf import OmegaConf


@partial(jax.jit, static_argnums=(0,))
def apply_model(model, state, batch_x, batch_y):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = model.apply({'params': params}, batch_x)
    one_hot = jax.nn.one_hot(batch_y, 2)

    assert(logits.shape == one_hot.shape)

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == batch_y)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def create_train_state(model, rng, config):
  """Creates initial `TrainState`."""
  params = model.init(rng, jnp.ones((1,) + tuple(model.input_shape)))['params']
  tx = optax.adam(config.optimizer.lr, config.optimizer.momentum)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_epoch(model, state, train_ds, epoch, summary_writer, optune_trial=None):
  """Train for a single epoch."""
  
  epoch_loss = []
  epoch_accuracy = []

  pbar = tqdm(train_ds)
  for step, batch in enumerate(pbar):
    
    batch_x,batch_y = batch

    grads, loss, accuracy = apply_model(model, state, batch_x, batch_y)


    state = update_model(state, grads)

    # Logging 

    pbar.set_description(f'Loss {loss}, Accuracy {accuracy}')

    summary_writer.scalar(f'epoch{epoch}_batch_loss', loss, step)
    summary_writer.scalar(f'epoch{epoch}_batch_accuracy', accuracy, step)

    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)


  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def train_and_evaluate(
                      config: OmegaConf,
                      model: nn.Module,
                      train_ds,
                      test_ds,
                      workdir: str,
                      optune_trial = None
                      ) -> train_state.TrainState:
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
    summary_writer.hparams(OmegaConf.to_container(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(model, init_rng, config)

    test_batch_x, test_batch_y = next(iter(test_ds))

    try:

        for epoch in range(1, config.train.num_epochs + 1):
            rng, input_rng = jax.random.split(rng)
            state, train_loss, train_accuracy = train_epoch(model, state, iter(train_ds), epoch, summary_writer)

            _, test_loss, test_accuracy = apply_model(model, state, test_batch_x,
                                                    test_batch_y)

            print(
                'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
                % (epoch, train_loss, train_accuracy * 100, test_loss,
                test_accuracy * 100))

            try:
                save_checkpoint(workdir, state, epoch, prefix='checkpoint_', keep=5, overwrite=False)
            except:
                return state

            summary_writer.scalar('train_loss', train_loss, epoch)
            summary_writer.scalar('train_accuracy', train_accuracy, epoch)
            summary_writer.scalar('test_loss', test_loss, epoch)
            summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at epoch {epoch}")

    summary_writer.flush()
    return state

