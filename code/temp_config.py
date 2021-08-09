
def get_config():
  """Get the default hyperparameter configuration."""

  return dict(
    learning_rate = 0.1,
    momentum = 0.9,
    batch_size = 128,
    num_epochs = 10
  )