
import tensorflow_datasets as tfds
import tensorflow as tf
import jax.numpy as jnp

def get_mnist():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds



import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils import make_absolute

STEAD_HDF5 = make_absolute("../datasets/stead/merge.hdf5")
STEAD_CSV = make_absolute("../datasets/stead/merge.csv")


class stead_generator:
    def __init__(self, hdf_file=STEAD_HDF5, csv_file=STEAD_CSV, test_ratio=0.25, noise_ratio=0.5, sigma=200):
        self.hdf_file = hdf_file
        self.csv_file = csv_file
        self.noise_ratio = noise_ratio

        self.sigma = sigma

        self.attrs = ['p_arrival_sample','s_arrival_sample']

        df = pd.read_csv(self.csv_file)

        self.test_df = df.sample(frac = test_ratio)
        self.train_df = df.drop(self.test_df.index)

    def __call__(self,test_split=False):

        df = self.test_df if test_split else self.train_df

        with h5py.File(self.hdf_file, 'r') as hf:

            def get_earthquake():
                # earthquake_iter = iter(df[df.trace_category == 'earthquake_local']['trace_name'].to_list())
                earthquake_iter = iter(df[(df.trace_category == 'earthquake_local') & (df.source_magnitude <= 4)]['trace_name'].to_list())
                
                for event in earthquake_iter:

                    dataset = hf.get('data/'+str(event))
                    x = np.array(dataset) / self.sigma
                    # y = np.array([dataset.attrs[at] for at in self.attrs] + [1],dtype=np.int32)
                    y = 1

                    yield x,y

            def get_noise():
                while True:
                    noise_iter = iter(df[df.trace_category == 'noise']['trace_name'].to_list())
                    for event in noise_iter:

                        dataset = hf.get('data/'+str(event))
                        x = np.array(dataset)/ self.sigma
                        # y = np.zeros((3,))
                        y = 0
                        yield x,y

            noise = iter(get_noise())

            for (x,y) in get_earthquake():

                while np.random.uniform() < self.noise_ratio:
                    yield next(noise)
                
                yield x,y


def get_stead(hdf_file=STEAD_HDF5,csv_file=STEAD_CSV, test_ratio=0.25, noise_ratio=0.5, batch_size=30):

    gen = stead_generator(hdf_file,csv_file,test_ratio,noise_ratio)

    train_ds = tf.data.Dataset.from_generator(
            gen,
            (tf.float32,tf.int32),
            (tf.TensorShape((6000,3)),tf.TensorShape(())),
        )

    test_ds = tf.data.Dataset.from_generator(
            gen,
            (tf.float32,tf.int32),
            (tf.TensorShape((6000,3)),tf.TensorShape(())),
            args = [True],
        )

    train_ds = tfds.as_numpy(train_ds.batch(batch_size))
    test_ds = tfds.as_numpy(test_ds.batch(batch_size))

    return train_ds, test_ds