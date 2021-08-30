
from typing import Optional, Tuple

from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import jax.nn.initializers as init

from flax import linen as nn

from einops import rearrange, repeat


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = jnp.expand_dims(x,-1)
    dtype, orig_x = x.dtype, x

    scales = jnp.logspace(0., jnp.log(max_freq / 2) / jnp.log(base), num_bands, base = base, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * jnp.pi
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis =-1)
    x = jnp.concatenate((x, orig_x), axis = -1)
    return x



class GEGLU(nn.Module):
  @nn.compact
  def __call__(self,x):
    x, gates = jnp.split(x,2,axis=-1)
    return x * nn.gelu(gates)

class FeedForward(nn.Module):
  dim: int
  mult: int = 4
  dropout: float = 0.

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.dim*self.mult*2)(x)
    x = GEGLU()(x)
    if self.dropout > 0:
      x = nn.Dropout(self.dropout)(x)
    x = nn.Dense(self.dim)(x)

    return x


class Attention(nn.Module):

  query_dim: int
  num_heads: int = 8
  head_dim: int = 64
  dropout: float = 0.


  def setup(self):

    inner_dim = self.head_dim*self.num_heads
    self.scale = self.head_dim ** -0.5 # Don't know that this does

    self.to_q = nn.Dense(inner_dim, use_bias=False)
    self.to_kv = nn.Dense(2*inner_dim, use_bias=False)

    self.to_out = nn.Dense(self.query_dim)

  @nn.compact
  def __call__(self,x,context=None):

    h = self.num_heads

    q = self.to_q(x)

    context = default(context,x)
    k, v = jnp.split(self.to_kv(context),2,axis=-1)

    # b n h d:  batch, num_latents, num_heads, inner_dimmension
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

    sim = jnp.einsum('b i d, b j d -> b i j', q, k) * self.scale

    attn = nn.softmax(sim,axis=-1)

    out = jnp.einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

    out = self.to_out(out)

    if self.dropout > 0:
      out = nn.Dropout(self.dropout)(out)

    return out

class AttentionBlock(nn.Module):

  query_dim: int
  num_heads: int = 8
  head_dim: int = 64
  attn_dropout: float = 0.
  ff_dropout: float = 0.

  def setup(self):

    self.attention = Attention(
        query_dim=self.query_dim,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        dropout=self.attn_dropout
      )
    self.ff = FeedForward(
        dim=self.query_dim,
        dropout=self.ff_dropout
      )

  @nn.compact
  def __call__(self,x,**kwargs):

    x = nn.LayerNorm()(x)
    if 'context' in kwargs:
      kwargs['context'] = nn.LayerNorm()(kwargs['context'])
    x = self.attention(x,**kwargs) + x
    x = nn.LayerNorm()(x)
    x = self.ff(x) + x
    
    return x

class Perceiver(nn.Module):

  input_shape: Tuple[int]

  weight_tie_pattern: Tuple[int]

  num_freq_bands: int
  max_freq: float
  freq_base: int = 2
  fourier_encode_data: bool = True
  
  num_latents: int = 512
  latent_dim: int = 512

  cross_heads: int = 1
  cross_head_dim: int = 64

  latent_heads: int = 8
  latent_head_dim: int = 64
  self_per_cross_attn: int = 1

  num_classes: int = 1000

  attn_dropout: float = 0.
  ff_dropout: float = 0.


  def setup(self):

    get_block = partial(AttentionBlock,query_dim = self.latent_dim, attn_dropout = self.attn_dropout, ff_dropout = self.ff_dropout)

    unique_layer_ids = set(self.weight_tie_pattern)
    self.unique_layers = {str(id): [
      get_block(num_heads = self.latent_heads,head_dim = self.latent_head_dim),
      [get_block(num_heads = self.cross_heads,head_dim = self.cross_head_dim) for _ in range(self.self_per_cross_attn)]
    ]
    for id in unique_layer_ids}
    
    self.to_logits = nn.Dense(self.num_classes)


  @nn.compact
  def __call__(self,data):

    b, *axis, _ = data.shape

    if self.fourier_encode_data:
      # calculate fourier encoded positions in the range of [-1, 1], for all axis

      axis_pos = list(map(lambda size: jnp.linspace(-1., 1., num = size), axis))

      pos = jnp.stack(jnp.meshgrid(*axis_pos), axis = -1)
      enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
      enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
      enc_pos = repeat(enc_pos, '... -> b ...', b = b)

      data = jnp.concatenate((data, enc_pos), axis = -1)

    # concat to channels of data and flatten axis
    data = rearrange(data, 'b ... d -> b (...) d')

    # self.sow('intermediates', 'features', data)

    x = self.param('initial_state',init.normal(),(self.num_latents, self.latent_dim))

    x = repeat(x, 'n d -> b n d', b = b)


    for id in self.weight_tie_pattern:

      cross_block, self_blocks = self.unique_layers[str(id)]
      x = cross_block(x,context=data)

      for self_block in self_blocks:
        x = self_block(x)

    x = jnp.mean(x,axis=-2)

    x = nn.LayerNorm()(x)
    return self.to_logits(x)
