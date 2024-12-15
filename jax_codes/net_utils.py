from typing import Any, Callable, Sequence, Optional, Union
from flax.core import freeze, unfreeze
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax_codes.mpk.multipole_cnn import MultipoleConv
from network.mpk.multipole_cnn_factory import MultipoleCNNFactory
import cloudpickle as pickle
import math

Array = Any
np = jnp


def save_obj(obj, name ):
    """
    Saves an object to a pickle file.

    Parameters
    ----------
    obj : any
        Object to be saved.
    name : str
        Name of the file to save the object to. The file will be saved with a '.pkl'
        extension.
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    """
    Loads an object from a pickle file.

    Parameters
    ----------
    name : str
        Name of the file to load the object from. The file should have a '.pkl'
        extension.

    Returns
    -------
    obj : any
        The loaded object.
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


class MLP(nn.Module):
  features: Sequence[int]
  act: nn.activation = nn.tanh

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


@jax.jit
def smooth_leaky(x: Array) -> Array:
  r"""Smooth Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{smooth\_leaky}(x) = \begin{cases}
      x, & x \leq -1\\
      - |x|^3/3, & -1 \leq x < 1\\
      3x & x > 1
    \end{cases}

  Args:
    x : input array
  """
  return jnp.where(x < -1, x, jnp.where((x < 1), ((-(jnp.abs(x)**3) / 3) + x*(x+2) + (1/3)), 3*x)) / 3.5



def next_power_of_two(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log2(number))

def conv_outs(W, K=2, P=0, S=3):
    return math.ceil(((W - K + (2*P)) / S )+1)

#@jax.jit
def get_padding_pow2(arraylen):
    """
    helper function to pad uneven strided outputs
    """
    
    next_power = next_power_of_two(arraylen)
    deficit = int(math.pow(2, next_power) - arraylen) # how much extra to pad
    
    # but we want to pad both sides of a given axis, so return a tuple here
    
    left = deficit // 2
    right = left + (deficit % 2)
    
    return (left, right)

def get_padding(arraylen, padto):
    """
    helper function to pad uneven strided outputs
    """
    
    #next_power = next_power_of_two(arraylen)
    deficit = int(padto - arraylen + 1) # how much extra to pad
    
    # but we want to pad both sides of a given axis, so return a tuple here
    left = deficit // 2
    right = left + (deficit % 2)
    
    return (left, right)



class MPK_layer(nn.Module):
    multipole_layers: Sequence[MultipoleConv]
    act: Callable = smooth_leaky

    @nn.compact
    def __call__(self, x):
        for i,l in enumerate(self.multipole_layers):
            z = l(x)
            x = self.act(z) if i == 0 else self.act(x + z)
        return x
        
        
# log transform
def log_transform(x):
    xo = jnp.abs(x.min(axis=0, keepdims=True)) + 0.01
    return xo * jnp.log(1.0 + (x / xo))


