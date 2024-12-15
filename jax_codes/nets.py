from typing import *
#from typing import Any, Callable, Sequence, Optional, Union

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
import numpy as np

from flax.core import freeze, unfreeze
import flax.linen as nn
import cloudpickle as pickle
import math

Array = Any
Dtype = Any  # this could be a real type?
#np = jnp



def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
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





def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class MultipoleConv(nn.Module):
    num_output_filters: int
    num_params: int
    multipole_kernels: jnp.ndarray
    pad_size: int
    num_input_filters: int = 1
    strides: Sequence[int] = None
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    backend: str = "scipy" # or "scipy"
    dtype: Optional[Dtype] = None

    def setup(self):
        #self.kernel_weights = self.param('kernel_weights', self.kernel_init, (self.num_params, 1))
        self.kernel_weights = self.param('kernel_weights', self.kernel_init, (self.num_params, self.num_input_filters))
        self.bias = self.param('bias', self.bias_init, (self.num_output_filters,))
        self.kernel = self.get_kernel()
        self.input_dilation = 1
        self.kernel_dilation = 1
        self.kernel_weights, self.bias, self.kernel = promote_dtype(self.kernel_weights, self.bias, self.kernel, dtype=self.dtype)

        

    def __call__(self, input_field):
        # TOASK: Do we want to support even kernel shapes?
        
        # print("kernel", self.kernel.shape)
        # print("mpk kernel", self.multipole_kernels.shape)
        # print("kernel weights", self.kernel_weights.shape)
        # print("kernel weights", self.kernel_weights)
        # print("input", padded_input.shape)

        # this is the default for one input filter (one field)
        if self.backend == "scipy":
            padded_input = jnp.pad(input_field, pad_width=self.pad_size, mode='wrap')
            
            def convolve(input, kernel, bias):
                return jax.scipy.signal.convolve(input, kernel, mode='valid') + bias
                
            batch_convolve = jax.vmap(convolve, in_axes=[None, 0, 0])
            res = batch_convolve(padded_input, self.kernel.squeeze(-1), self.bias)
            return res
            
            
        else:

            if self.strides is None:
               strides = [1]*(len(self.kernel.shape) - 2) # kernel dimension
            else:
                strides = self.strides
            
            dim = len(self.kernel.shape) - 2 # input kernel is one dim higher
            spatial_pad = ((self.pad_size, self.pad_size),) * dim

            # only pad the spatial dimensions -- input is in shape (spatial, ..., filters)
            pad_width = spatial_pad + ((0,0),)
            
            inputs = jnp.pad(input_field, pad_width=pad_width, mode='wrap')
            kernel_size = self.kernel.shape[1:-1] # go up to last one

            def maybe_broadcast(
                  x: Optional[Union[int, Sequence[int]]],
                ) -> Tuple[int, ...]:
                  if x is None:
                    # backward compatibility with using None as sentinel for
                    # broadcast 1
                    x = 1
                  if isinstance(x, int):
                    return (x,) * len(kernel_size)
                  return tuple(x)

            # TODO: do we need this ?
            # Combine all input batch dimensions into a single leading batch axis.
            # num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
            # if num_batch_dimensions != 1:
            #   input_batch_shape = inputs.shape[:num_batch_dimensions]
            #   total_batch_size = int(np.prod(input_batch_shape))
            #   flat_input_shape = (total_batch_size,) + inputs.shape[
            #     num_batch_dimensions:
            #   ]
            #   inputs = jnp.reshape(inputs, flat_input_shape)

            # fixing kernel dimensions and broadcasting for the lax backend
            strides = maybe_broadcast(self.strides)
            input_dilation = maybe_broadcast(self.input_dilation)
            kernel_dilation = maybe_broadcast(self.kernel_dilation)

            # expand input dims
            inputs = inputs[jnp.newaxis, ...]
            dimension_numbers = _conv_dimension_numbers(inputs.shape)

            # check in vs out features
            in_features = jnp.shape(inputs)[-1]
            feature_group_count = 1
            assert in_features % feature_group_count == 0
            
            # transpose kernel from the multipole convention (outfilters, spatial_weights) convention to the 
            # lax convention
            transpose_axes = [i+1 for i in range(len(self.kernel.shape) - 1)] + [0]
            kernel = jnp.transpose(self.kernel, tuple(transpose_axes))

            y = jax.lax.conv_general_dilated(
                                inputs,
                                kernel,
                                window_strides=strides,
                                padding="VALID",
                                lhs_dilation=input_dilation,
                                rhs_dilation=kernel_dilation,
                                dimension_numbers=dimension_numbers,
                                feature_group_count=1,
                                #precision=self.precision,
                              )
            
            return (y + self.bias).squeeze(axis=0)



    def get_kernel(self):
        # TODO: support multiple input filters        
        return jnp.dot(jnp.transpose(self.multipole_kernels), self.kernel_weights)






class MultipoleCNNFactory:

    def __init__(self, kernel_shape: Sequence[int] = None, polynomial_degrees: Sequence[int] = None,
                 num_input_filters: int = 1, output_filters: Sequence[int] = None, dtype: Any = None):

        self.kernel_shape = kernel_shape
        self.polynomial_degrees = polynomial_degrees
        self.num_input_filters = num_input_filters
        self.output_filters = output_filters
        self.dtype = dtype

        assert len(self.kernel_shape) < 4, 'Only kernels up to three dimensions are supported'

        if self.kernel_shape is None:
            self.kernel_shape = [3, 3]

        if self.polynomial_degrees is None:
            self.polynomial_degrees = [0, 1]

        self.kernel_size = np.prod(self.kernel_shape)
        self.dimension = len(self.kernel_shape)
        if self.output_filters is None:
            self.output_filters = [1 for _ in self.polynomial_degrees]

        assert len(self.output_filters) == len(self.polynomial_degrees), 'A filter must provided for each ℓ'

        self.indices, self.weight_indices = self.get_indices()
        self.num_params = self.weight_indices[-1].astype(int) + 1
        self.num_output_filters = self.indices[-1, -1].astype(int) + 1
        self.multipole_kernels = self.get_kernels()

    def get_indices(self):
        grid, euclidean_dist = self.get_distance()
        angles = self.get_angles(grid, euclidean_dist)
        indices = jnp.zeros((0, self.dimension + 2), dtype=int)
        weight_index = jnp.zeros(0, dtype=int)
        filter_counter = 0
        weight_counter = 0
        for i in range(len(self.polynomial_degrees)):
            l: int = self.polynomial_degrees[i]
            m_range = self.get_polynomial_orders(l)

            for m in m_range:
                if l > 0:
                    distance_modifier = self.get_symmetries(l, m, angles)
                    distance_modifier = distance_modifier.at[jnp.isnan(distance_modifier)].set(0)
                    modified_distances = euclidean_dist * distance_modifier
                else:
                    modified_distances = euclidean_dist
                #print("modified distances", modified_distances.shape)
                unique_distances = jnp.unique(modified_distances)
                #print("distances unique", unique_distances.shape)
                if l != 0 and unique_distances[0] == 0:
                    # FIXME: Figure out what to do with these kernel positions
                    # We do not need weights at the where the grid is zero.
                    unique_distances = jnp.delete(unique_distances, 0)
                for dist in unique_distances:
                    temp_indices = jnp.argwhere(dist == modified_distances)
                    num_elements = temp_indices.shape[0]
                    for in_filt in range(self.num_input_filters):
                        for out_filt in range(self.output_filters[i]):
                            # The input and output filter indices are appended
                            # to the kernel position index to put weight in
                            # correct filter
                            filter_index = jnp.array([in_filt, filter_counter + out_filt])
                            extra_indices = jnp.tile(filter_index, jnp.array([num_elements, 1]))
                            final_indices = jnp.append(temp_indices, extra_indices, axis=1)
                            # TOASK: in general, is this in-place operation okay?
                            indices = jnp.concatenate((indices, final_indices))
                            weight_index = jnp.concatenate((weight_index, np.tile(weight_counter, [num_elements])))
                            weight_counter += 1
                filter_counter += self.output_filters[i]
        return indices, weight_index

    def get_polynomial_orders(self, l):
        if self.dimension == 1:
            # There is only one rotation in 1D (a sign flip)
            m_range = range(1)
        elif self.dimension == 2:
            # There are nd rotations in 2D
            m_range = range(l + 1)
        else:
            # There are 2ℓ + 1 elements in 3D
            m_range = range(-l, l + 1)
        return m_range

    def get_distance(self):
        distance = jnp.mgrid[
            tuple(
                slice(-self.kernel_shape[dim] / 2 + 0.5,
                      self.kernel_shape[dim] / 2 + 0.5,
                      1)
                for dim in range(self.dimension))]
        return distance, jnp.sqrt(jnp.sum(distance ** 2., 0))

    def get_angles(self, grid, distance):
        if self.dimension == 1:
            return {"x": grid[0]}
        elif self.dimension == 2:
            return {"theta": jnp.arctan(jnp.abs(grid[1] / grid[0]))}
        elif self.dimension == 3:
            return {"theta": jnp.arccos(grid[2] / distance),
                    "phi": jnp.arctan(grid[1] / grid[0])}

    def get_symmetries(self, l, m, angles):
        if self.dimension == 1:
            return jnp.arange(self.kernel_shape[0])
        elif self.dimension == 2:
            return jnp.exp(complex(0., 1.) * angles["theta"] * l).real
        elif self.dimension == 3:
            theta = angles["theta"].flatten()
            phi = angles["phi"].flatten()
            res = jax.scipy.special.sph_harm(jnp.full(self.kernel_size, m), jnp.full(self.kernel_size, l), jnp.array(theta), jnp.array(phi)).real
            return res.reshape(self.kernel_shape)

    def get_kernels(self):
        global_kernel_shape = jnp.append(jnp.array(self.kernel_shape),
                                         jnp.array([self.num_input_filters, self.num_output_filters]))

        kernels = np.zeros(jnp.append(self.num_params, global_kernel_shape))

        for i, weight_index in enumerate(self.weight_indices):
            partial_kernel = kernels[weight_index]
            partial_kernel[tuple(self.indices[i])] = 1

        # TODO: support multiple input filters
        return kernels.squeeze((self.dimension + 1,))

    def build_cnn_model(self, pad_size=None, 
                        backend="lax", 
                        num_input_filters=1, 
                        strides=None):

        # FIXME: How to handle even shaped kernels? Do we even want to use them?
        if pad_size is None:
            pad_size = int(math.ceil((self.kernel_shape[0] - 1) / 2))
            
        return MultipoleConv(num_output_filters=self.num_output_filters,
                             num_params=self.num_params,
                             multipole_kernels=self.multipole_kernels,
                             num_input_filters=num_input_filters,
                             backend=backend,
                             strides=strides,
                             pad_size=pad_size,
                             dtype=self.dtype)




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


### MULTIPOLE KERNEL CODE BELOW ###
# TODO: move this to its own repo



