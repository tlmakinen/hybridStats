from typing import *

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
import numpy as np

Dtype = Any  # this could be a real type?

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
