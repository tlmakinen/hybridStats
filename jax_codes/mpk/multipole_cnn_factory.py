from typing import *

import numpy as np

import jax
import jax.numpy as jnp
import math
from .multipole_cnn import MultipoleConv

__version__ = '2.0'
__author__ = "Simon Ding & Lucas Makinen"


# TODO: Add documentation
# TOASK: Is there a better way to solve this than with a factory pattern?

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
