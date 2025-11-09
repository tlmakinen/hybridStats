from typing import Tuple
import math
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import optax
import matplotlib.pyplot as plt
from functools import partial
import flax.linen as nn
import jax.random as jr
#import netket as nk


def indices_vector(num_tomo):
    """
    compute auto- and cross-indices 
    for Cls calculation
    """
    indices = []
    cc = 0
    for catA in range(0,num_tomo,1):
        for catB in range(catA,num_tomo,1):
            indices.append([catA, catB])
            cc += 1
    return indices



@jax.jit
def compute_auto_cross_angular_power_spectrum(
    field1: jnp.ndarray,
    field2: jnp.ndarray,
    distance: float,
    size: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the auto or cross angular power spectrum of 2D arrays of data.

    Parameters:
        - field1 (jnp.ndarray): first 2D data field with shape (N, N).
        - field2 (jnp.ndarray): second 2D data field with shape (N, N). If the same field
            is provided, the auto- angular power spectrum is computed.
        - distance (float): Comoving distance to the plane.
        - size (float): Size of the patch in Mpc/h.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the filtered angular modes (ell) and
        the angular power spectrum values (Cl).

    """
    # Mesh properties
    Nx, Ny = field1.shape
    Npix = Nx * Ny

    # Fourier transform of the field
    field1_ft = jnp.fft.fftn(field1)
    field2_ft = jnp.fft.fftn(field2)


    # Angular size of the patch
    theta = size / distance

    # Fundamental angular mode
    ell_fundamental = 2.0 * jnp.pi / theta

    # Unbinned power spectrum. Mix the two fields.
    power_2D = field1_ft
    power_2D = power_2D.at[...].mul(jnp.conj(field2_ft))
    power_2D = power_2D.at[...].set(jnp.abs(power_2D).astype(jnp.float32) / Npix**2.)
    power_2D = power_2D.astype(jnp.float32)

    # Fourier modes of the box
    kx = jnp.fft.fftfreq(Nx, d=1.0) * Nx
    ky = jnp.fft.fftfreq(Ny, d=1.0) * Ny
    kx = kx[:, None]
    ky = ky[None, :]
    k_modes = jnp.sqrt(kx**2 + ky**2)

    # Angular modes
    ell_vals = (k_modes * ell_fundamental).flatten()

    # Binned power spectrum
    ell_bins = jnp.arange(0.5, Ny // 2 + 1, 1.0) * ell_fundamental
    ell = 0.5 * (ell_bins[1:] + ell_bins[:-1])
    power = power_2D.flatten()
    binned_power, _ = jnp.histogram(ell_vals, weights=power, bins=ell_bins)
    mode_counts, _ = jnp.histogram(ell_vals, bins=ell_bins)

    # Normalize the binned power spectrum
    Cl = (binned_power / mode_counts) * theta**2

    return ell, Cl



# def cls_allbins(tomo_data, key, chunk_size=10):
#     def get_spec(index, tomo_data, key):
#         if do_noise:
#             tomo_data = noise_simulator(key, tomo_data)
#         ell,cl = compute_auto_cross_angular_power_spectrum(tomo_data[index[0]], tomo_data[index[1]],
#                                                     chi_source, Lgrid[0])
    
#         return jnp.histogram(ell[:cl_cut], weights=cl[:cl_cut], bins=OUTBINS)[0]
#     gps = partial(get_spec, tomo_data=tomo_data, key=key)
#     return nk.jax.vmap_chunked(gps, chunk_size=chunk_size)(indices)

