import jax.numpy as jnp
from jax import jit
import jax

H_BAR = 6.582119569e-1 # eV * fs
INVERSE_H_BAR = 1/H_BAR

def low_pass_filter(real_space, steepness, ridge_position):
    """Smooth step function that transitions from 1 to 0 around ridge_position"""
    return 1/(jnp.exp(steepness*(real_space - ridge_position))+1)

def high_pass_filter(real_space, steepness, ridge_position):
    """Smooth step function that transitions from 0 to 1 around ridge_position"""
    return 1/(jnp.exp(-steepness*(real_space - ridge_position))+1)

def band_pass_filter(real_space, steepness, center, width):
    low_pass = low_pass_filter(real_space, steepness, center+width/2)
    high_pass = high_pass_filter(real_space, steepness, center-width/2)
    return jnp.clip((low_pass+high_pass-1), a_min=0, a_max=1) 

def gap_pass_filter(real_space, steepness, start, stop):
    low_pass = low_pass_filter(real_space, steepness, start)
    high_pass = high_pass_filter(real_space, steepness, stop)
    return jnp.clip((low_pass+high_pass), a_min=0, a_max=1) 

def prepare_k_space(N, spatial_step):
    k_space_grid = 2 * jnp.pi * jnp.fft.fftfreq(N, spatial_step)
    return k_space_grid

def prepare_real_space(space_size, spatial_step):
    real_space = jnp.arange(0, space_size, spatial_step)
    return real_space

@jit
def estimate_real_space_operator(dispersion, wavefunction, N):
    '''
    function for general operator estimation with dispersion in k-space.
    assumers dispersion is already converted from raw data to appropreate vector
    '''
    norm_factor = jnp.sqrt(N)
    momentum_wavefunction = jnp.fft.fft(wavefunction) / norm_factor
    momentum_operator = dispersion * momentum_wavefunction
    real_operator = jnp.fft.ifft(momentum_operator) * norm_factor
    return real_operator

@jit
def calculate_mean_density(reverse_space_size, wavefunction, spatial_step):
    abs_wav = jnp.abs(wavefunction)
    result = jnp.sum(abs_wav * abs_wav) * spatial_step * reverse_space_size
    return result

def generate_white_noise(rng_key, N, noise_amplitude):
    noise = jax.random.normal(rng_key, [N])
    rescaled_noise = noise/jnp.abs(noise) * noise_amplitude
    return rescaled_noise