import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
import jax

from .basics import estimate_real_space_operator, calculate_mean_density, H_BAR, INVERSE_H_BAR

@jit
def base_GPE_function(
        state, times,
        energy_dispersion, photonic_potential, rhabbi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step):
    '''
    state is a vector of matrices (here they are 1D vectors) - photon and exciton wavefunctions
    '''
    # separating state
    photon_wf, exciton_wf = state
    
    # Calculate derivatives first
    kinetic_term = estimate_real_space_operator(energy_dispersion, photon_wf, N)
    gain_term = estimate_real_space_operator(gain_dispersion, exciton_wf, N)
    mean_exciton_density = calculate_mean_density(reverse_space_size, exciton_wf, spatial_step)

    photon_derivative = - 1j * INVERSE_H_BAR * (kinetic_term + photonic_potential * photon_wf + (rhabbi_split * 0.5) * exciton_wf)

    exciton_derivative = - 1j * INVERSE_H_BAR * (
        1j * scattering_rate * jnp.exp(-mean_exciton_density/saturation_exciton_density) * pumping_spatial_profile * gain_term +
        white_noise +
        (ee_interaction_constant * jnp.abs(exciton_wf)**2  + excitonic_potential) * exciton_wf + 
        (rhabbi_split * 0.5) * photon_wf
    )

    # Check for NaN/Inf in the results (not inputs)
    derivatives = jnp.concatenate([jnp.real(photon_derivative), 
                                 jnp.imag(photon_derivative),
                                 jnp.real(exciton_derivative),
                                 jnp.imag(exciton_derivative)])
    
    has_problem = jnp.any(jnp.isnan(derivatives)) | jnp.any(jnp.isinf(derivatives))

    def error_fn(_):
        # Use jax.debug.print for JIT-compatible printing
        jax.debug.print("Warning: NaN or Inf detected in derivatives")
        jax.debug.print("photon_wf: {}", photon_wf)
        jax.debug.print("exciton_wf: {}", exciton_wf)
        # Return NaN values to signal error
        return jnp.nan * photon_derivative, jnp.nan * exciton_derivative

    def success_fn(_):
        return photon_derivative, exciton_derivative

    return jax.lax.cond(has_problem, error_fn, success_fn, operand=None)

@jit
def solve_GPE(
        state, times,
        energy_dispersion, photonic_potential, rhabi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step):
    """
    for now I am using basic jax ODE solver, Runge-Kutta method will be used later
    """
    return odeint(base_GPE_function, state, times,
        energy_dispersion, photonic_potential, rhabi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step)