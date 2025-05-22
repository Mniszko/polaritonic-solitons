import jax.numpy as jnp

from .basics import band_pass_filter, prepare_k_space, prepare_real_space, estimate_real_space_operator, calculate_mean_density, generate_white_noise, gap_pass_filter, H_BAR, INVERSE_H_BAR


def generate_pumping_spatial_profile(real_space, space_size, N, steepness, width):
    center = space_size/2 
    size = width/space_size 
    pumping_spatial_profile = band_pass_filter(real_space, steepness, center, width)
    return pumping_spatial_profile

def photonic_profile_second_order(k_space_grid):
    a0 = 1.064848204761849
    a1 = 0.052822773021815574
    return a0 + a1 * k_space_grid


def generate_gain_dispersion(k_space_grid, detuning, exciton_energy, gain_width):
    '''
    according to the paper, page 5 gain profile i 2 meV wide, centered at polariton states corresponding to detuning -5 meV.
    for detuning look at Fig. 2(a), where excitons are horizontal and photons - linear. Pumping intensity here is below excitonic line and falls for 2 meV range of photonic dispersion.

    gain_width - single valued width in energy units, taken on photon energy dispersion
    detuning - single valued detuning in energy units measured from exciton energy
    exciton_energy - single valued excitonic energy (constant function of momentum) (in the paper it is equal E = 3.475 eV)

    returns energy values between given interval, corresponding to k_space_grid values.
    '''
    gain_dispersion = photonic_profile_second_order(k_space_grid)
    lower_boundary = exciton_energy + detuning - gain_width/2
    upper_boundary = exciton_energy + detuning + gain_width/2
    #print(lower_boundary)
    #print(upper_boundary)
    
    mask = (gain_dispersion < lower_boundary) | (gain_dispersion > upper_boundary)
    gain_dispersion = gain_dispersion.at[mask].set(0)
    
    #print(gain_dispersion)
    return gain_dispersion

def generate_potentials(real_space, excitonic_scale, pumping_amplitude, pumping_spatial_profile, photonic_scale, excitonic_loss_amplitude, photonic_loss_amplitude, start, stop, steepness):
    '''
    I assume rescaled and filter-like functions at borders 
    '''
    # Create a window function that is 1 between start and stop, and 0 outside
    window_photonic = gap_pass_filter(real_space, steepness, start, stop)
    window_excitonic = gap_pass_filter(real_space, steepness, start, stop)
    
    # Photonic potential (real and imaginary parts)
    photonic_potential = window_photonic * photonic_scale  # Higher potential outside
    photonic_potential += 1j * window_photonic * photonic_loss_amplitude
    
    # Excitonic potential (real and imaginary parts)
    excitonic_potential = window_excitonic * excitonic_scale + pumping_amplitude * pumping_spatial_profile
    excitonic_potential += 1j * window_excitonic * excitonic_loss_amplitude
    
    return photonic_potential, excitonic_potential

def generate_initial_state(space, space_size, greater_scope_size, N):
    photon_wf = jnp.zeros(N) + 1j * jnp.zeros(N)
    exciton_wf = jnp.zeros(N) + 1j * jnp.zeros(N)
    return photon_wf, photon_wf

def generate_constant_initial_state(space, space_size, greater_scope_size, N):
    exciton_amplitude = 1
    photon_amplitude = 1
    photon_wf = jnp.ones(N) + 1j * jnp.zeros(N)
    exciton_wf = jnp.ones(N) + 1j * jnp.zeros(N)
    return photon_wf*photon_amplitude, photon_wf*exciton_amplitude

def generate_solitonic_initial_state(space, space_size, greater_scope_size, N):
    tempPhotonic = 0.7
    tempExcitonic = 0.3 # these ones should be replaced by spectrally defined ones with correct hoppfield profiles
    amplitude = 1
    photon_wf = (1/np.cosh(1 * (space - greater_scope_size/2)))*np.exp(1j*space)
    exciton_wf = (1/np.cosh(1 * (space - greater_scope_size/2)))*np.exp(1j*space)
    return tempExcitonic * exciton_wf * amplitude, tempPhotonic * photon_wf * amplitude

def dispersion_function(photonic_profile_function, exciton_energy, k_space_grid, rhabbi_split):
    """
    LPB dispersion based on Fig.4b from main article. 
    """
    photon_energy = photonic_profile_function(k_space_grid)

    decay_exciton, decay_photon = 0, 0 # unused due to fitting errors

    decaying_energy_exciton = exciton_energy - H_BAR * 1j * decay_exciton
    decaying_energy_photon = photon_energy - H_BAR * 1j * decay_photon

    energy_term = (exciton_energy + photon_energy) * 0.5

    sqrt_energy = photon_energy - photon_energy
    sqrt_rhabbi = H_BAR * rhabbi_split
    sqrt_term = jnp.sqrt(sqrt_energy**2 + rhabbi_split**2) * 0.5
    return jnp.real(energy_term - sqrt_term)

def hopfield_coefficients_from_dispersion(photonic_dispersion_function, exciton_energy, k_space_grid, rhabbi_split):
    """
    calculates hopfield coefficient in function k_space
    """
    photonic_dispersion = photonic_dispersion_function(k_space_grid)
    energy_term = photonic_dispersion - exciton_energy
    rhabbi_term = H_BAR * rhabbi_split
    nominative = 1 / jnp.sqrt(rhabbi_term**2 + energy_term**2)
    photon_coef = rhabbi_term * nominative
    exciton_coef = energy_term * nominative

    return photon_coef, exciton_coef


def create_preamble(
        rng_key,
        space_size,
        greater_scope_size,
        spatial_step,
        time_step,
        T,
        N,
        steepness,
        pumping_spatial_profile_width,
        excitonic_scale,
        photonic_scale,
        pumping_amplitude,
        excitonic_loss_amplitude,
        photonic_loss_amplitude,
        detuning,
        exciton_energy,
        gain_width,
        noise_amplitude,
        rhabbi_split,
        photonic_profile_function,
        initial_state_function
    ):


    k_space_grid = prepare_k_space(N, spatial_step)
    real_space = prepare_real_space(greater_scope_size, spatial_step)

    potential_well_start = (greater_scope_size-space_size)/2
    potential_well_stop = (greater_scope_size+space_size)/2

    initial_state = initial_state_function(real_space, space_size, greater_scope_size, N)
    energy_dispersion = dispersion_function(photonic_profile_function, exciton_energy, k_space_grid, rhabbi_split)
    pumping_spatial_profile = generate_pumping_spatial_profile(real_space, greater_scope_size, N, steepness, pumping_spatial_profile_width)
    photonic_potential, excitonic_potential = generate_potentials(real_space, excitonic_scale, pumping_amplitude, pumping_spatial_profile, photonic_scale, excitonic_loss_amplitude, photonic_loss_amplitude, potential_well_start, potential_well_stop, steepness)
    gain_dispersion = generate_gain_dispersion(k_space_grid, detuning, exciton_energy, gain_width)
    white_noise = generate_white_noise(rng_key, N, noise_amplitude)
    times = jnp.arange(0, T + time_step, time_step)
    photonic_profile_value = photonic_profile_second_order(k_space_grid)

    hopfield_photon_coef, hopfield_exciton_coef = hopfield_coefficients_from_dispersion(photonic_profile_function, exciton_energy, k_space_grid, rhabbi_split)
    preamble = {
        "k_space_grid" : k_space_grid,
        "real_space" : real_space,
        "initial_state" : initial_state,
        "energy_dispersion" : energy_dispersion,
        "pumping_spatial_profile" : pumping_spatial_profile,
        "photonic_potential" : photonic_potential,
        "excitonic_potential" : excitonic_potential,
        "gain_dispersion" : gain_dispersion,
        "white_noise" : white_noise,
        "times" : times,
        "photonic_profile": photonic_profile_value,
        "hoppfield_photon_coef": hopfield_photon_coef,
        "hoppfield_exciton_coef": hopfield_exciton_coef
    }
    return preamble


"""
def photonic_profile_third_order(k_space_grid):
    '''
    photonic dispersion according to Fig. 2(a), used as gain profile (maybe it incorrectly)
    '''
    a1 = 0.050507630878534444	
    a2 = 0.0019423639117504808	
    a3 = -3.0502383635017322e-05	
    return a1 * k_space_grid + a2 * k_space_grid**2 + a3 * k_space_grid**3
"""

'''
def hopfield_coefficients_from_energy(photonic_dispersion, exciton_energy, rhabbi_split):
    """
    calculates hopfield coefficient in function of directly given energy values
    """
    energy_term = photonic_dispersion - exciton_energy
    rhabbi_term = H_BAR * rhabbi_split
    nominative = 1 / jnp.sqrt(rhabbi_term**2 + energy_term**2)
    photon_coef = rhabbi_term * nominative
    exciton_coef = energy_term * nominative

    return photon_coef, exciton_coef
'''

"""
def hopfield_coefficients_linear_interpolation(energy):
    energies = [3.05924, 3.06561, 3.07196, 3.07829, 3.08459, 3.09086, 3.09711, 3.10333, 3.10952, 3.11569, 3.12182, 3.12794, 3.13402, 3.14008, 3.14611, 3.15211, 3.15808, 3.16402, 3.16992, 3.1758,  3.18165, 3.18748, 3.19327, 3.19902, 3.20475, 3.21045, 3.21611, 3.22174, 3.22735, 3.23291, 3.23844, 3.24395, 3.24941, 3.25485, 3.26024, 3.2656, 3.27094, 3.27623, 3.28148, 3.28671, 3.29188, 3.29704, 3.30214, 3.30721, 3.31224, 3.3172,  3.32224, 3.32689, 3.35109, 3.35532, 3.35993, 3.36433, 3.36871, 3.37297, 3.37716, 3.38124, 3.38523, 3.3891,  3.39286, 3.39652, 3.40004, 3.40346, 3.40675, 3.40991, 3.41295, 3.41587, 3.41865, 3.42131, 3.42386, 3.42627, 3.42857, 3.43075, 3.43283, 3.43479, 3.43664, 3.4384,  3.44005, 3.44162, 3.4431,  3.44449, 3.4458, 3.44704, 3.44821, 3.44931, 3.45034, 3.45132, 3.45225, 3.45312, 3.45394, 3.45472, 3.45546, 3.45615, 3.4568,  3.45742, 3.458,   3.45855, 3.45907, 3.45957, 3.46004, 3.46048, 3.46091, 3.46132, 3.46171, 3.46209]
    photon_coeficients = [0.99964, 0.99963, 0.99961, 0.9996,  0.99958, 0.99956, 0.99954, 0.99953, 0.9995, 0.99948, 0.99946, 0.99944, 0.99941, 0.99938, 0.99935, 0.99932, 0.99929, 0.99926, 0.99922, 0.99918, 0.99914, 0.9991,  0.99904, 0.99899, 0.99894, 0.99888, 0.99881, 0.99875, 0.99867, 0.99859, 0.99851, 0.9984,  0.9983,  0.99819, 0.99805, 0.99792, 0.99776, 0.99758, 0.99741, 0.99717, 0.99694, 0.99667, 0.99634, 0.996,   0.99561, 0.99499, 0.99502, 0.9928,  0.99052, 0.98592, 0.98455, 0.98149, 0.97841, 0.97425, 0.96966, 0.9638,  0.95718, 0.94919, 0.93994, 0.9294,  0.91706, 0.90346, 0.88783, 0.87082, 0.85174, 0.83144, 0.80908, 0.78559, 0.76082, 0.73476, 0.70832, 0.68104, 0.65393, 0.62726, 0.60076, 0.57424, 0.54751, 0.52049, 0.49311, 0.46542, 0.43749, 0.40949, 0.38159, 0.35402, 0.32702, 0.30082, 0.27563, 0.25164, 0.22899, 0.2078, 0.18812, 0.16999, 0.15337, 0.13823, 0.1245,  0.1121,  0.10093, 0.09088, 0.08188, 0.0738,  0.06657, 0.06009, 0.05428, 0.04907]

    low = -1
    for i in range(len(energies)):
        if energies[i] < energy and energies[i+1] > energy:
            low = i
            high = energies[i+1]
        else:
            continue
    if low == -1 or low+1 == len(energies):
        print('Energy value parsed outside of sampled region.')
        return NULL
    
    photon_coef = photon_coeficients[low] + (x - energies[low]) * (photon_coeficients[low+1] - photon_coeficients[low])/(energies[low+1] - energies[low])

    exciton_coef = 1 - photon_coef
    return photon_coef, exciton_coef
"""