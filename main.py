import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import argparse
import time 
from jax.experimental.ode import odeint
# constants
H_BAR = 1
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
    return jnp.clip((low_pass+high_pass)/2, a_min=0, a_max=1)

def prepare_k_space(N, spatial_step):
    k_space_grid = 2 * jnp.pi * jnp.fft.fftfreq(N, spatial_step)
    return k_space_grid

def dispersion_function(k_space_grid):
    '''
    For now this function will give out 1st order linear dispersion as in Fig. 4(b) from main paper. This corresponds with LPB TE0 dispersion, NOT solitonic modes.
    '''
    # coefficients from said linear lasing mode
    a1 = -0.06611716023097772	
    a2 = 0.007386416723936695	
    a3 = -9.400442787281089e-05
    return a1 * k_space_grid + a2 * k_space_grid**2 + a3 * k_space_grid**3

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

# this should be optimized for jnp matrices. Is it?
@jit
def base_GPE_function(
        state, times,
        energy_dispersion, photonic_potential, rhabi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step):
    '''
    state is a vector of matrices (here they are 1D vectors) - photon and exciton wavefunctions
    '''
    # separating state
    photon_wf, exciton_wf = state

    # calculating values defined with numerical methods
    kinetic_term = estimate_real_space_operator(energy_dispersion, photon_wf, N)
    gain_term = estimate_real_space_operator(gain_dispersion, exciton_wf, N)
    mean_exciton_density = calculate_mean_density(reverse_space_size, exciton_wf, spatial_step)

    # calculating derivatives
    photon_derivative = - 1j * INVERSE_H_BAR * (kinetic_term + photonic_potential * photon_wf + (rhabi_split * 0.5) * exciton_wf)

    exciton_derivative = - 1j * INVERSE_H_BAR * (
        1j * scattering_rate * jnp.exp(-mean_exciton_density/saturation_exciton_density) * pumping_spatial_profile * gain_term +
        white_noise +
        (ee_interaction_constant * jnp.abs(exciton_wf)**2  + excitonic_potential) * exciton_wf + 
        (rhabi_split * 0.5) * photon_wf
    )

    return photon_derivative, exciton_derivative

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


def prepare_real_space(space_size, spatial_step):
    real_space = jnp.arange(0, space_size, spatial_step)
    return real_space

def generate_pumping_spatial_profile(real_space, space_size, N, steepness, width):
    center = space_size/2 * N
    size = width/space_size * N
    pumping_spatial_profile = band_pass_filter(real_space, steepness, center, width)
    return pumping_spatial_profile

def photonic_profile(k_space_grid):
    '''
    photonic dispersion according to Fig. 2(a), used as gain profile (maybe it incorrectly)
    '''
    a1 = 0.050507630878534444	
    a2 = 0.0019423639117504808	
    a3 = -3.0502383635017322e-05	
    return a1 * k_space_grid + a2 * k_space_grid**2 + a3 * k_space_grid**3


def generate_gain_dispersion(k_space_grid, detuning, exciton_energy, gain_width):
    '''
    according to the paper, page 5 gain profile i 2 meV wide, centered at polariton states corresponding to detuning -5 meV.
    for detuning look at Fig. 2(a), where excitons are horizontal and photons - linear. Pumping intensity here is below excitonic line and falls for 2 meV range of photonic dispersion.

    gain_width - single valued width in energy units, taken on photon energy dispersion
    detuning - single valued detuning in energy units measured from exciton energy
    exciton_energy - single valued excitonic energy (constant function of momentum) (in the paper it is equal E = 3.475 eV)

    returns energy values between given interval, corresponding to k_space_grid values.
    '''
    gain_dispersion = photonic_profile(k_space_grid)
    lower_boundary = exciton_energy + detuning - gain_width/2
    upper_boundary = exciton_energy + detuning + gain_width/2
    #print(lower_boundary)
    #print(upper_boundary)
    
    mask = (gain_dispersion < lower_boundary) | (gain_dispersion > upper_boundary)
    gain_dispersion = gain_dispersion.at[mask].set(0)
    
    #print(gain_dispersion)
    return gain_dispersion

@jit
def calculate_mean_density(reverse_space_size, wavefunction, spatial_step):
    abs_wav = jnp.abs(wavefunction)
    result = jnp.sum(abs_wav * abs_wav) * spatial_step * reverse_space_size
    return result

def generate_potentials(real_space, excitonic_scale, pumping_amplitude, pumping_spatial_profile, photonic_scale, excitonic_loss_amplitude, photonic_loss_amplitude, start, stop, steepness):
    '''
    I assume rescaled and filter-like functions at borders 
    '''
    # Create a window function that is 1 between start and stop, and 0 outside
    window_photonic = low_pass_filter(real_space, steepness, start) + high_pass_filter(real_space, steepness, stop)
    window_excitonic = low_pass_filter(real_space, steepness, start) + high_pass_filter(real_space, steepness, stop)
    
    # Photonic potential (real and imaginary parts)
    photonic_potential = window_photonic * photonic_scale  # Higher potential outside
    photonic_potential += 1j * window_photonic * photonic_loss_amplitude
    
    # Excitonic potential (real and imaginary parts)
    excitonic_potential = window_excitonic * excitonic_scale + pumping_amplitude * pumping_spatial_profile
    excitonic_potential += 1j * window_excitonic * excitonic_loss_amplitude
    
    return photonic_potential, excitonic_potential

def generate_initial_state(N):
    photon_wf = jnp.zeros(N) + 1j * jnp.zeros(N)
    exciton_wf = jnp.zeros(N) + 1j * jnp.zeros(N)
    return photon_wf, photon_wf

def generate_white_noise(rng_key, N, noise_amplitude):
    noise = jax.random.normal(rng_key, [N])
    rescaled_noise = noise/jnp.abs(noise) * noise_amplitude
    return rescaled_noise

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
    ):


    k_space_grid = prepare_k_space(N, spatial_step)
    real_space = prepare_real_space(greater_scope_size, spatial_step)

    potential_well_start = (greater_scope_size-space_size)/2
    potential_well_stop = (greater_scope_size+space_size)/2

    print(potential_well_start)
    print(potential_well_stop)

    initial_state = generate_initial_state(N)
    energy_dispersion = dispersion_function(k_space_grid)
    pumping_spatial_profile = generate_pumping_spatial_profile(real_space, space_size, N, steepness, pumping_spatial_profile_width)
    photonic_potential, excitonic_potential = generate_potentials(real_space, excitonic_scale, pumping_amplitude, pumping_spatial_profile, photonic_scale, excitonic_loss_amplitude, photonic_loss_amplitude, potential_well_start, potential_well_stop, steepness)
    gain_dispersion = generate_gain_dispersion(k_space_grid, detuning, exciton_energy, gain_width)
    white_noise = generate_white_noise(rng_key, N, noise_amplitude)
    times = jnp.arange(0, T + time_step, time_step)
    photonic_profile_value = photonic_profile(k_space_grid)
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
        "photonic_profile": photonic_profile_value
    }
    return preamble

def plot_ode_solution_heatmap(solution, time_points=None, space_points=None, 
                             space_trim=(None, None), cmap='viridis', 
                             aspect='auto', title='ODE Solution Heatmap',
                             xlabel='Time', ylabel='Space Position',
                             figsize=(10, 6), shading='auto', 
                             colorbar_label='Solution Value',
                             plot_abs_squared=True):
    """
    Plot 2D ODE solution as a heatmap with time on x-axis and space on y-axis.
    
    Parameters:
    -----------
    solution : 2D numpy array
        Solution matrix with shape (time_size, space_size)
    time_points : array-like, optional
        Time coordinates (if None, uses indices)
    space_points : array-like, optional
        Space coordinates (if None, uses indices)
    space_trim : tuple (start, end), optional
        Spatial indices to trim (None means no trim)
    cmap : str, optional
        Matplotlib colormap name
    aspect : str or float, optional
        Aspect ratio of the plot
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size
    shading : str, optional
        Shading method for pcolormesh ('auto', 'flat', 'gouraud')
    colorbar_label : str, optional
        Label for the colorbar
    """
    
    if isinstance(solution, tuple):
        solution = jnp.array(solution)

    
    if jnp.iscomplexobj(solution):
        if plot_abs_squared:
            solution = jnp.abs(solution)**2
        else:
            solution = jnp.abs(solution)

    # Check and reshape solution if needed
    if solution.ndim == 3:
        # If solution has shape (time, space, channels), take first channel
        if solution.shape[2] == 2:  # might be complex stored as real/imag
            solution = solution[..., 0] + 1j*solution[..., 1]
            solution = jnp.abs(solution)**2 if plot_abs_squared else jnp.abs(solution)
        else:
            solution = solution[..., 0]  # take first channel

    # Trim spatial dimension if requested
    space_start, space_end = space_trim
    if space_start is None:
        space_start = 0
    if space_end is None:
        space_end = solution.shape[1]
        
    trimmed_solution = solution[:, space_start:space_end]
    
    # Ensure coordinates are 1D arrays
    time_points = jnp.asarray(time_points).flatten()
    space_points = jnp.asarray(space_points).flatten()

    # Create figure
    plt.figure(figsize=figsize)
    
    # Trim space points if they were provided
    if isinstance(space_points, (jnp.ndarray, list)):
        space_points = space_points[space_start:space_end]
    
    # Create the plot
    if shading == 'imshow':
        img = plt.imshow(trimmed_solution.T,
                        aspect=aspect,
                        origin='lower',
                        cmap=cmap,
                        extent=[time_points[0], time_points[-1],
                                space_points[0], space_points[-1]])
    else:
        img = plt.pcolormesh(time_points, space_points,
                            trimmed_solution.T,
                            shading=shading,
                            cmap=cmap)
    
    # Add labels and colorbar
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    cbar = plt.colorbar(img)
    cbar.set_label(colorbar_label)
    
    plt.tight_layout()
    return img

def plot_ode_solution_heatmap(solution, time_points=None, space_points=None, 
                            component=0,  # Which component to plot if multiple exist
                            cmap='viridis', title='ODE Solution Heatmap',
                            xlabel='Time', ylabel='Space Position',
                            figsize=(10, 6)):
    """
    solution: Tuple containing complex arrays with shape (components, time, space)
    component: Which solution component to plot (0 or 1, or 2 if it should be a complete probability function)
    """
    # Convert tuple element to numpy array and select component
    if component == 0 or component == 1:
        result = jnp.array(solution[component])
        plot_data = jnp.abs(result)**2
    if component == 2:
        result1 = jnp.array(solution[0])
        result2 = jnp.array(solution[1])
        plot_data = jnp.abs(result1**2) + jnp.abs(result2**2)
    
    # Compute |ψ|²
    
    # Handle coordinates
    if time_points is None:
        time_points = jnp.arange(plot_data.shape[0])
    if space_points is None:
        space_points = jnp.arange(plot_data.shape[1])
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.pcolormesh(time_points, space_points, plot_data.T,
                 shading='auto', cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='|ψ|²')
    plt.tight_layout()

def main():
    rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    space_size = 30 # micrometers

    # for making k_space finer, bigger space without (or with near-zero) wavefunction will be considered, here we define its size
    greater_scope_size = 210
    
    reverse_space_size = 1/greater_scope_size
    #spatial_step = 31.25 * 1e-3 # micrometers
    spatial_step = 0.05 # micrometers
    time_step = 1e-16 # seconds
    number_of_times = 10
    T = number_of_times * time_step
    N_space = int(space_size/spatial_step)
    N = int(greater_scope_size/spatial_step)

    if space_size/spatial_step != N_space or greater_scope_size/spatial_step != N :
        print("Non integer number of pixels")
        return 1

    steepness = 4
    pumping_spatial_profile_width = 1
    excitonic_scale = 20 # potential size at the border amounts to half scale
    photonic_scale = 20 # potential size at the border amounts to half scale
    pumping_amplitude = 1
    excitonic_loss_amplitude = 1
    photonic_loss_amplitude = 1
    rhabi_split = 1
    scattering_rate = 1
    saturation_exciton_density = 1
    detuning = -0.
    exciton_energy = 1
    gain_width = 0.1
    noise_amplitude = 0.01
    ee_interaction_constant = 0.1
    

    preamble = create_preamble(
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
    )
    k_space_grid = preamble["k_space_grid"]
    real_space =  preamble["real_space"]
    initial_state = preamble["initial_state"]
    energy_dispersion = preamble["energy_dispersion"]
    pumping_spatial_profile = preamble["pumping_spatial_profile"]
    photonic_potential = preamble["photonic_potential"]
    excitonic_potential = preamble["excitonic_potential"]
    gain_dispersion = preamble["gain_dispersion"]
    white_noise = preamble["white_noise"]
    times = preamble["times"]
    photonic_profile = preamble["photonic_profile"]

    print("length of k_space:\t",len(k_space_grid), "\nlength of real space:\t", N)
    print("number of grid elements in gain dispersion:\t", len(jnp.nonzero(gain_dispersion)), "\tplacement:\t",k_space_grid[jnp.nonzero(gain_dispersion)],"\tvalue:\t", gain_dispersion[jnp.nonzero(gain_dispersion)])
    print("maximum value of k space:\t", jnp.max(k_space_grid))
    print("k space step size:\t", (-jnp.min(k_space_grid)+jnp.max(k_space_grid))/len(k_space_grid))
    print("gain width:\t", gain_width)

    """
    plt.figure(figsize=(12, 6))
    # First subplot (plot 1)
    plt.subplot(1, 2, 1) 
    plt.plot(k_space_grid, energy_dispersion, '*', label='energy_dispersion')
    plt.plot(k_space_grid, pumping_spatial_profile, '*', label='pumping_spatial_profile')
    plt.plot(k_space_grid, gain_dispersion, '*', label='gain_dispersion')
    #plt.plot(k_space_grid, photonic_profile, '*', label='photonic_profile')
    plt.legend()
    
    x_low, x_high = 44,47
    x_low, x_high = 10,20
    plt.xlim((x_low, x_high))

    mask = (k_space_grid >= x_low) & (k_space_grid <= x_high)
    all_y_in_range = jnp.concatenate([
        energy_dispersion[mask],
        pumping_spatial_profile[mask],
        gain_dispersion[mask]
    ])
    padding = 0.05 * (-jnp.min(all_y_in_range) + jnp.max(all_y_in_range))
    plt.ylim((jnp.min(all_y_in_range)-padding, jnp.max(all_y_in_range)+padding))

    plt.title('k-space plots')

    # Second subplot (plot 2)
    plt.subplot(1, 2, 2) 
    plt.plot(real_space, photonic_potential, '*', label='photonic_potential')
    #plt.plot(real_space, -1j * photonic_potential, '*', label='photonic_potential imaginary')
    plt.plot(real_space, excitonic_potential, '*', label='excitonic_potential')
    #plt.plot(real_space, -1j * excitonic_potential, '*', label='excitonic_potential imaginary')
    plt.legend()
    plt.title('Real-space plots')

    plt.tight_layout()
    plt.show()
    """


    estimate_real_space_operator(energy_dispersion, initial_state[1], N)
    estimate_real_space_operator(gain_dispersion, initial_state[0], N)

    
    solution = solve_GPE(
        initial_state, times,
        energy_dispersion, photonic_potential, rhabi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step)
    
    space_start = int((N-N_space)/2)
    space_end = int((N+N_space)/2)
    print(int(space_start))
    print(int(space_end))

    """
    plot_ode_solution_heatmap(jnp.array(solution), time_points=times, space_points=real_space, 
                             space_trim=(space_start, space_end), cmap='viridis', 
                             aspect='auto', title='ODE Solution Heatmap',
                             xlabel='Time', ylabel='Space Position',
                             figsize=(10, 6), shading='auto', 
                             colorbar_label='Solution Value')
    plt.show()
    """
    # Plot first component
    plot_ode_solution_heatmap(solution, component=0)
    plt.show()
    # Plot second component with custom time axis
    time_axis = jnp.linspace(0, 10, 12)  # 12 time points
    plot_ode_solution_heatmap(solution, component=1, time_points=time_axis)
    plt.show()
    return 0

if __name__ == "__main__":
    main()