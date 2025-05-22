import matplotlib.pyplot as plt
import jax.numpy as jnp

from .Preamble import hopfield_coefficients_from_dispersion
from .basics import estimate_real_space_operator

def polariton_wavefunction(hopfield_photon_coef, hopfield_exciton_coef, photon_wf, exciton_wf, N):
    photon_part = estimate_real_space_operator(hopfield_photon_coef, photon_wf, N)
    exciton_part = estimate_real_space_operator(hopfield_exciton_coef, exciton_wf, N)
    lower_polariton_wavefunction = photon_part + exciton_part
    return lower_polariton_wavefunction

def plot_ode_solution_heatmap(savefig_title, solution, N, time_points=None, space_points=None, 
                            component=0,  # Which component to plot if multiple exist
                            cmap='viridis', title='ODE Solution Heatmap',
                            xlabel='Time [fs]', ylabel='Space [μm]',
                            figsize=(10, 6), hoppfield_photon_coef = None, hoppfield_exciton_coef = None):
    """
    solution: Tuple containing complex arrays with shape (components, time, space)
    component: Which solution component to plot (0 or 1, or 2 if it should be a complete probability function)
    """

    plt.figure(figsize=figsize)

    # Convert tuple element to numpy array and select component
    if component == 0:
        result = jnp.array(solution[component])
        plt.title('Photonic probability')
        plot_data = jnp.abs(result)**2
    elif component == 1:
        result = jnp.array(solution[component])
        plt.title('Excitonic probability')
        plot_data = jnp.abs(result)**2
    elif component == 2:
        if hoppfield_photon_coef == None or hoppfield_exciton_coef == None:
            raise ValueError("Required for polaritonic wavefunction calculation hopfield_photon_coef or hopfield_exciton_coef is missing in plotting function.")
        photon_wf = jnp.array(solution[0])
        exciton_wf = jnp.array(solution[1])
        lower_polariton_wf = polariton_wavefunction(hoppfield_photon_coef, hoppfield_exciton_coef, photon_wf, exciton_wf, N)
        plt.title('LPB polariton probability')
        plot_data = jnp.abs(lower_polariton_wf)**2 
    else:
        raise ValueError(f'Plotting error.\n\nNo component with index {component} exists in solution.')
        return 
        
    
    if time_points is None:
        time_points = jnp.arange(plot_data.shape[0])
    if space_points is None:
        space_points = jnp.arange(plot_data.shape[1])
    
    plt.pcolormesh(time_points, space_points, plot_data.T,
                 shading='auto', cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='|ψ|²')
    plt.tight_layout()
    plt.savefig(savefig_title, transparent=True)
    return

def plot_real_and_momentum(savefig_title, x_space, y_space, xlabel, ylabel, xlim, title, padding_factor=0.05):
    plt.figure(figsize=(10, 6))
    plt.plot(x_space, y_space, lw=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim[0], xlim[1])

    mask = (x_space >= xlim[0]) & (x_space <= xlim[1])
    y_in_range = y_space[mask]
    
    if len(y_in_range) > 0:
        y_min = jnp.min(y_in_range)
        y_max = jnp.max(y_in_range)
        y_range = y_max - y_min
        
        y_padding = y_range * padding_factor
        plt.ylim(y_min - y_padding, y_max + y_padding)
    else:
        print("Warning: No data points within the specified xlim range.")
    
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{savefig_title}.png", transparent=True)
    return

def plot_energy_momentum(solution, time_step, spatial_step, hoppfield_photon_coef, hoppfield_exciton_coef, N):
    photon_wf = jnp.array(solution[0])
    exciton_wf = jnp.array(solution[1])
    wavefunction_evolution = polariton_wavefunction(hoppfield_photon_coef, hoppfield_exciton_coef, photon_wf, exciton_wf, N)
    wavefunction_transformed = jnp.fft.fftshift(jnp.fft.fft2(wavefunction_evolution))
    spectrum = jnp.abs(wavefunction_transformed)**2
    Nx, Nt = wavefunction_evolution.shape
    momenta = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, spatial_step))
    energies = jnp.fft.fftshift(jnp.fft.fftfreq(Nt, time_step)) 

    plt.figure(figsize=(10,10))
    plt.imshow(spectrum.T, aspect='auto', extent=[energies[0], energies[-1], momenta[0], momenta[-1]])
    plt.ylabel("Energy [eV]")
    plt.xlabel("Momentum [μm⁻¹]")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("energy_spectrum.png", transparent=True)
    return