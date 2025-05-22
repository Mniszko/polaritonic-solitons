import argparse
import jax
import jax.numpy as jnp
import time 

import matplotlib.pyplot as plt

from src import create_preamble, generate_initial_state, generate_constant_initial_state, generate_solitonic_initial_state, plot_ode_solution_heatmap, solve_GPE, photonic_profile_second_order, plot_real_and_momentum, plot_energy_momentum

def main():
    #rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    rng_key = jax.random.PRNGKey(1)
    print("Using non random rng_key!")
    space_size = 30 # micrometers

    # for making k_space finer, bigger space without (or with near-zero) wavefunction will be considered, here we define its size
    # greater_scope_size = 210
    greater_scope_size = 90

    reverse_space_size = 1/greater_scope_size
    spatial_step = 31.25 * 1e-3 # micrometers
    #spatial_step = 0.05 # micrometers
    time_step = 0.1 # femtoseconds
    number_of_times = 60
    T = number_of_times * time_step
    N_space = int(space_size/spatial_step)
    N = int(greater_scope_size/spatial_step)

    if space_size/spatial_step != N_space or greater_scope_size/spatial_step != N :
        print("Non integer number of pixels")
        return 1

    steepness = 4
    pumping_spatial_profile_width = 10
    excitonic_scale = 20 # potential size at the border amounts to half scale
    photonic_scale = 20 # potential size at the border amounts to half scale
    pumping_amplitude = 1
    excitonic_loss_amplitude = 1
    photonic_loss_amplitude = 1
    rhabbi_split = 0.0773 # eV, value from LPB fit with straight line photonic profile
    scattering_rate = 1
    saturation_exciton_density = 1
    detuning = 0.
    exciton_energy = 3.475 # eV
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
        rhabbi_split,
        photonic_profile_second_order,
        generate_initial_state
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
    hoppfield_photon_coef = preamble["hoppfield_photon_coef"]
    hoppfield_exciton_coef = preamble["hoppfield_exciton_coef"]

    """
    real_xlim = [0,greater_scope_size]
    momentum_xlim = [30,60]
    plot_real_and_momentum("pumping_profile", real_space, pumping_spatial_profile, "Space (μm)", "Potential (eV)", real_xlim, "Pumping Spatial Profile")
    plot_real_and_momentum("excitonic_potential", real_space, jnp.real(excitonic_potential), "Space (μm)", "Potential (eV)", real_xlim, "Excitonic Potential, real part")
    plot_real_and_momentum("photonic_potential", real_space, jnp.real(photonic_potential), "Space (μm)", "Potential (eV)", real_xlim, "Photonic Potential, real part")
    plot_real_and_momentum("gain_dispersion", k_space_grid, gain_dispersion, "Momentum (μm⁻¹)", "Potential (eV)", momentum_xlim, "Gain Dispersion")
    plot_real_and_momentum("noise", real_space, white_noise, "Space (μm)", "Potential (eV)", real_xlim, "White Noise")
    plot_real_and_momentum("polaritonic_dispersion", k_space_grid, energy_dispersion, "Momentum (μm⁻¹)", "Energy (eV)", momentum_xlim, "Polaritonic Dispersion")
    plot_real_and_momentum("photonic_dispersion", k_space_grid, photonic_profile, "Momentum (μm⁻¹)", "Energy (eV)", momentum_xlim, "Photonic Dispersion")
    plot_real_and_momentum("hoppfield_photon", k_space_grid, hoppfield_photon_coef, "Momentum (μm⁻¹)", "[arb.]", [0,100], "Hopfield Coefficient - Photon")
    plot_real_and_momentum("hoppfield_exciton", k_space_grid, hoppfield_exciton_coef, "Momentum (μm⁻¹)", "[arb.]", [0,100], "Hopfield Coefficient - Exciton")
    """

    start_time = time.time()
    solution = solve_GPE(
        initial_state, times,
        energy_dispersion, photonic_potential, rhabbi_split, 
        scattering_rate, saturation_exciton_density, pumping_spatial_profile, gain_dispersion, white_noise, ee_interaction_constant, excitonic_potential, exciton_energy,
        N, reverse_space_size, spatial_step)
    end_time = time.time()

    space_start = int((N-N_space)/2)
    space_end = int((N+N_space)/2)
    print(f"time taken to integrate {number_of_times} timepoints with space of size {N}:\t{end_time-start_time}")
    print(f"so per one timepoint:\t{(end_time-start_time)/number_of_times}")
    print("above barrier buffer:\t", int(N/2-N_space/2))
    print("whole size:\t\t", N)
    print("relevant space size:\t", N_space)
    
    plot_ode_solution_heatmap("photon.png", solution, N, component=0, time_points=times, space_points=real_space, title="Photonic Wavefunction")
    plot_ode_solution_heatmap("exciton.png", solution, N, component=1, time_points=times, space_points=real_space, title="Excitonic Wavefunction")
    plot_ode_solution_heatmap("polariton.png", solution, N, component=2, hoppfield_photon_coef=hoppfield_photon_coef, hoppfield_exciton_coef=hoppfield_exciton_coef, time_points=times, space_points=real_space, title="Polaritonic Wavefunction")    
    
    plot_energy_momentum(solution, time_step, spatial_step, hoppfield_photon_coef, hoppfield_exciton_coef, N)
    """
    photon_wf = jnp.array(solution[0])
    exciton_wf = jnp.array(solution[1])
    lower_polariton_wf = polariton_wavefunction(hoppfield_photon_coef, hoppfield_exciton_coef, photon_wf, exciton_wf, N)

    dt = time_step
    dx = spatial_step
    psi = lower_polariton_wf
    # Perform 2D FFT and shift zero-frequency to center
    psi_fft = jnp.fft.fftshift(jnp.fft.fft2(psi))

    # Compute magnitude (for visualization)
    spectrum = jnp.abs(psi_fft)**2  # Power spectrum

    Nx, Nt = psi.shape
    k = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, dx))  # Spatial frequencies (momentum)
    omega = jnp.fft.fftshift(jnp.fft.fftfreq(Nt, dt))  # Temporal frequencies (energy)

    # Plotting (optional)
    plt.figure(figsize=(6,10))
    plt.imshow(spectrum.T, aspect='auto', extent=[omega[0], omega[-1], k[0], k[-1]])
    plt.ylabel("Energy [eV]")
    plt.xlabel("Momentum [μm⁻¹]")
    plt.colorbar()
    plt.show()

    plt.plot()
    """

    return 0


if __name__ == "__main__":
    main()