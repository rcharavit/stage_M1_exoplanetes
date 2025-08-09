##Toy Exoplanet Simulator
This repository contains a Python tool developed during my Master’s internship under the supervision of Dr. Arthur Vigan.
It can simulate a planet’s 3D Keplerian orbit, load a high-resolution stellar PHOENIX spectrum, generate the planet’s reflected spectrum, visualize the results, and prepare them for injection into NASA’s Roman Coronagraph simulator.

##Repository Contents
# orbital_spectra.py
Computation module
- Load PHOENIX spectra (I/O)
- Solve Kepler’s equation
- Compute orbital positions, anomalies, and velocities
- Generate the planet’s reflected-light spectrum

# plot_tools.py
Plotting module
- 1D visualization functions (spectra, phase curves, velocities, contrast…)
- 2D and 3D orbital plots with velocity vectors

# integrated_photometry.py
Integrated photometry module
- Compute integrated planet and star fluxes within the passband of a given filter
- Apply the filter’s transmission profile (interpolated onto the spectral grid)
- Perform numerical integration over the selected spectral band
- Return the planet/star contrast integrated over the band

# albedo_tools.py
Geometric albedo spectrum handling module
- Automatically load geometric albedo spectra from .dat files for various atmospheric materials (KCl, ZnS, Na₂S, etc.)
- Convert spectra from wavenumber space (cm⁻¹) to wavelength space (µm) while preserving physical units
- Compute the geometric albedo spectrum from net, reflected, and thermal fluxes
- Interpolate spectra onto a target wavelength grid
- Optionally simulate instrumental spectral degradation at a given resolution (R) via convolution

# test.py
Example execution script (CLI)
- Sets input file paths and parameters
- Imports the above modules
- Runs the full pipeline: computation → plotting
