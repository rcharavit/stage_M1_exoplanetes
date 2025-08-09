# Toy Exoplanet Simulator

This repository contains a Python tool developed during my Master’s internship under the supervision of Dr. Arthur Vigan. It can simulate a planet’s 3D Keplerian orbit, load a high-resolution stellar PHOENIX spectrum, generate the planet’s reflected spectrum, visualize the results, and prepare them for injection into NASA’s Roman Coronagraph simulator.

---

## Repository Contents

- **orbital_spectra.py**  
  *Computation module*  
  - Load PHOENIX spectra (I/O)  
  - Solve Kepler’s equation  
  - Compute orbital positions, anomalies, and velocities  
  - Generate the planet’s reflected-light spectrum  

- **plot_tools.py**  
  *Plotting module*  
  - 1D visualization functions (spectra, phase curves, velocities, contrast…)  
  - 2D and 3D orbital plots with velocity vectors  

- **test.py**  
  Example execution script (CLI)  
  - Sets input file paths and parameters  
  - Imports the two modules  
  - Runs the full pipeline: computation → plotting  

---

## Prerequisites

```bash
pip install numpy matplotlib astropy