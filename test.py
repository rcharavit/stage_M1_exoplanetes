# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:36:57 2025

@author: rache
"""
"""
Script principal de simulation exoplanétaire — modélisation orbitale et spectroscopie

Ce script exécute la simulation complète d’un système exoplanétaire vu en lumière réfléchie,
à partir de modèles orbitaux képlériens, de spectres stellaires (PHOENIX) et d’albédo planétaire.

Fonctionnalités principales :
-----------------------------
- Charge les spectres stellaires haute résolution (modèles PHOENIX).
- Charge et interpole les spectres d’albédo géométrique d’un matériau atmosphérique.
- Calcule l’orbite complète de la planète à partir des 6 paramètres orbitaux (a, e, i, ω, Ω, M⋆).
- Calcule à chaque instant :
    • l’anomalie vraie f(t)
    • la distance étoile-planète r(t)
    • la fonction de phase φ(t)
    • les vecteurs vitesse inertiels (vx, vy, vz)
    • la lumière réfléchie par la planète
    • le contraste spectroscopique C(t) = flux_planète / flux_étoile
- Calcule et affiche :
    • le spectre stellaire et le spectre planétaire
    • les courbes temporelles du contraste, de la vitesse radiale, de la distance r(t), etc.
    • l’orbite en 2D et 3D avec l’emplacement instantané t_obs
    • le décalage Doppler du spectre planétaire

Paramètres modifiables :
------------------------
- Données orbitales (a, e, i, ω, Ω, M⋆, Rₚ)
- Date d'observation t_obs (en jours depuis le périastre)
- Matériau atmosphérique pour l’albédo : "KCl", "ZnS", "Na2S", "ZnS_KCl", "no_clouds"
- Spectres PHOENIX : fichiers .fits à haute résolution

Ce fichier appelle les modules suivants :
-----------------------------------------
- orbital_spectra.py : calculs orbitaux et spectres planétaires
- albedo_tools.py : traitement et interpolation des spectres d’albédo
- plot_tools.py : fonctions d’affichage (spectres, orbites, contrastes, etc.)

Utilisation :
-------------
Lancer simplement le script :
    python main_orbit_model.py

Le script affichera automatiquement tous les graphes liés à la modélisation du système exoplanétaire.
"""


import numpy as np
import pandas as pd
import astropy.units as u
from astropy import constants as cst
from astropy.time import Time

# --- 1) Import des fonctions de vos modules ---
from orbital_spectra import (true_anomaly,
    load_phoenix_spec,
    orbit_parameters,
    orbit_xyz,
    phase_function,
    velocity_vectors,
    planet_refl_spec,
    compute_orbit,
    compute_points,
)
from plot_tools import (
    star_spec,
    planet_spec,
    contrast_spec,
    doppler_shift,
    phi_curve,
    radial_velocity,
    contrast,
    distance_SP,
    absolute_velocity,
    albedo,
    orbit_plane,
    orbit_3d,
    albedo_phase_spec,
)
from albedo_tools import load_albedo_spectrum, resample_and_degrade
from integrated_photometry import integrated_photometry

# --- 2) Paramètres utilisateur ---
# Fichiers PHOENIX
wave_fits = r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
spec_fits = r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\lte06500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

# Orbite et planète
days     = 365/2             # jours depuis le périastre
e        = 0.04839           # excentricité
a_AU     = 5.20289           # demi-grand axe en UA
Mstar    = 1.0           # masse de l'étoile en M_sun
inc_deg  = 90          # inclinaison en degrés
omega_deg= 90.0          # argument du périastre (°)
Omega_deg= 90.0          # nœud ascendant (°)
Rp       = 11.209*cst.R_earth.value   # rayon planétaire (m)
Mplanet = 317.8  #Masse de la planète (nb de fois la masse de la Terre)
Mstar = 1.0   #Masse de l'étoile (nb de fois la masse du soleil)

# Matériau d'albédo
material = "ZnS_KCl"          # choix : "KCl", "ZnS", "Na2S", "ZnS_KCl", "no_clouds"
R = 50                    #résolution spectrale


# --- Routine principale ---
def main():
    
    # 1) Charger les données PHOENIX et albédo
    wave_nm, flux_star = load_phoenix_spec(wave_fits, spec_fits)
    wvl_src, Ag_src = load_albedo_spectrum(material)
    Ag_KCl = resample_and_degrade(wvl_src, Ag_src, wave_nm * 1e-3)


    # 2) Paramètres orbitaux + vitesse réelle
    times_days, f_t, P_days, v_abs = orbit_parameters(
        e=e,
        a_AU=a_AU,
        Mstar=Mstar,
        Mplanet=Mplanet,
        inc=np.radians(inc_deg),
        omega=np.radians(omega_deg),
        Omega=np.radians(Omega_deg),
        nb=1000)

    # 3) Anomalie vraie au temps t_obs (f_obs)
    f_obs = true_anomaly(t_obs=days, P_days=P_days, e=e)

    # 4) Points clés de l’orbite
    planet, peri, asc, desc = compute_points(
        a=a_AU, e=e,
        inc=np.radians(inc_deg),
        omega=np.radians(omega_deg),
        Omega=np.radians(Omega_deg),
        f0_deg=np.degrees(f_obs))

    # 5) Orbite 3D
    X, Y, Z = orbit_xyz(a_AU, e, f_t,
                        inc=np.radians(inc_deg),
                        omega=np.radians(omega_deg),
                        Omega=np.radians(Omega_deg))
    phi, r_m = phase_function(X, Y, Z)

    # 6) Affichage 2D & 3D
    orbit_plane(compute_orbit, compute_points,
            a_AU, e, np.radians(inc_deg),
            np.radians(omega_deg), np.radians(Omega_deg),
            times_days, f_t, days,distance_pc=10)

    orbit_3d(compute_orbit, compute_points, velocity_vectors,
         a_AU, e, np.radians(inc_deg),
         np.radians(omega_deg), np.radians(Omega_deg),
         times_days, f_t, days,
         Mstar=Mstar, Mplanet=Mplanet, distance_pc=10)


    # 7) Index temporel le plus proche de t_obs
    i = np.argmin(np.abs(times_days - days))
    phi0 = phi[i]


    # 8) Vecteurs vitesse dans le repère inertiel

    vx, vy, vz = velocity_vectors(
    times_days=times_days,
    a_AU=a_AU,
    e=e,
    inc=np.radians(inc_deg),
    omega=np.radians(omega_deg),
    Omega=np.radians(Omega_deg),
    P_days=P_days,
    Mstar=Mstar, 
    Mplanet=Mplanet)


    # 9) Flux planétaire réfléchi
    flux_p_time, C, flux_p_spec = planet_refl_spec(
    wave_nm=wave_nm,
    r_m=r_m * u.m,
    phi_t=phi,
    flux_star=flux_star,
    material=material,
    Rp=Rp,
    t_index=i)

    ''' -----------------------------------------------------------------
        
        Bonus : Tester le simulateur du coronographe.
        Calcule la photométrie intégrée d’un spectre dans un filtre donné.'''

    # Lecture du fichier en ignorant les lignes de commentaire
    df = pd.read_csv(r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\Modèle exoplanétaire Rachel CHARAVIT M1\filter transmission curves\transmission_ID-11_3A_v0.csv", comment="#", sep=",")
    df.columns = ["lambda_nm", "T_percent"]
    
    # Conversion en arrays
    wave_filter = df["lambda_nm"].values                    # en nm
    trans_filter = df["T_percent"].values / 100             # en fraction (0–1)

    # Vérification des unités (important !)
    if not hasattr(flux_star, "unit"):
        flux_star = flux_star * u.photon / u.s / u.m**2 / u.nm
    if not hasattr(flux_p_spec, "unit"):
        flux_p_spec = flux_p_spec * u.photon / u.s / u.m**2 / u.nm

    # Appel de la fonction
    Fp, Fs, C_band = integrated_photometry(
        wave_nm,
        flux_p_spec,
        flux_star,
        wave_filter,
        trans_filter
        )

    # Affichage
    print(f"Flux intégré planète : {Fp:.2e} ph/s/m²")
    print(f"Flux intégré étoile  : {Fs:.2e} ph/s/m²")
    print(f"Contraste (Bande 1) : {C_band:.2e}")

    
    '''-----------------------------------------------------------'''

        # Valeurs observées à t_obs (c’est-à-dire days)
    C_obs    = C[i]
    v_rad    = vz       #composante radiale (vu depuis l’observateur)
    v_obs    = v_rad[i]
    Ag_obs   = phi[i]

    beta = float(v_rad[i] / cst.c.value)

    # --- Extraction des valeurs au bon instant ---
    
    # 10) Tracés spectraux
    star_spec(wave_nm, flux_star)
    planet_spec(wave_nm, flux_p_spec)
    contrast_spec(
    wave_nm=wave_nm,
    material=material,
    phi_val=phi[i],
    r_val=r_m[i],
    Rp=Rp,
    R=50,
    t_obs=days)
    doppler_shift(wave_nm, flux_star, flux_p_spec, beta)
    albedo_phase_spec(wave_nm, Ag_KCl, phi0, label=f"{material} (φ={phi0:.2f}, R=50)")


    # Tracés temporels avec point rouge à t_obs
    contrast(times_days=times_days, P_days=P_days, C=C, C_obs=C_obs, t_obs=days)
    phi_curve(times_days, P_days, phi, f_obs, t_obs=days)
    radial_velocity(times_days=times_days, P_days=P_days, v_rad=v_rad, v_obs=v_obs, t_obs=days)
    distance_SP(times_days=times_days,
            P_days=P_days,
            X=X, Y=Y, Z=Z,
            t_obs=days)
    absolute_velocity(
    times_days=times_days,
    P_days=P_days,
    v_abs=v_abs,
    v_obs=v_abs[i],
    t_obs=days)
    albedo(times_days=times_days, P_days=P_days, Ag_app=phi, Ag_max=1.0, Ag_obs=Ag_obs, t_obs=days)


    # 12) (Optionnel) Comparaison spectres d’albédo bruts
    lam0, A0 = load_albedo_spectrum(material)
    A_interp = resample_and_degrade(lam0, A0, wave_nm / 1e3)
    

if __name__ == "__main__":
    main()


