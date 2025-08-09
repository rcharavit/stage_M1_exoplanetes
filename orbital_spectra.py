# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:25:19 2025

@author: rache
"""

# orbital_spectra.py
"""
Module pour la simulation spectro-orbitales :
- lecture du spectre PHOENIX
- résolution de l’équation de Kepler
- calcul de spectres réfléchi et effet Doppler
- tracés statiques et interactifs

Fonctions principales :
    load_phoenix_spectrum(...)
    solve_ecc_anomaly(...)
    compute_orbit(...)
    compute_state(...)
    planet_reflected_spectrum(...)
    plot_spectra(...)
    main()  # démonstration CLI

Usage :
    from orbital_spectra import load_phoenix_spectrum, compute_orbit, plot_spectra
    wave, flux = load_phoenix_spectrum(...)
    r_m, phi, v = compute_orbit(...)
    flux_p = planet_reflected_spectrum(flux, r_m, phi, Ag, Rp)
    plot_spectra(wave, flux, flux_p, beta=v/c)

Exécuter en ligne de commande :
    python orbital_spectra.py
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.time import Time
from astropy import constants as const
from albedo_tools import load_albedo_spectrum, resample_and_degrade


# Pour les constantes globales
G       = const.G.value
au      = const.au.value


# --- Orbital mechanics ---

def anomaly_Kepler(M: np.ndarray,
                   e: float,
                   tol: float = 1e-8,
                   maxiter: int = 50
                   ) -> np.ndarray:
    """
Résout l’équation de Kepler M = E - e·sin(E) pour un tableau d’anomalies moyennes M.

Utilise la méthode de Newton–Raphson pour obtenir l’anomalie excentrique E 
associée à chaque valeur de M, avec une précision réglable (tolérance).

Paramètres
----------
M : ndarray
    Anomalie moyenne (en radians)
e : float
    Excentricité orbitale (0 ≤ e < 1)
tol : float
    Tolérance numérique pour l’arrêt de l’itération (par défaut : 1e-8)
maxiter : int
    Nombre maximal d’itérations (par défaut : 50)

Retour
------
E : ndarray
    Anomalie excentrique associée à M

    """
    E = M.copy()
    for _ in range(maxiter):
        f_val  = E - e * np.sin(E) - M
        fp     = 1 - e * np.cos(E)
        delta  = -f_val / fp
        E     += delta
        if np.all(np.abs(delta) < tol):
            break
    return E


def true_anomaly(t_obs: float,
                   P_days: float,
                   e: float,
                   tol: float = 1e-8,
                   maxiter: int = 50) -> float:
    """
Calcule l’anomalie vraie f_obs au temps d’observation t_obs pour une orbite de période P_days.

Résout numériquement l’équation de Kepler pour obtenir l’anomalie excentrique E,
puis la convertit en anomalie vraie f par la formule trigonométrique classique.

Paramètres
----------
t_obs : float
    Temps d’observation depuis le périastre (en jours)
P_days : float
    Période orbitale (en jours)
e : float
    Excentricité orbitale (0 ≤ e < 1)
tol : float
    Tolérance numérique pour la résolution de Kepler (défaut : 1e-8)
maxiter : int
    Nombre maximal d’itérations pour la méthode de Newton–Raphson (défaut : 50)

Retour
------
f_obs : float
    Anomalie vraie au temps t_obs (en radians)
"""

    M_obs = 2 * np.pi * t_obs / P_days
    E = M_obs
    for _ in range(maxiter):
        f_val = E - e * np.sin(E) - M_obs
        fp = 1 - e * np.cos(E)
        delta = -f_val / fp
        E += delta
        if abs(delta) < tol:
            break
    f_obs = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                           np.sqrt(1 - e) * np.cos(E / 2))
    return f_obs


# Rotation élémentaire (ω → i → Ω)
def rotate(xo, yo, zo,
           omega:float,   # en radians
           inc:float,     # en radians
           Omega: float):  # en radians
    """
Applique une rotation orbitale 3D à un vecteur (xo, yo, zo) selon les trois angles de l’orbite : ω, i, Ω.

Effectue successivement :
  - une rotation autour de l’axe z par l’argument du périastre ω (omega),
  - une rotation autour de l’axe x par l’inclinaison i (inc),
  - une rotation autour de l’axe z par la longitude du nœud ascendant Ω (Omega).

Paramètres
----------
xo, yo, zo : array-like
    Coordonnées initiales du vecteur dans le repère orbital (plan de l’orbite)
omega : float
    Argument du périastre (en radians)
inc : float
    Inclinaison de l’orbite (en radians)
Omega : float
    Longitude du nœud ascendant (en radians)

Retour
------
x3, y3, z3 : array-like
    Coordonnées du vecteur après rotation dans le repère inertiel
"""

    cosw, sinw = np.cos(omega), np.sin(omega)
    cosi, sini = np.cos(inc),   np.sin(inc)
    cosO, sinO = np.cos(Omega), np.sin(Omega)
    # par omega
    x1 =  cosw*xo - sinw*yo
    y1 =  sinw*xo + cosw*yo
    z1 =  zo
    # par inclinaison
    x2 =  x1
    y2 =  cosi*y1 - sini*z1
    z2 =  sini*y1 + cosi*z1
    # par noeud ascendant
    x3 =  cosO*x2 - sinO*y2
    y3 =  sinO*x2 + cosO*y2
    z3 =  z2
    return x3, y3, z3

# Orbite au foyer : contruction d'un nuage de point 3D
def compute_orbit(a: float, e: float, inc: float, omega: float, Omega: float, nb: int = 400): 
    """
Construit l’orbite elliptique en 3D dans le référentiel inertiel à partir des paramètres orbitaux.

Génère un nuage de nb points décrivant l’ellipse orbitale complète (de 0 à 2π)
dans le plan orbital, puis applique une rotation spatiale selon (ω, i, Ω)
pour obtenir les coordonnées (X, Y, Z) dans le référentiel inertiel.

Paramètres
----------
a : float
    Demi-grand axe de l’orbite (en unités arbitraires, typiquement UA)
e : float
    Excentricité orbitale
inc : float
    Inclinaison de l’orbite (en radians)
omega : float
    Argument du périastre (en radians)
Omega : float
    Longitude du nœud ascendant (en radians)
nb : int
    Nombre de points de l’orbite à générer (par défaut : 400)

Retour
------
X, Y, Z : ndarray
    Coordonnées 3D de l’orbite dans le référentiel inertiel, en unités de a
"""

    theta = np.linspace(0, 2*np.pi, nb)
    r     = a*(1 - e**2)/(1 + e*np.cos(theta))
    x_orb = r*np.cos(theta)
    y_orb = r*np.sin(theta)
    z_orb = np.zeros_like(theta)
    return rotate(x_orb, y_orb, z_orb, omega, inc, Omega)


def compute_points(a: float, e: float, inc: float, omega: float, Omega: float, f0_deg: float):
    
    """
Calcule quatre points clés de l’orbite dans le référentiel inertiel :
- la position de la planète à une anomalie vraie donnée f₀,
- le périastre,
- le nœud ascendant,
- le nœud descendant.

Chaque point est exprimé en coordonnées cartésiennes (x, y, z), en unités du demi-grand axe a.
La rotation orbitale (ω, i, Ω) est appliquée à tous les points.

Paramètres
----------
a : float
    Demi-grand axe de l’orbite
e : float
    Excentricité orbitale
inc : float
    Inclinaison (radians)
omega : float
    Argument du périastre (radians)
Omega : float
    Longitude du nœud ascendant (radians)
f0_deg : float
    Anomalie vraie de la planète à l’instant considéré (en degrés)

Retour
------
planet, peri, asc, desc : tuple(x, y, z)
    Coordonnées 3D des quatre points clés dans le référentiel inertiel
"""

    f0 = np.radians(f0_deg)
    # planet position
    r0     = a*(1 - e**2)/(1 + e*np.cos(f0))
    planet = rotate(r0*np.cos(f0), r0*np.sin(f0), 0.0, omega, inc, Omega)
    # periapsis
    peri   = rotate(a*(1 - e), 0.0, 0.0, omega, inc, Omega)
    # nodes
    f_asc  = (-omega) % (2*np.pi)
    f_desc = (np.pi - omega) % (2*np.pi)
    r_asc  = a*(1 - e**2)/(1 + e*np.cos(f_asc))
    r_desc = a*(1 - e**2)/(1 + e*np.cos(f_desc))
    asc    = rotate(r_asc*np.cos(f_asc), r_asc*np.sin(f_asc), 0.0, omega, inc, Omega)
    desc   = rotate(r_desc*np.cos(f_desc),r_desc*np.sin(f_desc),0.0, omega, inc, Omega)
    return planet, peri, asc, desc


def orbit_xyz(a_AU: float,
                          e: float,
                          f_t: np.ndarray,
                          inc: float,
                          omega: float,
                          Omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la position (X, Y, Z) en 3D pour chaque anomalie vraie f_t.

    Parameters
    ----------
    a_AU : float
        Demi-grand axe (UA)
    e : float
        Excentricité
    f_t : ndarray
        Anomalie vraie (en radians) pour chaque t
    inc : float
        Inclinaison en radians
    omega : float
        Argument du périastre en radians
    Omega : float
        Longitude du nœud ascendant en radians

    Returns
    -------
    X, Y, Z : ndarrays
        Coordonnées 3D de la planète sur toute l’orbite (en UA)
    """
    r = a_AU * (1 - e**2) / (1 + e * np.cos(f_t))  # distance à chaque t (en UA)
    x_orb = r * np.cos(f_t)
    y_orb = r * np.sin(f_t)
    z_orb = np.zeros_like(f_t)

    X, Y, Z = rotate(x_orb, y_orb, z_orb, omega, inc, Omega)
    return X, Y, Z



def phase_function(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Calcule la fonction de phase φ(t) à partir de la géométrie réelle (X, Y, Z)
    dans le repère inertiel, en supposant que l’observateur regarde dans la direction +Z.

    Parameters
    ----------
    X, Y, Z : ndarrays
        Coordonnées 3D (en UA ou m, unité cohérente) de la planète sur l’orbite complète.
        
    Returns
    -------
    phi : ndarray
        Fonction de phase φ(t) (entre 0 et 1)
        
     r_m : ndarray
         Distance étoile-planète (en m)
    """
    # vecteur position planète → étoile
    r_AU = np.sqrt(X**2 + Y**2 + Z**2)
    r_m = r_AU * au  # float, en mètres

    
    # cos(alpha) = direction observateur • direction planète
    # Observateur est supposé regarder vers le centre de l'orbite (le barycentre), le long de +Z
    # Donc vecteur d'observation : u_obs = [0, 0, 1]
    cos_alpha = Z / r_AU
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)        # sécurité numérique
    alpha = np.arccos(cos_alpha)

    # Loi de phase de Lambert sphérique
    phi = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

    return phi, r_m


#Fonction pour calculer les vecteurs vitesse
def velocity_vectors(times_days: np.ndarray, a_AU: float, e: float,
                     inc: float, omega: float, Omega: float,
                     P_days: float, Mstar: float, Mplanet: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
Calcule les composantes du vecteur vitesse (vx, vy, vz) dans le repère inertiel à chaque instant donné.

La fonction :
- Résout l'équation de Kepler pour obtenir l'anomalie excentrique E(t),
- En déduit l’anomalie vraie f(t),
- Calcule les composantes radiale et tangentielle de la vitesse dans le plan orbital,
- Applique les rotations orbitales (ω, i, Ω) pour obtenir les vitesses dans le repère inertiel.

Paramètres
----------
times_days : ndarray
    Grille temporelle (en jours)
a_AU : float
    Demi-grand axe de l’orbite (en unités astronomiques)
e : float
    Excentricité orbitale
inc : float
    Inclinaison (radians)
omega : float
    Argument du périastre (radians)
Omega : float
    Longitude du nœud ascendant (radians)
P_days : float
    Période orbitale (en jours)
Mstar : float
    Masse de l’étoile (en masses solaires)

Retour
------
vx, vy, vz : ndarray
    Composantes du vecteur vitesse (en m/s) dans le repère inertiel
"""

    a_m = a_AU * const.au.to(u.m).value  # demi-grand axe en mètres
    GAUSS = 0.017202098950
    n = GAUSS * np.sqrt(Mstar) / a_AU**1.5
    M_t   = n*times_days
    E_t   = anomaly_Kepler(M_t, e)
    f_t = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E_t / 2), np.sqrt(1 - e) * np.cos(E_t / 2))
    M_sun = const.M_sun.value
    mu = G * (Mstar * M_sun + Mplanet)
    h = np.sqrt(mu * a_m * (1 - e**2))
    v_r   = mu/h * e * np.sin(f_t)
    v_t   = mu/h * (1 + e*np.cos(f_t))
    vx_o  = v_r*np.cos(f_t) - v_t*np.sin(f_t)
    vy_o  = v_r*np.sin(f_t) + v_t*np.cos(f_t)
    vz_o  = np.zeros_like(f_t)
    vx, vy, vz = rotate(vx_o, vy_o, vz_o, omega, inc, Omega)
    return vx, vy, vz


def orbit_parameters(e: float, a_AU: float, Mstar: float, Mplanet: float,
                     inc: float, omega: float, Omega: float,
                     nb: int = 1000) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calcule les instants, l’anomalie vraie, la période orbitale et la vitesse orbitale.

    Parameters
    ----------
    e : float
        Excentricité
    a_AU : float
        Demi-grand axe (UA)
    Mstar : float
        Masse de l’étoile (en kg)
    Mplanet : float
        Masse de la planète (en kg)
    inc : float
        Inclinaison (radians)
    omega : float
        Argument du périastre (radians)
    Omega : float
        Longitude du nœud ascendant (radians)
    nb : int
        Nombre de points temporels

    Returns
    -------
    times_days : ndarray
        Grille temporelle en jours
    f_t : ndarray
        Anomalie vraie (radians)
    P_days : float
        Période orbitale en jours
    v_abs : ndarray
        Vitesse absolue en m/s
    """

    a_m = a_AU * const.au.value  # conversion en m
    Mstar_kg = Mstar * const.M_sun.value
    Mplanet_kg = Mplanet * const.M_earth.value
    mu = G * (Mstar_kg + Mplanet_kg)

    # Période orbitale
    P_sec = 2 * np.pi * np.sqrt(a_m**3 / mu)
    P_days = P_sec / 86400.0
    times_days = np.linspace(0, P_days, nb)

    # Mouvement moyen et anomalies
    n = 2 * np.pi / P_days
    M_t = n * times_days
    E_t = anomaly_Kepler(M_t, e)

    f_t = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E_t / 2),
                         np.sqrt(1 - e) * np.cos(E_t / 2))

    # Vitesse inertielle
    vx, vy, vz = velocity_vectors(times_days, a_AU, e,
                                  inc, omega, Omega, P_days,
                                  Mstar, Mplanet)
    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)

    return times_days, f_t, P_days, v_abs


# --- Input/Output ---
def load_phoenix_spec(wave_path: str,
                          spec_path: str,
                          lam_min: float = 3800.0,
                          lam_max: float = 7500.0
                          ) -> tuple[np.ndarray, u.Quantity]:
    """
Charge un spectre stellaire PHOENIX à partir de fichiers FITS, et le convertit en flux photonique utilisable.

La fonction :
- Charge la longueur d’onde et le flux spectral (en erg/s/cm²/cm) depuis les fichiers FITS fournis,
- Applique un masque de sélection sur l’intervalle spectral visible [lam_min, lam_max],
- Convertit le flux en unités SI puis en photons/s/m²/nm grâce aux équivalences spectrales d’Astropy,
- Retourne la grille spectrale en nanomètres et le flux photonique spectral associé.

Paramètres
----------
wave_path : str
    Chemin vers le fichier FITS contenant la grille de longueurs d’onde (en Å)
spec_path : str
    Chemin vers le fichier FITS contenant le spectre PHOENIX (en erg/s/cm²/cm)
lam_min : float
    Longueur d’onde minimale à conserver (en Å) — défaut : 3800 Å
lam_max : float
    Longueur d’onde maximale à conserver (en Å) — défaut : 7500 Å

Retour
------
wave_nm : ndarray
    Grille de longueurs d’onde en nanomètres
flux_star : Quantity
    Flux stellaire en photons/s/m²/nm (unité : ph/s/m²/nm)
"""
    data_wave, _ = fits.getdata(wave_path, header=True)
    data_spec, _ = fits.getdata(spec_path, header=True)
    mask = (data_wave >= lam_min) & (data_wave <= lam_max)
    wave_vis = data_wave[mask] * u.AA
    spec_vis = data_spec[mask] * u.erg / u.s / u.cm**2 / u.cm
    flux_AA = spec_vis.to(u.erg/u.s/u.cm**2/u.AA,
                           equivalencies=[(u.cm, 1e-8*u.AA)])
    flux_SI = flux_AA.to(u.W/u.m**2/u.m,
                         equivalencies=u.spectral_density(wave_vis))
    flux_star = flux_SI.to(u.photon/u.s/u.m**2/u.nm,
                           equivalencies=u.spectral_density(wave_vis))
    wave_nm = wave_vis.to(u.nm).value
    return wave_nm, flux_star


# --- Spectre réfléchi de la planète ---
def planet_refl_spec(wave_nm, r_m, phi_t, flux_star, material, Rp, t_index):
    """
Calcule le flux planétaire réfléchi en fonction du temps et de la longueur d’onde, ainsi que le contraste intégré.

La fonction :
- Interpole le spectre d’albédo du matériau donné sur la grille spectrale du modèle (sans dégradation),
- Calcule le flux réfléchi total de la planète à chaque instant en tenant compte de :
    • l’albédo apparent (albédo × fonction de phase),
    • la distance instantanée étoile–planète,
    • le flux stellaire incident,
- Intègre ce flux sur le spectre pour obtenir une courbe temporelle du contraste C(t),
- Calcule également le spectre réfléchi instantané au temps d’observation donné (t_index).

Paramètres
----------
wave_nm : ndarray
    Longueurs d’onde en nanomètres (grille spectrale du modèle)
r_m : ndarray
    Distance étoile–planète à chaque instant (en mètres ou sans unité)
phi_t : ndarray
    Fonction de phase à chaque instant (valeurs entre 0 et 1)
flux_star : Quantity
    Spectre stellaire en [ph/s/m²/nm]
material : str
    Nom du matériau atmosphérique utilisé pour l’albédo ("KCl", "ZnS", etc.)
Rp : float or Quantity
    Rayon de la planète (en mètres)
t_index : int
    Indice temporel correspondant à l’instant d’observation

Retour
------
flux_p_time : ndarray
    Flux planétaire réfléchi intégré dans le visible à chaque instant (en ph/s/m²)
C : ndarray
    Contraste temporel C(t) = flux_planète / flux_étoile (sans unité)
flux_p_spec : ndarray
    Spectre planétaire réfléchi instantané à t_index (en ph/s/m²/nm)
"""
    fs = flux_star.value  # [ph/s/m²/nm]
    wave = wave_nm
    nt = phi_t.size
    Rp_m = Rp.value if hasattr(Rp, "value") else Rp

    # 2) Albédo spectral (interpolation directe sans dégrader)
    wvl_src, Ag_src = load_albedo_spectrum(material)
    Ag_spec = np.interp(wave * 1e-3, wvl_src, Ag_src)  # nm → µm

    # 3) Flux réfléchi pour chaque instant
    flux_p_time = np.empty(nt)

    for i in range(nt):
        Ag_app = Ag_spec * phi_t[i]
        r_i = r_m[i].value if hasattr(r_m[i], 'value') else r_m[i]
        geom_factor = (Rp_m / r_i)**2
        flux_p = Ag_app * fs * geom_factor
        flux_p_time[i] = np.trapz(flux_p, x=wave)

    # 4) Flux stellaire total
    flux_star_total = np.trapz(fs, x=wave)

    # 5) Contraste temporel
    C = flux_p_time / flux_star_total

    # 6) Spectre planétaire instantané (même résolution que PHOENIX)
    Ag_app_index = Ag_spec * phi_t[t_index]
    r_i = r_m[t_index].value if hasattr(r_m[t_index], 'value') else r_m[t_index]
    geom_factor = (Rp_m / r_i)**2
    flux_p_spec = Ag_app_index * fs * geom_factor

    return flux_p_time, C, flux_p_spec







