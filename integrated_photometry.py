# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:35:03 2025

@author: rache
"""

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.units import Quantity

def integrated_photometry(wave_nm: np.ndarray,
                           flux_planet: Quantity,
                           flux_star: Quantity,
                           wave_filter: np.ndarray,
                           trans_filter: np.ndarray
                          ) -> tuple[float, float, float]:
    """
    Calcule la photométrie intégrée d’un spectre planétaire et stellaire dans un filtre donné,
    ainsi que le contraste bande-passante (planet/star).

    Paramètres
    ----------
    wave_nm : array
        Longueurs d’onde en nm du spectre (même que flux_*).
    flux_planet : Quantity
        Flux spectral réfléchi de la planète [photon / s / m² / nm].
    flux_star : Quantity
        Flux spectral de l’étoile [photon / s / m² / nm].
    wave_filter : array
        Longueurs d’onde en nm du profil de transmission du filtre.
    trans_filter : array
        Transmission associée (entre 0 et 1), même taille que wave_filter.

    Retours
    -------
    F_planet : float
        Flux intégré planète dans la bande [ph/s/m²]
    F_star : float
        Flux intégré étoile dans la bande [ph/s/m²]
    C_band : float
        Contraste intégré planète/étoile dans la bande
    """

    # Vérifications d’unité
    for flux in [flux_planet, flux_star]:
        if not isinstance(flux, Quantity):
            raise TypeError("Les flux doivent être des objets astropy.Quantity avec unités [ph/s/m²/nm]")
        if flux.unit != u.photon / u.s / u.m**2 / u.nm:
            raise ValueError("Les flux doivent être en [ph/s/m²/nm]")

    # Interpolation du filtre sur la grille spectrale
    T_interp = interp1d(wave_filter, trans_filter, bounds_error=False, fill_value=0.0)
    T_lambda = T_interp(wave_nm)

    # Application du filtre aux deux spectres
    Fp_values = flux_planet.value * T_lambda
    Fs_values = flux_star.value   * T_lambda

    # Intégration numérique dans la bande
    Fp_band = np.trapz(Fp_values, x=wave_nm)  # [ph/s/m²]
    Fs_band = np.trapz(Fs_values, x=wave_nm)  # [ph/s/m²]

    # Contraste bande-passante
    C_band = Fp_band / Fs_band if Fs_band > 0 else np.nan

    return Fp_band, Fs_band, C_band
