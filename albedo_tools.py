"""
Created on Sun Jul  6 12:14:18 2025

@author: rachel
"""
"""

Ce module permet de charger, interpoler et, si besoin, dégrader spectroscopiquement 
les spectres d'albédo géométrique de différents matériaux atmosphériques, 
utilisés pour modéliser la lumière réfléchie par une exoplanète.

Fonctionnalités principales :
-----------------------------
- Charger automatiquement les spectres d'albédo à partir de fichiers .dat selon le matériau sélectionné.
- Convertir les spectres depuis le domaine du nombre d’onde (cm⁻¹) vers celui des longueurs d’onde (µm),
  en tenant compte des unités physiques.
- Extraire l’albédo géométrique spectral à partir du flux net, réfléchi et thermique.
- Interpoler les spectres sur une grille cible de longueurs d’onde.
- Simuler une dégradation instrumentale en résolution spectrale (R) à l’aide d’un noyau de convolution 1D.

Matériaux disponibles :
-----------------------
- "KCl", "ZnS", "Na2S", "ZnS_KCl", "no_clouds"

Utilisation typique :
---------------------
- Utiliser `load_albedo_spectrum(material)` pour obtenir le spectre brut d’albédo (λ, A_g)
- Utiliser `resample_and_degrade(...)` pour interpoler ce spectre sur la grille du modèle,
  avec ou sans dégradation spectrale selon la résolution instrumentale R souhaitée.
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve


# Dictionnaire des chemins de fichiers, clé = matériau
_MATERIAL_FILES = {
    "KCl":       r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\reflectivity_spectra\reflectivity_spectra\GJ1214b_spec\KCl\.spectra_KCl.dat",
    "ZnS":       r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\reflectivity_spectra\reflectivity_spectra\GJ1214b_spec\ZnS\spectra_ZnS.dat",
    "ZnS_KCl":   r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\reflectivity_spectra\reflectivity_spectra\GJ1214b_spec\ZnS_KCl\spectra_ZnS_KCl.dat",
    "no_clouds": r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\reflectivity_spectra\reflectivity_spectra\GJ1214b_spec\no_clouds\spectra_no_clouds.dat",
    "Na2S":      r"C:\M1 SOAC\S2\Stage M1\Travaux\Programmation\reflectivity_spectra\reflectivity_spectra\GJ1214b_spec\Na2S\spectra_Na2S.dat",
    }

def _load_raw(material: str) -> np.ndarray:
    """
Charge les données brutes du spectre d’un matériau atmosphérique depuis un fichier .dat.

La fonction :
- Vérifie que le nom du matériau est valide et que le fichier existe,
- Charge uniquement les colonnes utiles du fichier : 
    • colonne 0 : nombre d’onde σ (en cm⁻¹),
    • colonne 1 : flux net (réflexion + thermique),
    • colonne 4 : flux planétaire total,
    • colonne 5 : composante thermique du flux planétaire.

Paramètres
----------
material : str
    Nom du matériau (doit correspondre à une clé de _MATERIAL_FILES)

Retour
------
data : ndarray de shape (N, 4)
    Tableau contenant σ, flux_net, flux_planet, flux_therm pour chaque longueur d’onde
"""

    #Vérification de la clé matériau
    if material not in _MATERIAL_FILES:
    # Si l’utilisateur demande un matériau qui n’existe pas dans notre dictionnaire
        raise KeyError(f"Matériau inconnu : {material}")
    path = _MATERIAL_FILES[material] #Récupération du chemin de fichier associé
    if not os.path.isfile(path): #Vérification que le fichier existe physiquement sur le disque
        raise FileNotFoundError(f"Spectre albédo introuvable : {path}")
    # colonnes : 0=wavenumber [cm⁻¹], 1=flux_net, 4=flux_planet, 5=flux_planet_thermal
    return np.loadtxt(path, usecols=(0,1,4,5))

def wnb_to_wvl(sigma_cm1: np.ndarray) -> np.ndarray:
    """Convertit des nombres d’onde [cm⁻¹] en longueurs d’onde [µm]."""
    return 1e4 / sigma_cm1

def fsigma_to_flambda(sigma_cm1: np.ndarray, fsigma: np.ndarray) -> np.ndarray:
    """
    Convertit un spectre fsigma [W/m²/cm⁻¹] en flambda [W/m²/µm],
    via flambda = fsigma * (cm⁻¹→µm⁻¹)/(λ²).
    """
    lam = wnb_to_wvl(sigma_cm1)
    return fsigma * 1e4 / lam**2

def load_albedo_spectrum(material: str) -> tuple[np.ndarray,np.ndarray]:
    """
Charge et calcule le spectre d’albédo géométrique d’un matériau atmosphérique.

La fonction :
- Charge les flux spectraux bruts (net, total planétaire, thermique) en fonction du nombre d’onde σ [cm⁻¹],
- Convertit chaque spectre de f(σ) en f(λ) avec conservation de l’énergie spectrale,
- Calcule l’albédo géométrique spectral selon la formule :
      A_g(λ) = (F_plan - F_therm) / (F_plan - F_net)

Paramètres
----------
material : str
    Nom du matériau à charger (clé de _MATERIAL_FILES)

Retour
------
wvl_um : ndarray
    Longueurs d’onde en microns (µm)
A_g0 : ndarray
    Albédo géométrique spectral non interpolé, en fonction de λ
"""

    sigma, f_net, f_planet, f_therm = _load_raw(material).T
    fl_net   = fsigma_to_flambda(sigma, f_net)
    fl_plan  = fsigma_to_flambda(sigma, f_planet)
    fl_therm = fsigma_to_flambda(sigma, f_therm)
    numerateur = fl_plan - fl_therm
    denominateur = fl_plan - fl_net
    with np.errstate(divide='ignore', invalid='ignore'):
        A0 = np.where(denominateur != 0, numerateur / denominateur, 0.0)
    A0 = np.nan_to_num(A0, nan=0.0, posinf=0.0, neginf=0.0)

    return wnb_to_wvl(sigma), A0


def degrade_resolution(wvl_um: np.ndarray, spectrum: np.ndarray, R: int) -> np.ndarray:
    """
Simule la dégradation spectrale instrumentale d’un spectre en le convoluant avec un noyau de type Boxcar.

Cette fonction modifie la résolution spectrale d’un spectre en λ, en appliquant une convolution 
avec un noyau de largeur équivalente à λ/R, pour approximer une résolution spectroscopique donnée R.

Si la largeur du noyau est trop petite (moins de 3 pixels), la fonction renvoie le spectre original.

Paramètres
----------
wvl_um : ndarray
    Grille spectrale uniforme en longueur d’onde (en microns)
spectrum : ndarray
    Valeurs spectrales à lisser (albédo, flux, etc.)
R : int
    Résolution spectrale souhaitée (R = λ / Δλ)

Retour
------
spectrum_convolved : ndarray
    Spectre lissé par convolution (même taille que `spectrum`)
"""

    # largeur en pixels : δλ = λ/R ; sur grille uniforme en λ on approxime par constante
    dlam = np.mean(np.diff(wvl_um))
    kernel_width = int(np.round((dlam * R) / dlam))
    if kernel_width < 3:
        return spectrum
    kernel = Box1DKernel(kernel_width)
    # on utilise la convolution directe qui supporte boundary='extend'
    return convolve(spectrum, kernel, boundary='extend')

def resample_and_degrade(
    wvl_src: np.ndarray,
    A_src: np.ndarray,
    wvl_target: np.ndarray,
    R: int = None
) -> np.ndarray:
    """
Interpole un spectre d’albédo sur une nouvelle grille, avec option de dégradation instrumentale.

Cette fonction :
- Interpole le spectre (wvl_src, A_src) sur une grille cible `wvl_target`,
- Détecte automatiquement si la grille cible est en nm ou µm,
- Applique en option une dégradation spectrale (convolution Box1D) pour simuler un instrument
  de résolution R (si R est spécifié).

Paramètres
----------
wvl_src : ndarray
    Grille spectrale d’origine (en microns)
A_src : ndarray
    Albédo géométrique associé à wvl_src
wvl_target : ndarray
    Grille spectrale cible (en microns ou en nanomètres, détecté automatiquement)
R : int, optionnel
    Résolution spectrale simulée (R = λ / Δλ). Si None, aucune dégradation n’est appliquée.

Retour
------
A_interp : ndarray
    Albédo interpolé sur la grille cible, avec dégradation éventuelle
"""

    # si wvl_target en nm (>50), on convertit en µm
    if np.max(wvl_target) > 50:
        wvl_tgt_um = wvl_target * 1e-3
    else:
        wvl_tgt_um = wvl_target
    interp = interp1d(wvl_src, A_src, bounds_error=False, fill_value=0.0)
    A_interp = interp(wvl_tgt_um)
    if R is not None:
        A_interp = degrade_resolution(wvl_tgt_um, A_interp, R)
    return A_interp
