# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:32:50 2025

@author: rache
"""

"""
Module de tracés pour résultats orbitales et spectraux.
Sépare la partie plotting des calculs.
Fonctions :
    plot_phase_curve(times, P_days, phi)
    plot_radial_velocity(times, P_days, v_rad)
    plot_contrast(times, P_days, contrast)
    plot_distance(times, P_days, r)
    plot_absolute_velocity(times, P_days, v_abs)
    plot_albedo(times, P_days, Ag_app, Ag)
    plot_orbit_3d(compute_orbit, compute_points, a, e, inc, omega, Omega, f0_deg)
    plot_orbit_plane(compute_orbit, compute_points, a, e, inc, omega, Omega, f0_deg, view)
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from albedo_tools import resample_and_degrade

# Visualise le spectre et illustre le décalage doppler

def star_spec(wave_nm, flux_star):
    plt.figure(num=1, figsize=(8,5))
    plt.clf()
    plt.plot(wave_nm, flux_star.value, color='tab:blue')
    plt.xlabel('λ (nm)')
    plt.ylabel('Flux étoile (ph/s/m²/nm)')
    plt.title('Spectre étoile')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def planet_spec(wave_nm, flux_planet):
    plt.figure(num=2, figsize=(8,5))
    plt.clf()
    plt.plot(wave_nm, flux_planet, color='tab:orange')
    plt.xlabel('λ (nm)')
    plt.ylabel('Flux planétaire réfléchi (ph/s/m²/nm)')
    plt.title('Spectre planétaire réfléchi')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def contrast_spec(
    wave_nm,
    material,
    phi_val,
    r_val,
    Rp,
    R=50,
    t_obs=None
):
    """
    Affiche le contraste spectral C(λ) = A_g(λ) · Φ(α) · (Rp / r(t))²
    à la résolution spectrale R (par exemple, R=50 comme Roman).

    Paramètres
    ----------
    wave_nm : ndarray
        Longueurs d’onde (en nm).
    material : str
        Nom du matériau ("KCl", "ZnS", etc.).
    phi_val : float
        Valeur de la fonction de phase au temps t_obs (entre 0 et 1).
    r_val : float
        Distance étoile–planète au temps t_obs (en m).
    Rp : float
        Rayon de la planète (en m).
    R : float
        Résolution spectrale souhaitée (défaut = 50).
    t_obs : float ou None
        Temps d'observation pour le titre (optionnel).
    """
    
    from albedo_tools import load_albedo_spectrum

    # --- Charger et interpoler l’albédo spectral du matériau ---
    wvl_src, A_src = load_albedo_spectrum(material)
    Ag_interp = resample_and_degrade(wvl_src, A_src, wave_nm, R=None)  # pas de dégradation

    # --- Calcul analytique du contraste spectral ---
    C_lambda = Ag_interp * phi_val * (Rp / r_val)**2

    # --- Dégradation à la résolution souhaitée ---
    C_lambda_R = resample_and_degrade(wave_nm * 1e-3, C_lambda, wave_nm, R=R)

    # --- Tracé ---
    plt.figure(num=3, figsize=(7, 4))
    plt.clf()
    plt.plot(wave_nm, C_lambda_R, label=f"{material}, R = {R}")
    plt.xlabel("Longueur d’onde (nm)")
    plt.ylabel("Contraste spectral C(λ)")
    if t_obs is not None:
        plt.title(f"Contraste spectral C(λ) – {material} à t = {t_obs:.1f} j")
    else:
        plt.title(f"Contraste spectral C(λ) – {material}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def doppler_shift(wave_nm, flux_star, flux_planet, beta):
    plt.figure(num=4, figsize=(8,5))
    plt.clf()
    plt.plot(wave_nm,          flux_star.value/flux_star.value.max(),
             label='Star (norm.)')
    plt.plot(wave_nm*(1+beta), flux_planet/flux_planet.max(),
             label='Planet (Doppler)', alpha=0.8)
    plt.xlabel('λ (nm)')
    plt.title(f'Effet Doppler β={beta:.2e}')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def phi_curve(times_days: np.ndarray,
              P_days: float,
              phi: np.ndarray,
              f_obs: float,
              t_obs: float):
    
    frac = times_days / P_days
    frac_obs = t_obs / P_days
    phi_obs = (np.sin(f_obs) + (np.pi - f_obs) * np.cos(f_obs)) / np.pi

    plt.figure(num=5, figsize=(6, 4))
    plt.clf()
    plt.plot(frac, phi, '-', label='Φ(t)')
    plt.plot(frac_obs, phi_obs, 'ro', label=f't_obs = {t_obs:.1f} j')
    plt.xlim(0, 1)
    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Fonction de phase φ(t)")
    plt.title("Évolution de la fonction de phase sur une période")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def radial_velocity(times_days: np.ndarray,
                    P_days: float,
                    v_rad,
                    t_obs: float,
                    v_obs: float):
    frac = times_days / P_days
    v = np.asarray(v_rad, dtype=float)
    frac_obs = t_obs / P_days

    plt.figure(num=6, figsize=(6,4))
    plt.clf()
    plt.plot(frac, v, '-', label='v_rad(t)')
    plt.plot(frac_obs, v_obs, 'ro', label=f't_obs = {t_obs:.1f} j')
    plt.xlim(0, 1)
    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Vitesse radiale vₕ(t) [m/s]")
    plt.title("Courbe de la vitesse radiale sur une période")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def contrast(times_days: np.ndarray,
             P_days: float,
             C: np.ndarray,
             C_obs: float,
             t_obs: float):
    
    frac = times_days / P_days
    frac_obs = t_obs / P_days

    plt.figure(num=7, figsize=(6, 4))
    plt.clf()
    plt.plot(frac, C, '-', label='Contraste C(t)')
    plt.plot(frac_obs, C_obs, 'ro', label=f't_obs = {t_obs:.1f} j')
    plt.xlim(0, 1)
    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Contraste C(t)")
    plt.title("Courbe du contraste sur une période")
    plt.yscale('log')  # ← pour afficher C(t) en échelle log
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def distance_SP(times_days, P_days,
                X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                t_obs: float):
    """
    Affiche la distance étoile-planète en fonction du temps,
    calculée géométriquement dans le référentiel 3D (X, Y, Z).
    """
    r = np.sqrt(X**2 + Y**2 + Z**2)  # Distance réelle (UA)
    P = P_days
    frac = times_days / P
    frac_obs = t_obs / P

    # Valeur à t_obs (approximation par le plus proche indice)
    i = np.argmin(np.abs(times_days - t_obs))
    r_obs = r[i]

    # Plot
    plt.figure(num=8, figsize=(6,4))
    plt.clf()
    plt.plot(frac, r, '-', label='r(t)')
    plt.plot(frac_obs, r_obs, 'ro', label=f't_obs = {t_obs:.1f} j')
    plt.xlim(0,1)
    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Distance étoile-planète (UA)")
    plt.title("Variation de la distance planète-étoile sur une période")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def absolute_velocity(times_days: np.ndarray,
                      P_days: float,
                      v_abs: np.ndarray,
                      v_obs: float,
                      t_obs: float):
    frac = times_days / P_days
    frac_obs = t_obs / P_days

    plt.figure(num=9, figsize=(6,4))
    plt.clf()
    plt.plot(frac, v_abs, '-', label='|v|(t)')
    plt.plot(frac_obs, v_obs, 'ro', label=f't_obs = {t_obs:.1f} j')
    plt.xlim(0, 1)
    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Vitesse absolue (m/s)")
    plt.title("Courbe de la vitesse absolue sur une période")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def albedo(times_days: np.ndarray,
           P_days: float,
           Ag_app: np.ndarray,
           Ag_max: float = 1.0,
           Ag_obs: float | None = None,
           t_obs: float | None = None,
           use_log: bool = True,
           eps: float | None = None):
    """
    Trace l'albédo/phase apparent(e) en fonction du temps.
    - use_log=True : axe Y logarithmique robuste (0 interdit -> on clippe).
    - eps : seuil min (si None, choisi automatiquement).

    Ag_app doit être compris dans [0,1] physiquement ; on clippe si besoin.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # x = fraction de période (float pur)
    frac = np.asarray(times_days, dtype=float) / float(P_days)

    # y = albédo apparent, clip physique [0,1]
    y = np.asarray(Ag_app, dtype=float)
    y = np.clip(y, 0.0, 1.0)

    # borne supérieure physique
    ymax_phys = min(1.0, float(Ag_max) if np.isfinite(Ag_max) else 1.0)

    # sécurisation log : choisir eps automatiquement si non fourni
    if use_log:
        # min strictly positive in data (si tout est 0, on prend 1e-6)
        pos = y[y > 0]
        if eps is None:
            if pos.size:
                # un seuil ~100× plus petit que la plus petite valeur non nulle, borné à 1e-6
                eps = max(1e-6, float(np.nanmin(pos)) / 100.0)
            else:
                eps = 1e-6
        # on force un plancher strictly > 0
        y_plot = np.clip(y, eps, ymax_phys)
    else:
        y_plot = y
        eps = None  # pas utilisé

    # point d'observation (optionnel)
    have_obs = (Ag_obs is not None) and (t_obs is not None)
    if have_obs:
        frac_obs = float(t_obs) / float(P_days)
        y_obs = float(Ag_obs)
        y_obs = np.clip(y_obs, eps if use_log else 0.0, ymax_phys)

    # tracé
    plt.figure(num=10, figsize=(6, 4))
    plt.clf()
    plt.plot(frac, y_plot, linewidth=2, label="Albédo apparent")

    if have_obs:
        plt.plot(frac_obs, y_obs, 'ro', label=f"t_obs = {t_obs:.1f} j")

    plt.xlim(0, 1)

    if use_log:
        plt.yscale('log')
        # bornes Y stables et propres
        ymin = eps
        ymax = max(np.nanmax(y_plot), ymin * 10)
        plt.ylim(ymin, ymax * 1.05)
    else:
        plt.ylim(0, ymax_phys * 1.05)

    plt.xlabel("Temps depuis périastre (fraction de période)")
    plt.ylabel("Albédo géométrique apparent")
    plt.title("Variation apparente de l'albédo géométrique")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()



# Tracés 3D et 2D orbitaux
    
def orbit_3d(compute_orbit, compute_points, velocity_vectors,
             a, e, inc, omega, Omega,
             times_days: np.ndarray,
             f_t: np.ndarray,
             days: float, Mstar: float, Mplanet:float, distance_pc: float):
    """
Affiche l'orbite 3D, positionne la planète à l'instant days et dessine
ses vecteurs de vitesse.

Parameters
----------
compute_orbit : function
    Renvoie X, Y, Z (UA) pour l'orbite complète.
compute_points : function
    Renvoie (planet, peri, asc, desc) pour un angle f0_deg.
velocity_vectors : function
    Renvoie (times_days, vx, vy, vz).
a, e : float
    Demi-grand axe (UA) et excentricité.
inc, omega, Omega : float
    Angles en radians.
times_days : ndarray
    Grille temporelle (jours).
f_t : ndarray
    Anomalie vraie (radians).
days : float
    Jour depuis le périastre pour positionner la planète.
"""


    # 1) calcul des points
    i = np.argmin(np.abs(times_days - days))
    f0_deg = np.degrees(f_t[i])
    X, Y, Z = compute_orbit(a, e, inc, omega, Omega)
    planet, peri, asc, desc = compute_points(a, e, inc, omega, Omega, f0_deg)
    
    # --- Conversion en arcsec ---
    facteur_arcsec = 1 / distance_pc
    X, Y, Z = np.array(X) * facteur_arcsec, np.array(Y) * facteur_arcsec, np.array(Z) * facteur_arcsec
    planet = np.array(planet) * facteur_arcsec
    peri   = np.array(peri) * facteur_arcsec
    asc    = np.array(asc) * facteur_arcsec
    desc   = np.array(desc) * facteur_arcsec


    # 2) vecteurs vitesse
    vx, vy, vz = velocity_vectors(times_days=times_days,
    a_AU=a,
    e=e,
    inc=inc,
    omega=omega,
    Omega=Omega,
    P_days=(times_days[-1] - times_days[0]),
    Mstar=Mstar,
    Mplanet=Mplanet)

    # 3) tracé 3D
    fig = plt.figure(num=11, figsize=(8,6), clear=True)
    ax  = fig.add_subplot(111, projection='3d')
    lim = 1.5 * a

    # plan écliptique
    xx, yy = np.meshgrid([-lim, lim], [-lim, lim])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='lightblue', alpha=0.3)

    # noeuds, périastre, orbite, étoiles
    ax.plot([asc[0], desc[0]], [asc[1], desc[1]], [asc[2], desc[2]], '--k', label='Noeuds')
    ax.plot([0, peri[0]], [0, peri[1]], [0, peri[2]], ':k',  label='Périastre')
    ax.scatter(*asc,  color='green', marker='^', s=80, label='Noeud ascendant')
    ax.scatter(*desc, color='red',   marker='v', s=80, label='Noeud descendant')
    ax.scatter(*peri, color='black', marker='x', s=100, label='Périastre')
    ax.quiver(0,0,0, 0,0,1, length= 0.5, normalize=True, arrow_length_ratio=0.2, color='k', label='Visée')
    ax.plot(X, Y, Z, '-k', linewidth=2, label='Orbite')
    ax.scatter(*planet, color='blue',  s=80, label='Planète')
    ax.scatter(0,0,0, color='yellow', s=200, label='Étoile')

    # vecteurs vitesse
    lim = 1.5 * a * facteur_arcsec
    L = lim * 0.2
    for comp, col, txt in zip([vx, vy, vz], ['r','g','b'], ['v_x','v_y','v_z']):
        if comp is vx:
            sx, sy, sz = np.sign(comp[i]), 0, 0
        elif comp is vy:
            sx, sy, sz = 0, np.sign(comp[i]), 0
        else:  # comp is vz
            sx, sy, sz = 0, 0, np.sign(comp[i])

        ax.quiver(planet[0], planet[1], planet[2], sx, sy, sz,
                  length=L, normalize=True, color=col, arrow_length_ratio=0.2)
        ax.text(planet[0]+sx*L*1.1, planet[1]+sy*L*1.1, planet[2]+sz*L*1.1,
                txt, color=col)

    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    ax.set_xlabel('X (arcsec)')
    ax.set_ylabel('Y (arcsec)')
    ax.set_zlabel('Z (arcsec)')
    ax.set_title(f'Orbite 3D – t={days:.1f} d')
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.show()

def orbit_plane(compute_orbit, compute_points,
                a, e, inc, omega, Omega,
                times_days: np.ndarray,
                f_t: np.ndarray,
                days: float,
                distance_pc: float,
                views=('xy','xz','yz')):
    """
    Projections 2D de l'orbite et position de la planète à days.
    """
    i = np.argmin(np.abs(times_days - days))
    f0_deg = np.degrees(f_t[i])
    X, Y, Z = compute_orbit(a, e, inc, omega, Omega)
    planet, peri, asc, desc = compute_points(a, e, inc, omega, Omega, f0_deg)
    
    facteur_arcsec = 1 / distance_pc
    X, Y, Z = np.array(X) * facteur_arcsec, np.array(Y) * facteur_arcsec, np.array(Z) * facteur_arcsec
    planet = np.array(planet) * facteur_arcsec
    peri   = np.array(peri) * facteur_arcsec
    asc    = np.array(asc) * facteur_arcsec
    desc   = np.array(desc) * facteur_arcsec
    
    fig, axes = plt.subplots(1,3, num=12, figsize=(15,5), clear=True)
    lim = 2 * a * facteur_arcsec
    idxs = {'xy':(0,1), 'xz':(0,2), 'yz':(1,2)}
    
    for ax, view in zip(axes, views):
        ax.fill([-lim,lim,lim,-lim], [-lim,-lim,lim,lim], color='lightblue', alpha=0.3) if view == 'xy' else ax.axhline(0, color='lightblue', linewidth=2, alpha=0.3)

        xi, yi = idxs[view]
        coords = (X, Y) if view == 'xy' else ((X, Z) if view == 'xz' else (Y, Z))

        # Orbite et ligne du périastre/noeuds
        ax.plot(*coords, '-k', label='Orbite')
        ax.plot([asc[xi], desc[xi]], [asc[yi], desc[yi]], '--k', label='Noeuds')
        ax.plot([0, peri[xi]], [0, peri[yi]], ':k', label='Périastre')

        # Objets
        ax.scatter(planet[xi], planet[yi], color='blue', s=80, label='Planète')
        ax.scatter(0, 0, color='yellow', s=200, label='Étoile')

        # Ajouts visuels : croix et triangles
        ax.scatter(peri[xi], peri[yi], color='black', marker='x', s=80, label='Périastre (croix)')
        ax.scatter(asc[xi],  asc[yi],  color='green', marker='^', s=80, label='Noeud ascendant')
        ax.scatter(desc[xi], desc[yi], color='red',   marker='v', s=80, label='Noeud descendant')

        # Mise en forme
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(view[0].upper() + ' (arcsec)')
        ax.set_ylabel(view[1].upper() + ' (arcsec)')
        ax.set_title(f'{view.upper()} plane – t={days:.1f} d')
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
    
def albedo_phase_spec(wave_nm: np.ndarray,
                      Ag_spec: np.ndarray,
                      phi: float,
                      label: str = None):
    """
    Trace l'albédo apparent A_p(λ) = φ · A_g(λ) sur la grille wave_nm.

    Parameters
    ----------
    wave_nm : ndarray
        Longueurs d'onde en nm.
    Ag_spec : ndarray
        Albédo géométrique spectrale (sans unité), même shape que wave_nm.
    phi : float
        Fonction de phase au temps d'observation (sans unité).
    label : str, optional
        Étiquette pour la légende.
    """
    wave = np.asarray(wave_nm, dtype=float)
    Ag   = np.asarray(Ag_spec, dtype=float)
    Ap   = phi * Ag

    plt.figure(num=13, figsize=(8,4))
    plt.clf()
    plt.plot(wave, Ap, label=label or f'Apparent Aₚ(λ) (φ={phi:.2f})')
    plt.xlabel("λ (nm)")
    plt.ylabel("Albédo apparent Aₚ(λ)")
    plt.title("Spectre de l’albédo apparent")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
