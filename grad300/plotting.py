"""Visualization routines for GRAD-300 analysis."""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def project_tpi_to_map(result, nx=51, ny=51, method='histogram2d',
                        bbc_column=None, statistic='mean'):
    """Project RA/Dec points into a 2‑D grid.

    Parameters
    ----------
    result : dict
        Must contain 'ra' and 'dec' arrays.  If ``bbc_column`` is provided the
        dictionary should also contain ``'clean'`` with a structured numpy array
        including that column.
    nx, ny : int
        Number of bins in RA and Dec.
    method : {'loop','histogram2d','binned_statistic'}
        Projection algorithm.
    bbc_column : str or None
        Name of the BBC column to aggregate.
    statistic : str
        Statistic used when ``method='binned_statistic'``.

    Returns
    -------
    dict
        Contains 'hit_map','data_map','ra_edges','dec_edges','ra_center','dec_center',
        'nx','ny'.
    """
    ra = result['ra']
    dec = result['dec']
    
    # Handle empty arrays
    if len(ra) == 0 or len(dec) == 0:
        return {
            'hit_map': np.zeros((ny, nx)),
            'data_map': np.zeros((ny, nx)),
            'ra_edges': np.linspace(0, 1, nx + 1),
            'dec_edges': np.linspace(0, 1, ny + 1),
            'ra_center': np.linspace(0, 1, nx),
            'dec_center': np.linspace(0, 1, ny),
            'nx': nx, 'ny': ny
        }

    ra_min, ra_max = np.min(ra), np.max(ra)
    dec_min, dec_max = np.min(dec), np.max(dec)
    
    # Add margin (5% of range, or 1 degree if range is zero)
    ra_range = max(ra_max - ra_min, 1.0)
    dec_range = max(dec_max - dec_min, 1.0)
    ra_margin = ra_range * 0.05
    dec_margin = dec_range * 0.05

    ra_edges = np.linspace(ra_min - ra_margin, ra_max + ra_margin, nx + 1)
    dec_edges = np.linspace(dec_min - dec_margin, dec_max + dec_margin, ny + 1)
    ra_center = (ra_edges[:-1] + ra_edges[1:]) / 2
    dec_center = (dec_edges[:-1] + dec_edges[1:]) / 2

    hit_map = np.zeros((ny, nx))
    data_map = np.zeros((ny, nx))

    if method == 'loop':
        for i in range(len(ra)):
            ix = np.digitize(ra[i], ra_edges) - 1
            iy = np.digitize(dec[i], dec_edges) - 1
            if 0 <= ix < nx and 0 <= iy < ny:
                hit_map[iy, ix] += 1
                if bbc_column and bbc_column in result['clean'].dtype.names:
                    data_map[iy, ix] += result['clean'][bbc_column][i]
        if bbc_column and np.sum(hit_map) > 0:
            np.divide(data_map, hit_map, where=hit_map > 0, out=data_map)

    elif method == 'histogram2d':
        hit_map, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
        if bbc_column and bbc_column in result['clean'].dtype.names:
            data_sum, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges],
                                            weights=result['clean'][bbc_column])
            np.divide(data_sum, hit_map, where=hit_map > 0, out=data_map)

    elif method == 'binned_statistic':
        hit_map, _, _, _ = stats.binned_statistic_2d(
            dec, ra, None, statistic='count', bins=[dec_edges, ra_edges]
        )
        if bbc_column and bbc_column in result['clean'].dtype.names:
            data_map, _, _, _ = stats.binned_statistic_2d(
                dec, ra, result['clean'][bbc_column],
                statistic=statistic, bins=[dec_edges, ra_edges]
            )

    return {
        'hit_map': hit_map,
        'data_map': data_map,
        'ra_edges': ra_edges,
        'dec_edges': dec_edges,
        'ra_center': ra_center,
        'dec_center': dec_center,
        'nx': nx, 'ny': ny
    }


def plot_tpi_maps_comparison(maps_dict, target, time, out_dir, bbc_column=None):
    """Create a six-panel figure comparing hit and data maps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    hit_map = maps_dict['hit_map']
    data_map = maps_dict['data_map']
    ra_center = maps_dict['ra_center']
    dec_center = maps_dict['dec_center']
    extent = [ra_center[0], ra_center[-1], dec_center[0], dec_center[-1]]

    im0 = axes[0, 0].imshow(hit_map, origin='lower', extent=extent,
                              cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f"Hit map - {np.sum(hit_map>0)} pixels occupés")
    axes[0, 0].set_xlabel("RA (deg)")
    axes[0, 0].set_ylabel("Dec (deg)")
    axes[0, 0].invert_xaxis()
    plt.colorbar(im0, ax=axes[0, 0], label="Nombre d'échantillons")

    if np.any(hit_map > 0):
        axes[0, 1].hist(hit_map[hit_map > 0].flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_xlabel("Échantillons par pixel")
        axes[0, 1].set_ylabel("Nombre de pixels")
        axes[0, 1].set_title("Distribution échantillonnage")
        axes[0, 1].axvline(np.mean(hit_map[hit_map>0]), color='r', ls='--',
                           label=f"Moyenne: {np.mean(hit_map[hit_map>0]):.2f}")
        axes[0, 1].legend()

    if np.any(data_map != 0):
        im2 = axes[0, 2].imshow(data_map, origin='lower', extent=extent,
                                cmap='inferno', aspect='auto')
        axes[0, 2].set_title(f"Carte {bbc_column if bbc_column else 'signal'}")
        axes[0, 2].set_xlabel("RA (deg)")
        axes[0, 2].set_ylabel("Dec (deg)")
        axes[0, 2].invert_xaxis()
        plt.colorbar(im2, ax=axes[0, 2], label="Signal (ADU)")

    axes[1, 0].imshow(hit_map > 0, origin='lower', extent=extent,
                      cmap='gray', aspect='auto')
    axes[1, 0].set_title("Couverture binaire")
    axes[1, 0].set_xlabel("RA (deg)")
    axes[1, 0].set_ylabel("Dec (deg)")
    axes[1, 0].invert_xaxis()

    stats_text = (
        f"Points totaux: {np.sum(hit_map):.0f}\n"
        f"Pixels occupés: {np.sum(hit_map>0)}/{maps_dict['nx']*maps_dict['ny']}\n"
        f"Taux couverture: {np.sum(hit_map>0)/(maps_dict['nx']*maps_dict['ny'])*100:.1f}%\n"
        f"Max par pixel: {np.max(hit_map):.0f}"
    )
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].axis('off')

    # Distribution 2D des points RA/Dec
    try:
        # Utiliser les points originaux plutôt que les centres de bins
        if 'ra' in maps_dict and 'dec' in maps_dict:
            ra_points = maps_dict['ra']
            dec_points = maps_dict['dec']
        else:
            # Fallback aux centres de bins
            ra_points = ra_center
            dec_points = dec_center
            
        if len(ra_points) > 10 and len(dec_points) > 10:
            axes[1, 2].hist2d(ra_points, dec_points, bins=30, cmap='plasma')
            axes[1, 2].set_xlabel("RA (deg)")
            axes[1, 2].set_ylabel("Dec (deg)")
            axes[1, 2].set_title("Distribution 2D des points")
            axes[1, 2].invert_xaxis()
    except Exception as e:
        print(f"     ⚠️  Erreur histogram2d: {e}")
        axes[1, 2].axis('off')

    plt.suptitle(f"{target} - {time} - Cartes d'échantillonnage")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{target}_tpi_maps_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     💾 Cartes complètes: {out_path}")
    return out_path


def study_resolution_effect_optimized(result, target, time, out_dir,
                                       min_res=14, max_res=51,
                                       metric='count', ref_shape=None):
    """Search best square resolution for TPI projection.

    The function analyses a range of square grids and returns the resolution
    that maximises the chosen metric.  By default it uses the *occupied pixel
    ratio* (occupied / total bins) which matches the behaviour requested in the
    current discussion.  For historical reasons the helper also supports
    ``metric='count'`` which picks the largest number of occupied cells.

    Optionally a reference resolution may be supplied (e.g. the shape of a
    corresponding image) and the function will compare its ratio to the best
    computed value, favouring the reference if it is equal or better.

    Parameters
    ----------
    result : dict
        Same as :func:`project_tpi_to_map`.
    target, time : str
        Used for titles and file names when plotting.
    out_dir : str
        Directory where the diagnostic figure is saved.
    min_res, max_res : int
        Range of square resolutions to try (inclusive).  Defaults mirror the
        original behaviour (14–51).
    metric : {'count', 'ratio'}
        Selection criterion. ``'count'`` (default) chooses the grid with the
        maximal number of occupied pixels; ``'ratio'`` picks the maximal
        occupancy fraction.
    ref_shape : tuple or None
        Optional `(nx, ny)` resolution of a reference image.  If provided the
        occupied ratio for this grid is computed and compared; the returned
        resolution will be ``ref_shape`` when its ratio is >= the best
        candidate.

    Returns
    -------
    best : tuple
        The (nx, ny) resolution ultimately selected.
    ratio_max : float
        Occupied‑pixel ratio corresponding to ``best``.
    ref_ratio : float or None
        Ratio computed for ``ref_shape`` (if given), otherwise ``None``.  Can
        be used by the caller for additional messaging.
    """
    resolutions = [(i, i) for i in range(min_res, max_res + 1)]
    counts = []
    ratios = []
    
    # Check if result has valid data
    if result is None or 'ra' not in result or len(result['ra']) == 0:
        print("     ⚠️  Pas de données valides pour l'étude de résolution")
        default_res = (min_res, min_res)
        return default_res, 0.0, None
    
    for rx, ry in resolutions:
        maps = project_tpi_to_map(result, nx=rx, ny=ry, method='binned_statistic')
        occupied = np.sum(maps['hit_map'] > 0)
        counts.append(occupied)
        ratios.append(occupied / (rx * ry))

    # choose index based on metric
    if metric == 'count':
        idx = np.argmax(counts)
    else:
        idx = np.argmax(ratios)

    best = resolutions[idx]
    ratio_max = ratios[idx]

    ref_ratio = None
    if ref_shape is not None:
        # compute ratio for reference resolution, unless it matches one we
        # already computed
        if ref_shape in resolutions:
            ref_ratio = ratios[resolutions.index(ref_shape)]
        else:
            maps = project_tpi_to_map(result, nx=ref_shape[0], ny=ref_shape[1],
                                      method='binned_statistic')
            ref_ratio = np.sum(maps['hit_map'] > 0) / (ref_shape[0] * ref_shape[1])
        if ref_ratio >= ratio_max:
            print(f"     🔁 Reference resolution {ref_shape} beats/equals best "
                  f"candidate (ratio {ref_ratio:.4f} >= {ratio_max:.4f}); using it")
            best = ref_shape
            ratio_max = ref_ratio

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [str(r[0]) for r in resolutions]
    x = np.arange(len(resolutions))
    colors = plt.cm.viridis(np.array(ratios) / max(ratios) if max(ratios) > 0 else 1)
    bars = ax.bar(x, ratios, color=colors, alpha=0.7)
    bars[idx].set_color('red')
    bars[idx].set_alpha(0.9)
    ax.set_xlabel("Résolution (N x N)")
    ax.set_ylabel("Ratio pixels occupés / surface totale")
    ax.set_title(
        f"{target} - {time} - résolution optimale: {best[0]}x{best[1]} "
        f"(ratio={ratio_max:.4f})"
    )
    step = max(1, len(resolutions) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=np.mean(ratios), color='gray', linestyle='--', alpha=0.5,
               label=f'Moyenne: {np.mean(ratios):.4f}')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{target}_resolution_optimale_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     💾 Résolution optimale: {best[0]}x{best[1]} (ratio={ratio_max:.4f})")
    return best, ratio_max, ref_ratio


def compare_bbc_maps(result, target, time, out_dir, bbc_cols, nx=None, ny=None):
    """Plot a 2x2 grid comparing BBC signal maps.

    If nx/ny not provided the optimal resolution is computed first.
    """
    if nx is None or ny is None:
        print("     🔍 Calcul de la résolution optimale pour la comparaison...")
        (nx, ny), ratio, _ = study_resolution_effect_optimized(result, target, time, out_dir)
        print(f"     Utilisation de la résolution optimale: {nx} x {ny} (ratio={ratio:.4f})")

    n_bbc = min(len(bbc_cols), 4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    ra_min, ra_max = np.min(result['ra']), np.max(result['ra'])
    dec_min, dec_max = np.min(result['dec']), np.max(result['dec'])
    ra_range = max(ra_max - ra_min, 1.0)
    dec_range = max(dec_max - dec_min, 1.0)
    ra_margin = ra_range * 0.05
    dec_margin = dec_range * 0.05
    extent = [ra_min - ra_margin, ra_max + ra_margin,
              dec_min - dec_margin, dec_max + dec_margin]

    # Find global min/max for consistent color scaling
    vmin, vmax = np.inf, -np.inf
    for bbc in bbc_cols[:n_bbc]:
        maps = project_tpi_to_map(result, nx=nx, ny=ny,
                                  method='binned_statistic',
                                  bbc_column=bbc, statistic='mean')
        data_valid = maps['data_map'][maps['hit_map'] > 0]
        if data_valid.size:
            vmin = min(vmin, np.min(data_valid))
            vmax = max(vmax, np.max(data_valid))
    if vmin == np.inf:
        vmin = vmax = None

    # Plot each BBC
    for i, bbc in enumerate(bbc_cols[:n_bbc]):
        maps = project_tpi_to_map(result, nx=nx, ny=ny,
                                  method='binned_statistic',
                                  bbc_column=bbc, statistic='mean')
        data_map = maps['data_map']
        hit_map = maps['hit_map']
        
        if np.any(hit_map > 0):
            masked = np.ma.masked_where(hit_map == 0, data_map)
            im = axes[i].imshow(masked, origin='lower', extent=extent,
                                cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{bbc} - {np.sum(hit_map>0)} pixels occupés")
            axes[i].set_xlabel("RA (deg)")
            axes[i].set_ylabel("Dec (deg)")
            axes[i].invert_xaxis()
            plt.colorbar(im, ax=axes[i], fraction=0.046, label='Signal (ADU)')
            
            # Statistics
            if np.any(hit_map > 0):
                stats_txt = (f"Max: {np.max(data_map):.1f}\n"
                            f"Moy: {np.mean(data_map[hit_map>0]):.1f}\n"
                            f"Min: {np.min(data_map[hit_map>0]):.1f}")
                axes[i].text(0.05, 0.95, stats_txt, transform=axes[i].transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[i].text(0.5, 0.5, f"Pas de données\npour {bbc}",
                         ha='center', va='center', transform=axes[i].transAxes,
                         fontsize=12, color='red')
    
    # Turn off unused subplots
    for j in range(i+1, 4):
        axes[j].axis('off')
    
    plt.suptitle(f"{target} - {time} - Comparaison des BBCs (résolution {nx}x{ny})", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{target}_bbc_comparison_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     💾 Comparaison BBCs: {out_path}")
    return nx, ny


def compare_resolutions(result, target, time, out_dir, res_opt, res_orig, ratio_opt, ratio_orig):
    """Plot side-by-side comparison of optimized vs original resolution hit maps.

    Parameters
    ----------
    result : dict
        Same as :func:`project_tpi_to_map`.
    target, time : str
        Used for titles and file names.
    out_dir : str
        Directory where the figure is saved.
    res_opt : tuple
        Optimized (nx, ny) resolution.
    res_orig : tuple
        Original (nx, ny) resolution.
    ratio_opt : float
        Occupancy ratio for optimized resolution.
    ratio_orig : float
        Occupancy ratio for original resolution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Optimized resolution
    maps_opt = project_tpi_to_map(result, nx=res_opt[0], ny=res_opt[1], method='binned_statistic')
    im0 = axes[0].imshow(maps_opt['hit_map'], origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title(f"Optimisée: {res_opt[0]}x{res_opt[1]} (ratio={ratio_opt:.6f})", 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("RA (pixels)")
    axes[0].set_ylabel("Dec (pixels)")
    occupied_opt = np.sum(maps_opt['hit_map'] > 0)
    axes[0].text(0.05, 0.95, f"{occupied_opt}/{res_opt[0]*res_opt[1]} pixels occupés",
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im0, ax=axes[0], label="Échantillons")

    # Original resolution
    maps_orig = project_tpi_to_map(result, nx=res_orig[0], ny=res_orig[1], method='binned_statistic')
    im1 = axes[1].imshow(maps_orig['hit_map'], origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title(f"Originale: {res_orig[0]}x{res_orig[1]} (ratio={ratio_orig:.6f})", 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("RA (pixels)")
    axes[1].set_ylabel("Dec (pixels)")
    occupied_orig = np.sum(maps_orig['hit_map'] > 0)
    axes[1].text(0.05, 0.95, f"{occupied_orig}/{res_orig[0]*res_orig[1]} pixels occupés",
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(im1, ax=axes[1], label="Échantillons")

    plt.suptitle(f"{target} - {time} - Comparaison des résolutions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{target}_resolution_comparison_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     💾 Comparaison résolutions: {out_path}")
    return out_path


def plot_rfi_timeline(rfi_detected, target, time, out_dir):
    """Plot timeline of RFI detections across BBCs.
    
    Parameters
    ----------
    rfi_detected : dict
        RFI detection dictionary from detect_rfi_in_tpi()
    target, time : str
        Target name and observation time
    out_dir : str
        Output directory
        
    Returns
    -------
    str or None
        Path to saved figure, or None if no RFI
    """
    if not rfi_detected:
        print("     ℹ️  Aucun RFI détecté à visualiser")
        return None
    
    n_bbc = len(rfi_detected)
    fig, axes = plt.subplots(n_bbc, 1, figsize=(12, 3*n_bbc), sharex=True)
    
    # Handle single BBC case
    if n_bbc == 1:
        axes = [axes]
    
    for i, (bbc_col, rfi_info) in enumerate(sorted(rfi_detected.items())):
        timestamps = rfi_info['timestamps']
        severity = rfi_info['severity']
        method = rfi_info.get('method', 'unknown')
        
        # Convert JD to relative time (minutes)
        if len(timestamps) > 0:
            t_rel = (timestamps - timestamps[0]) * 24 * 60  # minutes
            
            # Scatter plot with severity color
            scatter = axes[i].scatter(t_rel, [i]*len(t_rel), 
                                      c=severity, cmap='hot', 
                                      s=50, alpha=0.7, vmin=threshold)
            axes[i].set_yticks([])
            axes[i].set_ylabel(bbc_col, rotation=0, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            axes[i].text(0.98, 0.95, 
                        f"{rfi_info['count']} pts ({rfi_info['percentage']:.1f}%)",
                        transform=axes[i].transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[i].text(0.5, 0.5, "Aucun RFI", ha='center', va='center')
    
    axes[-1].set_xlabel("Temps relatif (minutes)")
    plt.suptitle(f"{target} - {time} - RFI détecté (méthode: {method})")
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"{target}_rfi_timeline_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     💾 Timeline RFI: {out_path}")
    return out_path


def plot_fwhm_estimation(data_map, ra_center, dec_center, target, time, out_dir):
    """Plot a 2D map with FWHM profiles and estimation results.

    Parameters
    ----------
    data_map : ndarray
        2D intensity map
    ra_center : ndarray
        RA coordinates (degrees)
    dec_center : ndarray
        Dec coordinates (degrees)
    target : str
        Target name
    time : str
        Observation timestamp
    out_dir : str
        Output directory for saving figure

    Returns
    -------
    dict
        FWHM estimation results
    """
    try:
        from grad300.fwhm import estimate_fwhm_2d_map
    except ImportError:
        print("  ⚠️  fwhm module not available")
        return None

    # Clean NaN values from data_map (replace with 0)
    data_map_clean = np.nan_to_num(data_map, nan=0.0)
    
    # Check if there's valid data
    if np.sum(data_map_clean) == 0:
        print("  ⚠️  Pas de données valides pour l'estimation FWHM")
        return None

    # Estimate FWHM
    fwhm_result = estimate_fwhm_2d_map(data_map_clean, ra_center, dec_center, method='gaussian')

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))

    # Main map
    ax1 = plt.subplot(131)
    im = ax1.imshow(data_map_clean, extent=[ra_center[0], ra_center[-1], dec_center[0], dec_center[-1]],
                    origin='lower', cmap='viridis', aspect='auto')
    
    # Mark peak
    peak_ra = ra_center[fwhm_result['peak_ra']]
    peak_dec = dec_center[fwhm_result['peak_dec']]
    ax1.plot(peak_ra, peak_dec, 'r+', markersize=15, markeredgewidth=2)
    
    ax1.set_xlabel("RA (°)")
    ax1.set_ylabel("Dec (°)")
    ax1.set_title(f"{target} - {time}")
    plt.colorbar(im, ax=ax1, label="Intensité")

    # RA profile
    ax2 = plt.subplot(132)
    profile_ra = data_map_clean[fwhm_result['peak_dec'], :]
    ax2.plot(ra_center, profile_ra, 'b-', linewidth=2, label='Profil RA')
    
    if fwhm_result['fwhm_ra'] is not None:
        # Mark FWHM points
        half_max = fwhm_result['peak_value'] / 2
        ax2.axhline(half_max, color='r', linestyle='--', alpha=0.5, label='Demi-max')
        ax2.text(0.5, 0.95, f'FWHM = {fwhm_result["fwhm_ra"]:.3f}°', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel("RA (°)")
    ax2.set_ylabel("Intensité")
    ax2.set_title("Profil RA")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Dec profile
    ax3 = plt.subplot(133)
    profile_dec = data_map_clean[:, fwhm_result['peak_ra']]
    ax3.plot(dec_center, profile_dec, 'g-', linewidth=2, label='Profil Dec')
    
    if fwhm_result['fwhm_dec'] is not None:
        # Mark FWHM points
        half_max = fwhm_result['peak_value'] / 2
        ax3.axhline(half_max, color='r', linestyle='--', alpha=0.5, label='Demi-max')
        ax3.text(0.5, 0.95, f'FWHM = {fwhm_result["fwhm_dec"]:.3f}°', 
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax3.set_xlabel("Dec (°)")
    ax3.set_ylabel("Intensité")
    ax3.set_title("Profil Dec")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    out_path = os.path.join(out_dir, f"{target}_fwhm_{time}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print results
    print(f"\n  📏 Estimation FWHM:")
    if fwhm_result.get('fwhm_ra'):
        print(f"     FWHM RA:  {fwhm_result['fwhm_ra']:.4f}°")
    if fwhm_result.get('fwhm_dec'):
        print(f"     FWHM Dec: {fwhm_result['fwhm_dec']:.4f}°")
    print(f"     💾 {out_path}")

    return fwhm_result