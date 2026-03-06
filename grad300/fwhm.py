"""FWHM (Full Width at Half Maximum) estimation for GRAD-300 data."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def estimate_fwhm_parabolic(profile, freqs=None):
    """Estimate FWHM using parabolic interpolation around the peak.

    Parameters
    ----------
    profile : ndarray
        1D array of intensities (spectrum, spatial profile, etc.)
    freqs : ndarray, optional
        Frequency/distance array corresponding to profile. If None,
        assumes uniform spacing from 0 to len(profile)-1.

    Returns
    -------
    dict
        Dictionary containing:
        - 'peak_value': maximum intensity
        - 'peak_index': index of maximum
        - 'peak_freq': frequency/distance at peak (if freqs provided)
        - 'fwhm': estimated FWHM
        - 'fwhm_freq': FWHM in frequency/distance units (if freqs provided)
        - 'half_max': value at half maximum
    """
    if len(profile) < 3:
        return {'peak_value': np.max(profile), 'fwhm': None, 'error': 'Profile too short'}

    # Find peak
    peak_idx = np.argmax(profile)
    peak_value = profile[peak_idx]
    half_max = peak_value / 2.0

    # Get frequency axis if not provided
    if freqs is None:
        freqs = np.arange(len(profile))

    result = {
        'peak_value': peak_value,
        'peak_index': peak_idx,
        'peak_freq': freqs[peak_idx],
        'half_max': half_max,
    }

    # Find indices where profile crosses half maximum
    above_half = profile >= half_max
    
    if np.sum(above_half) < 3:
        result['fwhm'] = None
        result['error'] = 'Profile too narrow or noisy'
        return result

    # Find edges of the region above half maximum
    diff = np.diff(above_half.astype(int))
    # Transitions from False to True (rising edge)
    rising = np.where(diff == 1)[0]
    # Transitions from True to False (falling edge)
    falling = np.where(diff == -1)[0]

    # Handle cases where peak is at edge
    if above_half[0]:
        rising = np.insert(rising, 0, 0)
    if above_half[-1]:
        falling = np.append(falling, len(above_half) - 1)

    if len(rising) == 0 or len(falling) == 0:
        result['fwhm'] = None
        result['error'] = 'Could not find half-max crossings'
        return result

    # Use the first rising and last falling edges (main peak)
    left_idx = rising[0]
    right_idx = falling[-1]

    # Interpolate to find more precise half-max crossing points
    if left_idx > 0:
        # Linear interpolation on left side
        try:
            x_left = np.interp(half_max, [profile[left_idx - 1], profile[left_idx]], 
                             [freqs[left_idx - 1], freqs[left_idx]])
        except:
            x_left = freqs[left_idx]
    else:
        x_left = freqs[left_idx]

    if right_idx < len(profile) - 1:
        # Linear interpolation on right side
        try:
            x_right = np.interp(half_max, [profile[right_idx + 1], profile[right_idx]], 
                              [freqs[right_idx + 1], freqs[right_idx]])
        except:
            x_right = freqs[right_idx]
    else:
        x_right = freqs[right_idx]

    fwhm = x_right - x_left

    result['fwhm'] = fwhm
    result['fwhm_left_freq'] = x_left
    result['fwhm_right_freq'] = x_right

    return result


def estimate_fwhm_gaussian_fit(profile, freqs=None):
    """Estimate FWHM by fitting a Gaussian to the peak.

    Parameters
    ----------
    profile : ndarray
        1D array of intensities
    freqs : ndarray, optional
        Frequency/distance array. If None, assumes uniform spacing.

    Returns
    -------
    dict
        Dictionary containing:
        - 'peak_value': maximum intensity
        - 'fwhm': estimated FWHM from Gaussian fit
        - 'sigma': standard deviation of fitted Gaussian
        - 'fit_quality': R² of the fit
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {'error': 'scipy.optimize not available'}

    if len(profile) < 3:
        return {'error': 'Profile too short'}

    if freqs is None:
        freqs = np.arange(len(profile))

    # Find peak
    peak_idx = np.argmax(profile)
    peak_value = profile[peak_idx]

    # Extract region around peak (±3 sigma estimate)
    width = int(len(profile) / 4)  # Rough estimate
    left = max(0, peak_idx - width)
    right = min(len(profile), peak_idx + width + 1)

    x_fit = freqs[left:right]
    y_fit = profile[left:right]

    # Define Gaussian function
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Initial guesses
    mu0 = freqs[peak_idx]
    sigma0 = (freqs[right - 1] - freqs[left]) / 4
    amp0 = peak_value

    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, 
                              p0=[amp0, mu0, sigma0],
                              maxfev=2000)
        amp, mu, sigma = popt

        # FWHM = 2.355 * sigma for Gaussian
        fwhm = 2.355 * sigma

        # Calculate fit quality (R²)
        y_pred = gaussian(x_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'peak_value': peak_value,
            'peak_freq': mu,
            'fwhm': fwhm,
            'sigma': sigma,
            'amplitude': amp,
            'fit_quality': r_squared
        }
    except Exception as e:
        return {'error': f'Curve fit failed: {e}', 'peak_value': peak_value}


def estimate_fwhm_2d_map(data_map, ra_center, dec_center, method='parabolic'):
    """Estimate FWHM of a 2D source from TPI map by extracting 1D profiles.

    Parameters
    ----------
    data_map : ndarray
        2D array with the intensity map (bins × bins)
    ra_center : ndarray
        RA coordinates of map bins (in degrees)
    dec_center : ndarray
        Dec coordinates of map bins (in degrees)
    method : str
        Method: 'parabolic' or 'gaussian'

    Returns
    -------
    dict
        Dictionary containing:
        - 'fwhm_ra': FWHM along RA direction (degrees)
        - 'fwhm_dec': FWHM along Dec direction (degrees)
        - 'fwhm_radial': FWHM of radial profile (degrees)
        - 'peak_value': peak intensity
        - 'results_ra': full result dict for RA profile
        - 'results_dec': full result dict for Dec profile
        - 'results_radial': full result dict for radial profile
    """
    # Find peak in the map
    peak_idx = np.unravel_index(np.argmax(data_map), data_map.shape)
    peak_value = data_map[peak_idx]
    peak_ra_idx, peak_dec_idx = peak_idx[1], peak_idx[0]

    # Extract 1D profiles through the peak
    
    # RA profile (at peak Dec)
    profile_ra = data_map[peak_dec_idx, :]
    coords_ra = ra_center if len(ra_center) == data_map.shape[1] else np.arange(len(profile_ra))
    
    # Dec profile (at peak RA)
    profile_dec = data_map[:, peak_ra_idx]
    coords_dec = dec_center if len(dec_center) == data_map.shape[0] else np.arange(len(profile_dec))

    # Calculate FWHM for each direction
    if method == 'parabolic':
        result_ra = estimate_fwhm_parabolic(profile_ra, coords_ra)
        result_dec = estimate_fwhm_parabolic(profile_dec, coords_dec)
    elif method == 'gaussian':
        result_ra = estimate_fwhm_gaussian_fit(profile_ra, coords_ra)
        result_dec = estimate_fwhm_gaussian_fit(profile_dec, coords_dec)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Radial profile
    result_radial = estimate_fwhm_radial_profile(data_map, peak_idx, 
                                                coords_ra, coords_dec, method)

    return {
        'peak_value': peak_value,
        'peak_ra': peak_ra_idx,
        'peak_dec': peak_dec_idx,
        'fwhm_ra': result_ra.get('fwhm'),
        'fwhm_dec': result_dec.get('fwhm'),
        'fwhm_radial': result_radial.get('fwhm'),
        'results_ra': result_ra,
        'results_dec': result_dec,
        'results_radial': result_radial,
    }


def estimate_fwhm_radial_profile(data_map, peak_idx, ra_center, dec_center, method='parabolic'):
    """Estimate FWHM from a radial profile centered on the source peak.

    Parameters
    ----------
    data_map : ndarray
        2D intensity map
    peak_idx : tuple
        (dec_idx, ra_idx) of the peak
    ra_center : ndarray
        RA coordinates (degrees)
    dec_center : ndarray
        Dec coordinates (degrees)
    method : str
        'parabolic' or 'gaussian'

    Returns
    -------
    dict
        FWHM estimation result
    """
    peak_dec_idx, peak_ra_idx = peak_idx
    ny, nx = data_map.shape

    # Create radial distance grid
    if len(ra_center) == nx and len(dec_center) == ny:
        ra_grid, dec_grid = np.meshgrid(ra_center, dec_center)
    else:
        ra_grid, dec_grid = np.meshgrid(np.arange(nx), np.arange(ny))

    peak_ra = ra_grid[peak_dec_idx, peak_ra_idx]
    peak_dec = dec_grid[peak_dec_idx, peak_ra_idx]

    # Calculate angular distance from peak (in degrees)
    # Use simple Euclidean distance (good approximation for small fields)
    delta_ra = (ra_grid - peak_ra) * np.cos(np.radians(peak_dec))
    delta_dec = dec_grid - peak_dec
    distance = np.sqrt(delta_ra ** 2 + delta_dec ** 2) * 60  # Convert to arcmin

    # Compute radial profile by averaging in annuli
    r_max = np.max(distance)
    r_bins = np.linspace(0, r_max, max(nx, ny) // 2)
    radial_profile = np.zeros(len(r_bins) - 1)

    for i in range(len(r_bins) - 1):
        mask = (distance >= r_bins[i]) & (distance < r_bins[i + 1])
        if np.sum(mask) > 0:
            radial_profile[i] = np.mean(data_map[mask])
        else:
            radial_profile[i] = 0

    r_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Estimate FWHM from radial profile
    if method == 'parabolic':
        return estimate_fwhm_parabolic(radial_profile, r_centers)
    elif method == 'gaussian':
        return estimate_fwhm_gaussian_fit(radial_profile, r_centers)
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_fwhm_table(profiles_dict, freqs=None, method='parabolic'):
    """Estimate FWHM for multiple profiles.

    Parameters
    ----------
    profiles_dict : dict
        Dictionary with profile names as keys and profile arrays as values.
    freqs : ndarray, optional
        Frequency/distance array.
    method : str
        Method to use: 'parabolic' or 'gaussian'

    Returns
    -------
    dict
        Dictionary with profile names as keys and FWHM estimates as values.
    """
    results = {}
    
    if method == 'parabolic':
        func = estimate_fwhm_parabolic
    elif method == 'gaussian':
        func = estimate_fwhm_gaussian_fit
    else:
        raise ValueError(f"Unknown method: {method}")

    for name, profile in profiles_dict.items():
        results[name] = func(profile, freqs)

    return results


def print_fwhm_results(result, label=''):
    """Print FWHM estimation results nicely.

    Parameters
    ----------
    result : dict
        Result dictionary from FWHM estimation function.
    label : str
        Label to print with results.
    """
    if 'error' in result:
        print(f"  {label}: ERROR - {result['error']}")
        return

    print(f"  {label}:")
    if 'peak_value' in result:
        print(f"    Peak intensity: {result['peak_value']:.2f}")
    if 'peak_freq' in result:
        print(f"    Peak frequency: {result['peak_freq']:.2f}")
    if 'fwhm' in result and result['fwhm'] is not None:
        print(f"    FWHM: {result['fwhm']:.4f}")
        if 'fwhm_freq' in result:
            print(f"    FWHM (freq units): {result['fwhm_freq']:.4f}")
    if 'fit_quality' in result:
        print(f"    Fit quality (R²): {result['fit_quality']:.4f}")
    if 'sigma' in result:
        print(f"    Sigma: {result['sigma']:.4f}")
