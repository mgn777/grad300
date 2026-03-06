"""Time-position-intensity processing routines."""

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from scipy.signal import savgol_filter
import astropy.units as u


def sigma_clip_mask_robust(data, k=3):
    """Return boolean mask of values within k*sigma based on MAD."""
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    sigma = 1.4826 * mad if mad > 0 else 1.0  # Avoid division by zero
    return np.abs(data - median) < k * sigma


def process_tpi(info, location, window=7, polyorder=2, start_ignore=1, end_ignore=0):
    """Load a TPI file, clean it and compute ra/dec plus derivatives.

    Parameters
    ----------
    info : dict
        Metadata dict with keys 'path', 'target', 'time'.
    location : astropy.coordinates.EarthLocation
        Location of the telescope for coordinate transformation.
    window : int
        Window length for Savitzky-Golay filter.
    polyorder : int
        Polynomial order for the filter.
    start_ignore, end_ignore : float
        Seconds at beginning/end to ignore for temporal trimming.

    Returns
    -------
    dict or None
        Dictionary containing raw/clean data, ra/dec, velocities, etc., or
        ``None`` if the file cannot be processed.
    """
    from astropy.io import fits

    try:
        with fits.open(info['path']) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data

            if len(data) == 0:
                print("  ⚠️  Fichier vide")
                return None

            required = ['JD', 'Azimuth', 'Elevation']
            for col in required:
                if col not in data.dtype.names:
                    print(f"  ⚠️  Colonne {col} manquante")
                    return None

            jd = data['JD']
            az = data['Azimuth']
            el = data['Elevation']

            if len(jd) == 0:
                print("  ⚠️  Pas de données JD")
                return None

            # Time trimming
            t = (jd - jd[0]) * 86400
            mask_time = (t > start_ignore) & (t < (t[-1] - end_ignore))
            
            if not np.any(mask_time):
                print("  ⚠️  Aucune donnée après trimming temporel")
                return None
                
            jd = jd[mask_time]
            az = az[mask_time]
            el = el[mask_time]
            t = (jd - jd[0]) * 86400

            # Convert to RA/Dec
            time = Time(jd, format='jd')
            altaz = AltAz(az=az*u.deg, alt=el*u.deg,
                          location=location, obstime=time)
            sky = SkyCoord(altaz)
            ra_raw, dec_raw = sky.icrs.ra.deg, sky.icrs.dec.deg

            # First sigma clip on RA/Dec
            ra_mask = sigma_clip_mask_robust(ra_raw)
            dec_mask = sigma_clip_mask_robust(dec_raw)
            mask_comb = ra_mask & dec_mask

            if np.sum(mask_comb) < window:
                print(f"  ⚠️  Pas assez de points après clipping (minimum {window} requis)")
                return {
                    'raw': data,
                    'clean': data[mask_time],
                    'ra': ra_raw,
                    'dec': dec_raw,
                    't': t,
                    'jd': jd,
                    'az': az,
                    'el': el
                }

            # Calculate velocities and accelerations using Savitzky-Golay
            dt = np.median(np.diff(t[mask_comb]))
            
            # Ensure dt is positive and reasonable
            if dt <= 0 or np.isnan(dt):
                dt = 1.0
            
            vit_ra = savgol_filter(ra_raw[mask_comb], window, polyorder, deriv=1, delta=dt)
            vit_dec = savgol_filter(dec_raw[mask_comb], window, polyorder, deriv=1, delta=dt)

            acc_ra = savgol_filter(ra_raw[mask_comb], window, polyorder, deriv=2, delta=dt)
            acc_dec = savgol_filter(dec_raw[mask_comb], window, polyorder, deriv=2, delta=dt)

            v_norm = np.sqrt(vit_ra**2 + vit_dec**2)
            a_norm = np.sqrt(acc_ra**2 + acc_dec**2)

            # Second sigma clip on velocities and accelerations
            mask_vel = sigma_clip_mask_robust(v_norm, k=3)
            mask_acc = sigma_clip_mask_robust(a_norm, k=3)
            mask_va = mask_vel & mask_acc

            # Apply all masks
            final_mask = np.zeros(len(ra_raw), dtype=bool)
            final_mask[np.where(mask_comb)[0][mask_va]] = True

            jd_clean = jd[final_mask]
            az_clean = az[final_mask]
            el_clean = el[final_mask]
            t_clean = (jd_clean - jd_clean[0]) * 86400 if len(jd_clean) > 0 else np.array([])

            # Recompute RA/Dec for cleaned data
            if len(jd_clean) > 0:
                time_clean = Time(jd_clean, format='jd')
                altaz_clean = AltAz(az=az_clean*u.deg, alt=el_clean*u.deg,
                                    location=location, obstime=time_clean)
                sky_clean = SkyCoord(altaz_clean)
                ra_clean, dec_clean = sky_clean.icrs.ra.deg, sky_clean.icrs.dec.deg
            else:
                ra_clean, dec_clean = np.array([]), np.array([])

            return {
                'raw': data,
                'clean': data[mask_time][mask_comb][mask_va] if len(jd_clean) > 0 else np.array([]),
                'ra': ra_clean,
                'dec': dec_clean,
                'ra_raw': ra_raw,
                'dec_raw': dec_raw,
                'vit_ra': vit_ra[mask_va] if len(vit_ra) > 0 else np.array([]),
                'vit_dec': vit_dec[mask_va] if len(vit_dec) > 0 else np.array([]),
                'acc_ra': acc_ra[mask_va] if len(acc_ra) > 0 else np.array([]),
                'acc_dec': acc_dec[mask_va] if len(acc_dec) > 0 else np.array([]),
                'v_norm': v_norm[mask_va] if len(v_norm) > 0 else np.array([]),
                'a_norm': a_norm[mask_va] if len(a_norm) > 0 else np.array([]),
                't': t_clean,
                'jd': jd_clean,
                'az': az_clean,
                'el': el_clean,
                'mask_time': mask_time,
                'mask_comb': mask_comb,
                'mask_va': mask_va
            }

    except Exception as e:
        print(f"  ❌ Erreur traitement TPI: {e}")
        return None


def detect_rfi_in_tpi(result, method='variance', threshold=3.0):
    """Detect RFI in TPI data using variance-based analysis.

    Parameters
    ----------
    result : dict
        TPI processing result from process_tpi()
    method : str
        Detection method: 'variance' (variance ratio), 'iqr', 'sigma', 
        'temporal', 'mad_relative'
    threshold : float
        Detection threshold (interpretation depends on method)

    Returns
    -------
    dict
        Dictionary with BBC column names as keys and dict of RFI info as values.
    """
    rfi_detected = {}
    
    # Check if result is valid
    if result is None or 'clean' not in result or len(result['clean']) == 0:
        return rfi_detected

    # Get BBC columns
    bbc_cols = [c for c in result['clean'].dtype.names if 'BBC' in c]

    if not bbc_cols:
        return rfi_detected

    jd_times = result['clean']['JD']

    if method == 'variance':
        # Variance-based detection: find BBCs with unusually high variance
        bbc_variances = {}
        bbc_data_dict = {}

        for bbc_col in bbc_cols:
            bbc_data = result['clean'][bbc_col]
            if len(bbc_data) < 10:
                continue
            # Use robust variance estimate (median absolute deviation squared)
            median_val = np.median(bbc_data)
            mad = np.median(np.abs(bbc_data - median_val))
            robust_variance = mad ** 2 if mad > 0 else 0  # MAD² as robust variance estimate
            bbc_variances[bbc_col] = robust_variance
            bbc_data_dict[bbc_col] = bbc_data

        if not bbc_variances:
            return rfi_detected

        # Calculate median variance across all BBCs
        all_variances = list(bbc_variances.values())
        median_variance = np.median(all_variances) if all_variances else 0

        # Find BBCs with variance significantly higher than median
        for bbc_col, variance in bbc_variances.items():
            variance_ratio = variance / median_variance if median_variance > 0 else 1.0

            if variance_ratio > threshold:
                # This BBC has unusually high variance - mark all its points as RFI
                bbc_data = bbc_data_dict[bbc_col]
                rfi_timestamps = jd_times
                rfi_values = bbc_data
                severity_scores = np.full(len(bbc_data), variance_ratio)

                rfi_detected[bbc_col] = {
                    'timestamps': rfi_timestamps,
                    'values': rfi_values,
                    'severity': severity_scores,
                    'count': len(rfi_timestamps),
                    'percentage': 100.0,  # All points in high-variance BBC
                    'avg_severity': variance_ratio,
                    'variance_ratio': variance_ratio,
                    'method': 'variance'
                }

    elif method == 'mad_relative':
        # MAD on relative deviations from continuum
        # Calculate continuum as median across all BBCs for each time point
        bbc_data_matrix = np.array([result['clean'][bbc_col] for bbc_col in bbc_cols])
        continuum = np.median(bbc_data_matrix, axis=0)  # Median across BBCs for each time point

        # Calculate relative deviations for each BBC and time point
        relative_deviations = {}
        for i, bbc_col in enumerate(bbc_cols):
            bbc_data = result['clean'][bbc_col]
            # Avoid division by zero by using small epsilon for continuum values near zero
            safe_continuum = np.where(np.abs(continuum) < 1e-10, 1e-10, continuum)
            rel_dev = np.abs(bbc_data - continuum) / np.abs(safe_continuum)
            relative_deviations[bbc_col] = rel_dev

        # For each BBC, find maximum relative deviation
        bbc_max_rel_dev = {}
        bbc_mean_rel_dev = {}
        for bbc_col, rel_dev in relative_deviations.items():
            bbc_max_rel_dev[bbc_col] = np.max(rel_dev) if len(rel_dev) > 0 else 0
            bbc_mean_rel_dev[bbc_col] = np.mean(rel_dev) if len(rel_dev) > 0 else 0

        # Use MAD to detect BBCs with abnormally high relative deviations
        all_max_devs = list(bbc_max_rel_dev.values())
        if all_max_devs:
            median_max_dev = np.median(all_max_devs)
            mad = np.median(np.abs(all_max_devs - median_max_dev)) if all_max_devs else 0

            for bbc_col in bbc_cols:
                max_dev = bbc_max_rel_dev.get(bbc_col, 0)
                mean_dev = bbc_mean_rel_dev.get(bbc_col, 0)

                # MAD score: how many MADs above the median
                mad_score = (max_dev - median_max_dev) / mad if mad > 0 else 0

                if mad_score > threshold:
                    # This BBC has abnormally high relative deviation - flag it
                    bbc_data = result['clean'][bbc_col]
                    rfi_timestamps = jd_times
                    rfi_values = bbc_data
                    severity_scores = np.full(len(bbc_data), mad_score)

                    rfi_detected[bbc_col] = {
                        'timestamps': rfi_timestamps,
                        'values': rfi_values,
                        'severity': severity_scores,
                        'count': len(rfi_timestamps),
                        'percentage': 100.0,  # Flag entire BBC
                        'avg_severity': mad_score,
                        'max_relative_deviation': max_dev,
                        'mean_relative_deviation': mean_dev,
                        'mad_score': mad_score,
                        'mad_score_threshold': threshold,
                        'method': 'mad_relative'
                    }

    elif method == 'temporal':
        # Temporal-based detection: find outliers in local windows
        for bbc_col in bbc_cols:
            bbc_data = result['clean'][bbc_col]
            
            if len(bbc_data) < 20:
                continue
                
            window_size = max(10, len(bbc_data) // 20)
            local_medians = []
            local_mads = []
            
            for i in range(len(bbc_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(bbc_data), i + window_size // 2)
                window_data = bbc_data[start_idx:end_idx]
                
                local_median = np.median(window_data)
                local_mad = np.median(np.abs(window_data - local_median))
                
                local_medians.append(local_median)
                local_mads.append(local_mad)

            local_medians = np.array(local_medians)
            local_mads = np.array(local_mads)
            
            # Use MAD for robust sigma estimation
            sigma_local = 1.4826 * local_mads
            sigma_local = np.where(sigma_local < 1e-10, 1e-10, sigma_local)
            
            deviations = np.abs(bbc_data - local_medians)
            outlier_mask = deviations > threshold * sigma_local
            
            if np.any(outlier_mask):
                rfi_timestamps = jd_times[outlier_mask]
                rfi_values = bbc_data[outlier_mask]
                severity_scores = deviations[outlier_mask] / sigma_local[outlier_mask]
                
                percentage = len(rfi_timestamps) / len(bbc_data) * 100
                
                # Only flag if significant RFI (>1% of points)
                if percentage > 1.0:
                    rfi_detected[bbc_col] = {
                        'timestamps': rfi_timestamps,
                        'values': rfi_values,
                        'severity': severity_scores,
                        'count': len(rfi_timestamps),
                        'percentage': percentage,
                        'avg_severity': np.mean(severity_scores) if len(severity_scores) > 0 else 0,
                        'method': 'temporal'
                    }

    else:
        # Fallback to other methods (sigma, iqr) for compatibility
        for bbc_col in bbc_cols:
            bbc_data = result['clean'][bbc_col]

            if len(bbc_data) < 10:
                continue

            rfi_timestamps = []
            rfi_values = []
            severity_scores = []

            if method == 'iqr':
                q1, q3 = np.percentile(bbc_data, [25, 75])
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    outlier_mask = (bbc_data < lower_bound) | (bbc_data > upper_bound)
                    if np.any(outlier_mask):
                        rfi_timestamps = jd_times[outlier_mask]
                        rfi_values = bbc_data[outlier_mask]
                        median_val = np.median(bbc_data)
                        severity_scores = np.abs(bbc_data[outlier_mask] - median_val) / iqr

            elif method == 'sigma':
                median_val = np.median(bbc_data)
                mad = np.median(np.abs(bbc_data - median_val))
                sigma = 1.4826 * mad if mad > 0 else np.std(bbc_data)
                if sigma > 0:
                    deviations = np.abs(bbc_data - median_val)
                    outlier_mask = deviations > threshold * sigma
                    if np.any(outlier_mask):
                        rfi_timestamps = jd_times[outlier_mask]
                        rfi_values = bbc_data[outlier_mask]
                        severity_scores = deviations[outlier_mask] / sigma

            if len(rfi_timestamps) > 0 and len(severity_scores) > 0:
                percentage = len(rfi_timestamps) / len(bbc_data) * 100
                avg_severity = np.mean(severity_scores)
                
                # Only flag if significant RFI (>1% of points) and severity > threshold
                if percentage > 1.0 and avg_severity > 2.0:
                    rfi_detected[bbc_col] = {
                        'timestamps': rfi_timestamps,
                        'values': rfi_values,
                        'severity': severity_scores,
                        'count': len(rfi_timestamps),
                        'percentage': percentage,
                        'avg_severity': avg_severity,
                        'method': method
                    }

    return rfi_detected


def flag_rfi_in_tpi(result, rfi_detected, flag_value=np.nan):
    """Flag RFI-affected data points in TPI result.

    Parameters
    ----------
    result : dict
        TPI processing result from process_tpi()
    rfi_detected : dict
        RFI detection dictionary from detect_rfi_in_tpi()
    flag_value : any
        Value to use for flagged data (default: np.nan)

    Returns
    -------
    dict
        Copy of result with RFI-flagged data
    """
    if result is None or 'clean' not in result or len(result['clean']) == 0:
        return result
        
    # Create a copy of the clean data array
    flagged_data = result['clean'].copy()
    
    # Get BBC columns
    bbc_cols = [c for c in result['clean'].dtype.names if 'BBC' in c]
    
    if not bbc_cols:
        return result
    
    rfi_counts = {}
    
    # Flag data based on RFI detection
    for bbc_col, rfi_info in rfi_detected.items():
        if bbc_col not in bbc_cols:
            continue
            
        # Get timestamps of RFI
        rfi_timestamps = rfi_info['timestamps']
        
        # Find indices in the clean data
        all_timestamps = result['clean']['JD']
        
        # For each RFI timestamp, find matching index
        flagged_indices = []
        for ts in rfi_timestamps:
            idx = np.where(np.abs(all_timestamps - ts) < 1e-6)[0]
            if len(idx) > 0:
                flagged_indices.append(idx[0])
        
        # Flag the data
        if flagged_indices:
            flagged_data[bbc_col][flagged_indices] = flag_value
            rfi_counts[bbc_col] = len(flagged_indices)
    
    # Create flagged result
    flagged_result = result.copy()
    flagged_result['clean_flagged'] = flagged_data
    flagged_result['rfi_flagged'] = rfi_counts
    
    return flagged_result


def combine_rfi_detections(rfi_list):
    """Combine multiple RFI detection dictionaries.

    Parameters
    ----------
    rfi_list : list
        List of RFI detection dictionaries

    Returns
    -------
    dict
        Combined RFI detection dictionary
    """
    combined = {}
    
    for rfi_dict in rfi_list:
        if not rfi_dict:
            continue
            
        for bbc_col, rfi_info in rfi_dict.items():
            if bbc_col not in combined:
                combined[bbc_col] = {
                    'timestamps': rfi_info['timestamps'].copy(),
                    'values': rfi_info['values'].copy(),
                    'severity': rfi_info['severity'].copy(),
                    'count': rfi_info['count'],
                    'percentage': rfi_info['percentage'],
                    'avg_severity': rfi_info['avg_severity'],
                    'methods': [rfi_info.get('method', 'unknown')]
                }
            else:
                # Combine timestamps (avoid duplicates)
                existing_ts = set(combined[bbc_col]['timestamps'])
                new_ts = set(rfi_info['timestamps'])
                
                # Find new timestamps
                all_ts = list(existing_ts | new_ts)
                all_ts.sort()
                
                # Rebuild arrays
                all_values = []
                all_severities = []
                
                for ts in all_ts:
                    if ts in existing_ts:
                        idx = np.where(combined[bbc_col]['timestamps'] == ts)[0][0]
                        all_values.append(combined[bbc_col]['values'][idx])
                        all_severities.append(combined[bbc_col]['severity'][idx])
                    else:
                        idx = np.where(rfi_info['timestamps'] == ts)[0][0]
                        all_values.append(rfi_info['values'][idx])
                        all_severities.append(rfi_info['severity'][idx])
                
                combined[bbc_col]['timestamps'] = np.array(all_ts)
                combined[bbc_col]['values'] = np.array(all_values)
                combined[bbc_col]['severity'] = np.array(all_severities)
                combined[bbc_col]['count'] = len(all_ts)
                combined[bbc_col]['percentage'] = combined[bbc_col]['count'] / len(all_ts) * 100
                combined[bbc_col]['avg_severity'] = np.mean(all_severities)
                combined[bbc_col]['methods'].append(rfi_info.get('method', 'unknown'))
    
    return combined