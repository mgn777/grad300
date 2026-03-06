"""Spectral processing helpers for GRAD-300."""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def process_spectrum(info, basefreq, bandwidth, nchan, rfi_timestamps=None):
    """Generate a calibration plot for a spectrum FITS file.

    Parameters
    ----------
    info : dict
        Metadata dictionary with keys 'path', 'target', 'time'.
    basefreq : float
        Center frequency of the band (Hz).
    bandwidth : float
        Total bandwidth (Hz).
    nchan : int
        Number of frequency channels.
    rfi_timestamps : dict, optional
        Dictionary with BBC column names as keys and JD timestamp arrays
        where RFI was detected as values.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the plotted spectrum, or ``None`` if the file
        could not be processed due to missing columns.
    """
    with fits.open(info['path'], ignore_missing_end=True) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data

        # quick sanity check
        if 'LEFT_POL' not in data.dtype.names or 'STATUS' not in data.dtype.names:
            print(f"  ⚠️  {os.path.basename(info['path'])}: colonnes manquantes")
            return None

        on_rows = data[data['STATUS'] == 'on']
        off_rows = data[data['STATUS'] == 'off']

        if len(on_rows) == 0 or len(off_rows) == 0:
            print(f"  ⚠️  {os.path.basename(info['path'])}: pas de ON/OFF")
            return None

        on = on_rows[0]
        off = off_rows[0]

        spec_on = on['LEFT_POL'] + on['RIGHT_POL']
        spec_off = off['LEFT_POL'] + off['RIGHT_POL']
        spec_cal = spec_on - spec_off

        freq = basefreq + np.linspace(0, bandwidth, nchan)

        fig = plt.figure(figsize=(12, 5))
        plt.plot(freq/1e6, spec_off, label='OFF', alpha=0.7)
        plt.plot(freq/1e6, spec_cal, label='ON-OFF', linewidth=2)
        plt.xlabel("Fréquence (MHz)")
        plt.ylabel("Signal (ADU)")
        plt.title(f"{info['target']} - {info['time']}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mark RFI frequencies if provided
        if rfi_timestamps:
            # Assume BBC channels are evenly distributed across the frequency band
            bbc_cols = list(rfi_timestamps.keys())
            if bbc_cols:
                n_bbc = len(bbc_cols)
                freq_per_bbc = bandwidth / n_bbc

                for i, bbc_col in enumerate(bbc_cols):
                    if bbc_col in rfi_timestamps and len(rfi_timestamps[bbc_col]) > 0:
                        # Calculate frequency range for this BBC
                        bbc_start_freq = basefreq + i * freq_per_bbc
                        bbc_end_freq = basefreq + (i + 1) * freq_per_bbc

                        # Mark the frequency range with a vertical span
                        plt.axvspan(bbc_start_freq/1e6, bbc_end_freq/1e6,
                                  alpha=0.3, color='red',
                                  label=f'RFI {bbc_col}' if i == 0 else "")

                        # Add text annotation
                        mid_freq = (bbc_start_freq + bbc_end_freq) / 2 / 1e6
                        plt.text(mid_freq, plt.ylim()[1] * 0.95,
                               f'{bbc_col}\n{len(rfi_timestamps[bbc_col])} pts',
                               ha='center', va='top', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='red', alpha=0.7))

        return fig


def create_rfi_highlighted_spectrum(info, basefreq, bandwidth, nchan, rfi_timestamps=None, output_path=None):
    """Create a spectrum plot with RFI frequencies highlighted.

    Parameters
    ----------
    info : dict
        Metadata dictionary with keys 'path', 'target', 'time'.
    basefreq : float
        Center frequency of the band (Hz).
    bandwidth : float
        Total bandwidth (Hz).
    nchan : int
        Number of frequency channels.
    rfi_timestamps : dict, optional
        Dictionary with BBC column names as keys and JD timestamp arrays
        where RFI was detected as values.
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the spectrum with RFI highlights, or None if failed.
    """
    try:
        with fits.open(info['path'], ignore_missing_end=True) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data

            if 'LEFT_POL' not in data.dtype.names or 'STATUS' not in data.dtype.names:
                print(f"  ⚠️  {os.path.basename(info['path'])}: colonnes manquantes")
                return None

            on_rows = data[data['STATUS'] == 'on']
            off_rows = data[data['STATUS'] == 'off']

            if len(on_rows) == 0 or len(off_rows) == 0:
                print(f"  ⚠️  {os.path.basename(info['path'])}: pas de ON/OFF")
                return None

            on = on_rows[0]
            off = off_rows[0]

            spec_on = on['LEFT_POL'] + on['RIGHT_POL']
            spec_off = off['LEFT_POL'] + off['RIGHT_POL']
            spec_cal = spec_on - spec_off

        freq = basefreq + np.linspace(0, bandwidth, nchan)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Main spectrum plot
        ax1.plot(freq/1e6, spec_off, label='OFF', alpha=0.7, color='blue')
        ax1.plot(freq/1e6, spec_cal, label='ON-OFF', linewidth=2, color='black')
        ax1.set_xlabel("Fréquence (MHz)")
        ax1.set_ylabel("Signal (ADU)")
        ax1.set_title(f"Spectre {info['target']} - {info['time']}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # RFI frequency bands
        rfi_bbc_count = 0
        if rfi_timestamps and len(rfi_timestamps) > 0:
            # Assume BBC channels are evenly distributed across the frequency band
            bbc_cols = list(rfi_timestamps.keys())
            if bbc_cols:
                n_bbc = len(bbc_cols)
                freq_per_bbc = bandwidth / n_bbc

                for i, bbc_col in enumerate(sorted(bbc_cols)):
                    if bbc_col in rfi_timestamps and len(rfi_timestamps[bbc_col]) > 0:
                        # Calculate frequency range for this BBC
                        bbc_start_freq = basefreq + i * freq_per_bbc
                        bbc_end_freq = basefreq + (i + 1) * freq_per_bbc

                        # Mark the frequency range with a vertical span
                        ax1.axvspan(bbc_start_freq/1e6, bbc_end_freq/1e6,
                                  alpha=0.3, color='red',
                                  label=f'RFI {bbc_col}' if rfi_bbc_count == 0 else "")

                        # Add text annotation
                        mid_freq = (bbc_start_freq + bbc_end_freq) / 2 / 1e6
                        ax1.text(mid_freq, ax1.get_ylim()[1] * 0.95,
                               f'{bbc_col}\n{len(rfi_timestamps[bbc_col])} pts',
                               ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='red', alpha=0.8))
                        rfi_bbc_count += 1

        # RFI statistics plot
        if rfi_timestamps and len(rfi_timestamps) > 0:
            bbc_names = []
            rfi_counts = []
            for bbc_col in sorted(rfi_timestamps.keys()):
                if len(rfi_timestamps[bbc_col]) > 0:
                    bbc_names.append(bbc_col)
                    rfi_counts.append(len(rfi_timestamps[bbc_col]))

            if rfi_counts:
                bars = ax2.bar(range(len(bbc_names)), rfi_counts, color='red', alpha=0.7)
                ax2.set_xticks(range(len(bbc_names)))
                ax2.set_xticklabels(bbc_names, rotation=45, ha='right')
                ax2.set_ylabel("Nombre de points RFI")
                ax2.set_title("Statistiques RFI par BBC")
                ax2.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, count in zip(bars, rfi_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Aucun RFI détecté", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Statistiques RFI")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Spectre RFI sauvegardé: {output_path}")

        return fig

    except Exception as e:
        print(f"  ❌ Erreur création spectre RFI: {e}")
        return None


def _detect_rfi_in_array(data, method='sigma', threshold=3.0):
    """Detect RFI in a 1D array using specified method."""
    if len(data) < 10:
        return []

    rfi_indices = []

    if method == 'sigma':
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        sigma = 1.4826 * mad if mad > 0 else np.std(data)

        if sigma > 0:
            deviations = np.abs(data - median_val)
            rfi_mask = deviations > threshold * sigma
            rfi_indices = np.where(rfi_mask)[0].tolist()

    elif method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        rfi_mask = (data < lower_bound) | (data > upper_bound)
        rfi_indices = np.where(rfi_mask)[0].tolist()

    return rfi_indices


def detect_rfi_in_spectrum(info, basefreq, bandwidth, nchan, method='sigma', threshold=3.0, n_bbc=8):
    """Detect RFI directly in spectrum data (OFF and ON-OFF).

    Parameters
    ----------
    info : dict
        Metadata dictionary with keys 'path', 'target', 'time'.
    basefreq : float
        Center frequency of the band (Hz).
    bandwidth : float
        Total bandwidth (Hz).
    nchan : int
        Number of frequency channels.
    method : str
        Detection method: 'sigma' (sigma clipping), 'iqr', 'mad_relative'
    threshold : float
        Detection threshold
    n_bbc : int
        Number of BBC channels (default: 8)

    Returns
    -------
    dict
        Dictionary with BBC info. Keys are BBC column names, values are dicts with:
        - 'status': 'bad' if >90% channels affected, 'rfi' if 10-90% affected
        - 'channels': list of affected channel indices (empty if 'bad')
        - 'frequencies': list of absolute frequency indices
        - 'count': number of affected channels
        - 'percentage': percentage of affected channels
    """
    try:
        with fits.open(info['path'], ignore_missing_end=True) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data

            if 'LEFT_POL' not in data.dtype.names or 'STATUS' not in data.dtype.names:
                return {}

            on_rows = data[data['STATUS'] == 'on']
            off_rows = data[data['STATUS'] == 'off']

            if len(on_rows) == 0 or len(off_rows) == 0:
                return {}

            on = on_rows[0]
            off = off_rows[0]

            spec_on = on['LEFT_POL'] + on['RIGHT_POL']
            spec_off = off['LEFT_POL'] + off['RIGHT_POL']
            spec_cal = spec_on - spec_off

        # Assume BBC channels are evenly distributed across the frequency band
        chan_per_bbc = nchan // n_bbc

        rfi_detected = {}
        bad_bbc_threshold = 0.90  # >90% = bad BBC, not localized RFI
        min_rfi_threshold = 0.05  # <5% = ignore (noise)

        for i in range(n_bbc):
            bbc_col = f'BBC{i+1:02d}u'  # BBC01u, BBC02u, etc.
            bbc_start_chan = i * chan_per_bbc
            bbc_end_chan = min((i + 1) * chan_per_bbc, nchan)
            n_channels = bbc_end_chan - bbc_start_chan

            # Extract data for this BBC band
            spec_off_bbc = spec_off[bbc_start_chan:bbc_end_chan]
            spec_cal_bbc = spec_cal[bbc_start_chan:bbc_end_chan]

            # Detect RFI in OFF spectrum
            rfi_off = _detect_rfi_in_array(spec_off_bbc, method, threshold)
            # Detect RFI in ON-OFF spectrum
            rfi_cal = _detect_rfi_in_array(spec_cal_bbc, method, threshold)

            # Combine RFI detections
            rfi_channels = set(rfi_off) | set(rfi_cal)
            rfi_percentage = len(rfi_channels) / n_channels if n_channels > 0 else 0

            # Only track significant RFI (>min_rfi_threshold)
            if rfi_percentage >= min_rfi_threshold:
                # Determine if this is a bad BBC or localized RFI
                if rfi_percentage >= bad_bbc_threshold:
                    # BBC is bad (>90% channels affected)
                    status = 'bad'
                    channels_list = []  # Don't list individual channels for bad BBCs
                else:
                    # Localized RFI spikes
                    status = 'rfi'
                    channels_list = list(rfi_channels)

                freq_indices = [bbc_start_chan + ch for ch in rfi_channels]
                rfi_values_off = spec_off[freq_indices] if freq_indices else []
                rfi_values_cal = spec_cal[freq_indices] if freq_indices else []

                rfi_detected[bbc_col] = {
                    'status': status,
                    'channels': channels_list,
                    'frequencies': freq_indices,
                    'values_off': rfi_values_off,
                    'values_cal': rfi_values_cal,
                    'count': len(rfi_channels),
                    'percentage': rfi_percentage * 100
                }
        return rfi_detected

    except Exception as e:
        print(f"  ❌ Erreur détection RFI spectre: {e}")
        return {}


def create_rfi_point_highlight_spectrum(info, basefreq, bandwidth, nchan, rfi_timestamps=None, output_path=None):
    """Create a spectrum plot with RFI points highlighted (not removed).

    Parameters
    ----------
    info : dict
        Metadata dictionary with keys 'path', 'target', 'time'.
    basefreq : float
        Center frequency of the band (Hz).
    bandwidth : float
        Total bandwidth (Hz).
    nchan : int
        Number of frequency channels.
    rfi_timestamps : dict, optional
        Dictionary with BBC column names as keys and JD timestamp arrays
        where RFI was detected as values.
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the spectrum with RFI points highlighted, or None if failed.
    """
    try:
        with fits.open(info['path'], ignore_missing_end=True) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data

            if 'LEFT_POL' not in data.dtype.names or 'STATUS' not in data.dtype.names:
                print(f"  ⚠️  {os.path.basename(info['path'])}: colonnes manquantes")
                return None

            on_rows = data[data['STATUS'] == 'on']
            off_rows = data[data['STATUS'] == 'off']

            if len(on_rows) == 0 or len(off_rows) == 0:
                print(f"  ⚠️  {os.path.basename(info['path'])}: pas de ON/OFF")
                return None

            on = on_rows[0]
            off = off_rows[0]

            spec_on = on['LEFT_POL'] + on['RIGHT_POL']
            spec_off = off['LEFT_POL'] + off['RIGHT_POL']
            spec_cal = spec_on - spec_off

        freq = basefreq + np.linspace(0, bandwidth, nchan)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Main spectrum plot
        ax1.plot(freq/1e6, spec_off, label='OFF', alpha=0.7, color='blue')
        ax1.plot(freq/1e6, spec_cal, label='ON-OFF', linewidth=2, color='black')
        ax1.set_xlabel("Fréquence (MHz)")
        ax1.set_ylabel("Signal (ADU)")
        ax1.set_title(f"Spectre avec RFI mis en évidence {info['target']} - {info['time']}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark RFI points in spectrum based on BBC bands
        rfi_bbc_count = 0
        if rfi_timestamps and len(rfi_timestamps) > 0:
            # Assume BBC channels are evenly distributed across the frequency band
            bbc_cols = list(rfi_timestamps.keys())
            if bbc_cols:
                n_bbc = len(bbc_cols)
                freq_per_bbc = bandwidth / n_bbc
                chan_per_bbc = nchan // n_bbc

                for i, bbc_col in enumerate(sorted(bbc_cols)):
                    if bbc_col in rfi_timestamps and len(rfi_timestamps[bbc_col]) > 0:
                        # Calculate frequency and channel range for this BBC
                        bbc_start_freq = basefreq + i * freq_per_bbc
                        bbc_end_freq = basefreq + (i + 1) * freq_per_bbc
                        bbc_start_chan = i * chan_per_bbc
                        bbc_end_chan = min((i + 1) * chan_per_bbc, nchan)

                        # Mark the frequency range with a vertical span
                        ax1.axvspan(bbc_start_freq/1e6, bbc_end_freq/1e6,
                                  alpha=0.2, color='red',
                                  label=f'RFI {bbc_col}' if rfi_bbc_count == 0 else "")

                        # Highlight the actual RFI-affected channels with markers
                        freq_range = freq[bbc_start_chan:bbc_end_chan]
                        spec_off_range = spec_off[bbc_start_chan:bbc_end_chan]
                        spec_cal_range = spec_cal[bbc_start_chan:bbc_end_chan]

                        # Plot markers on RFI-affected channels
                        ax1.scatter(freq_range/1e6, spec_off_range,
                                  color='red', s=20, alpha=0.8, marker='x',
                                  label=f'RFI OFF {bbc_col}' if rfi_bbc_count == 0 else "")
                        ax1.scatter(freq_range/1e6, spec_cal_range,
                                  color='darkred', s=20, alpha=0.8, marker='x',
                                  label=f'RFI ON-OFF {bbc_col}' if rfi_bbc_count == 0 else "")

                        # Add text annotation
                        mid_freq = (bbc_start_freq + bbc_end_freq) / 2 / 1e6
                        ax1.text(mid_freq, ax1.get_ylim()[1] * 0.95,
                               f'{bbc_col}\n{len(rfi_timestamps[bbc_col])} pts',
                               ha='center', va='top', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='red', alpha=0.8))
                        rfi_bbc_count += 1

        # Statistics plot
        if rfi_timestamps and len(rfi_timestamps) > 0:
            bbc_names = []
            rfi_counts = []
            for bbc_col in sorted(rfi_timestamps.keys()):
                if len(rfi_timestamps[bbc_col]) > 0:
                    bbc_names.append(bbc_col)
                    rfi_counts.append(len(rfi_timestamps[bbc_col]))

            if rfi_counts:
                bars = ax2.bar(range(len(bbc_names)), rfi_counts, color='red', alpha=0.7)
                ax2.set_xticks(range(len(bbc_names)))
                ax2.set_xticklabels(bbc_names, rotation=45, ha='right')
                ax2.set_ylabel("Nombre de points RFI")
                ax2.set_title(f"Points RFI détectés: {rfi_bbc_count} BBC(s)")
                ax2.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, count in zip(bars, rfi_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, "Aucun RFI détecté", ha='center', va='center',
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title("Aucun RFI détecté")
        else:
            ax2.text(0.5, 0.5, "Aucun RFI détecté", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Spectre non modifié")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Spectre RFI mis en évidence sauvegardé: {output_path}")

        return fig

    except Exception as e:
        print(f"  ❌ Erreur création spectre RFI mis en évidence: {e}")
        return None


def create_clean_spectrum(info, basefreq, bandwidth, nchan, rfi_timestamps=None, output_path=None, n_bbc=8):
    """Create a spectrum plot with RFI points highlighted from both TPI and spectrum analysis.

    Parameters
    ----------
    info : dict
        Metadata dictionary with keys 'path', 'target', 'time'.
    basefreq : float
        Center frequency of the band (Hz).
    bandwidth : float
        Total bandwidth (Hz).
    nchan : int
        Number of frequency channels.
    rfi_timestamps : dict, optional
        Dictionary with BBC column names as keys and JD timestamp arrays
        where RFI was detected in TPI as values.
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.
    n_bbc : int
        Number of BBC channels (default: 8)

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the spectrum with RFI points highlighted, or None if failed.
    """
    try:
        with fits.open(info['path'], ignore_missing_end=True) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data

            if 'LEFT_POL' not in data.dtype.names or 'STATUS' not in data.dtype.names:
                print(f"  ⚠️  {os.path.basename(info['path'])}: colonnes manquantes")
                return None

            on_rows = data[data['STATUS'] == 'on']
            off_rows = data[data['STATUS'] == 'off']

            if len(on_rows) == 0 or len(off_rows) == 0:
                print(f"  ⚠️  {os.path.basename(info['path'])}: pas de ON/OFF")
                return None

            on = on_rows[0]
            off = off_rows[0]

            spec_on = on['LEFT_POL'] + on['RIGHT_POL']
            spec_off = off['LEFT_POL'] + off['RIGHT_POL']
            spec_cal = spec_on - spec_off

        freq = basefreq + np.linspace(0, bandwidth, nchan)

        # NOTE: We do NOT detect RFI independently in spectrum because without knowing the true
        # BBC frequency distribution in the 1024 channels, we cannot reliably map spectrum 
        # channels to BBCs. Instead, we rely on RFI detection from TPI data (where BBC columns
        # are explicitly labeled) and visualize those BBCs in the spectrum.

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Main spectrum plot
        ax1.plot(freq/1e6, spec_off, label='OFF', alpha=0.7, color='blue')
        ax1.plot(freq/1e6, spec_cal, label='ON-OFF', linewidth=2, color='black')
        ax1.set_xlabel("Fréquence (MHz)")
        ax1.set_ylabel("Signal (ADU)")
        ax1.set_title(f"Spectre avec RFI mis en évidence {info['target']} - {info['time']}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark RFI points from spectrum analysis
        rfi_bbc_count = 0
        all_rfi_bbc = set()

        # First, mark RFI from TPI analysis (BBC bands)
        if rfi_timestamps and len(rfi_timestamps) > 0:
            bbc_cols = list(rfi_timestamps.keys())
            if bbc_cols:
                freq_per_bbc = bandwidth / n_bbc
                chan_per_bbc = nchan // n_bbc

                for i, bbc_col in enumerate(sorted(bbc_cols)):
                    if bbc_col in rfi_timestamps and len(rfi_timestamps[bbc_col]) > 0:
                        bbc_start_freq = basefreq + i * freq_per_bbc
                        bbc_end_freq = basefreq + (i + 1) * freq_per_bbc
                        bbc_start_chan = i * chan_per_bbc
                        bbc_end_chan = min((i + 1) * chan_per_bbc, nchan)

                        # Mark the frequency range with a vertical span
                        ax1.axvspan(bbc_start_freq/1e6, bbc_end_freq/1e6,
                                  alpha=0.2, color='red',
                                  label=f'RFI TPI {bbc_col}' if rfi_bbc_count == 0 else "")

                        # Add text annotation
                        mid_freq = (bbc_start_freq + bbc_end_freq) / 2 / 1e6
                        ax1.text(mid_freq, ax1.get_ylim()[1] * 0.95,
                               f'{bbc_col}\nTPI: {len(rfi_timestamps[bbc_col])} pts',
                               ha='center', va='top', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='red', alpha=0.8))
                        all_rfi_bbc.add(bbc_col)
                        rfi_bbc_count += 1

        # Statistics plot based on TPI detections only
        if rfi_timestamps and all_rfi_bbc:
            bbc_names = []
            rfi_counts = []
            for bbc_col in sorted(all_rfi_bbc):
                # Count from TPI only
                tpi_count = len(rfi_timestamps.get(bbc_col, [])) if rfi_timestamps else 0
                bbc_names.append(bbc_col)
                rfi_counts.append(tpi_count)

            if rfi_counts:
                # Color bars: red for RFI detected
                colors = ['red' for _ in rfi_counts]
                bars = ax2.bar(range(len(bbc_names)), rfi_counts, color=colors, alpha=0.7)
                ax2.set_xticks(range(len(bbc_names)))
                ax2.set_xticklabels(bbc_names, rotation=45, ha='right')
                ax2.set_ylabel("Nombre de points RFI (TPI)")
                ax2.set_title(f"RFI détecté dans {len(bbc_names)} BBC(s) (détection TPI)")
                ax2.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, count in zip(bars, rfi_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, "Aucun RFI détecté dans TPI", ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Aucun RFI détecté dans TPI")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Spectre RFI mis en évidence sauvegardé: {output_path}")

        return fig

    except Exception as e:
        print(f"  ❌ Erreur création spectre RFI mis en évidence: {e}")
        return None