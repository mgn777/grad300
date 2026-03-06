"""I/O helpers for GRAD-300 pipeline."""

import os
import glob
import re
import numpy as np
from astropy.io import fits


def find_files(base_path, obs_date, file_pattern, target=None):
    """Searches the FITS directory of a given observation date and returns
    metadata grouped by file type (spectrum, tpi, image).

    Parameters
    ----------
    base_path : str
        Root directory containing observation subfolders.
    obs_date : str
        Observation date string (e.g. "20250204").
    file_pattern : str
        Regular expression used to parse filenames and extract type/target.
    target : str, optional
        If provided, only files matching this target are returned.
    """
    fits_dir = os.path.join(base_path, obs_date, "FITS")
    files = {'spectrum': [], 'tpi': [], 'image': []}

    for fpath in glob.glob(os.path.join(fits_dir, "*.fits")):
        match = re.match(file_pattern, os.path.basename(fpath))
        if match and (target is None or match.group(5) == target):
            ftype = match.group(3).lower()
            if ftype in files:
                files[ftype].append({
                    'path': fpath,
                    'time': match.group(2),
                    'target': match.group(5)
                })
    return files


def get_image_resolution(base_path, obs_date, target, time=None):
    """Return the useful (non-empty) resolution of a reference image.

    The routine inspects matching *_IMAGE-*-<target>_*.fits files and
    computes the shape of the non-zero/non-NaN region. If ``time`` is provided,
    looks for the image with the closest matching timestamp.
    It prints diagnostic information and returns a tuple (ny, nx) or ``None`` 
    if no file is found.
    """
    fits_dir = os.path.join(base_path, obs_date, "FITS")
    pattern = os.path.join(fits_dir, f"*_IMAGE-*-{target}_*.fits")

    image_files = sorted(glob.glob(pattern))
    if not image_files:
        return None

    # If time is provided, find the closest timestamp
    if time:
        try:
            tpi_time = int(time)
            # Extract timestamp from filename: '20250212-083216_IMAGE...' → '083216'
            closest_file = min(image_files, key=lambda f: abs(int(os.path.basename(f).split('-')[1].split('_')[0]) - tpi_time))
            image_file = closest_file
        except (ValueError, IndexError):
            image_file = image_files[0]
    else:
        image_file = image_files[0]

    try:
        with fits.open(image_file) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    data = hdu.data
                    shape_original = data.shape  # (ny, nx)
                    print(f"  Image originale: {shape_original[1]}x{shape_original[0]}")

                    # build mask of valid pixels
                    if np.any(np.isnan(data)):
                        valid_mask = ~np.isnan(data)
                    else:
                        valid_mask = data != 0

                    rows = np.where(np.any(valid_mask, axis=1))[0]
                    cols = np.where(np.any(valid_mask, axis=0))[0]

                    if len(rows) and len(cols):
                        y_min, y_max = rows[0], rows[-1]
                        x_min, x_max = cols[0], cols[-1]
                        useful_shape = (y_max - y_min + 1, x_max - x_min + 1)
                        # Use minimum dimension for both to ensure square grid
                        n = min(useful_shape)
                        print(
                            f"  Région utile détectée: {useful_shape[1]}x{useful_shape[0]} "
                            f"(pixels {x_min}-{x_max}, {y_min}-{y_max})"
                        )
                        print(
                            f"  Taux de remplissage: {np.sum(valid_mask)/valid_mask.size*100:.1f}%"
                        )
                        return (n, n)
                    else:
                        print("  ⚠️  Aucun pixel valide détecté dans l'image")
                        return shape_original
    except Exception as e:
        print(f"  ⚠️  Erreur lecture image: {e}")
    return None
