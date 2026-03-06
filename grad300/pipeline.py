"""High-level orchestration of the GRAD-300 processing pipeline."""

import os
import numpy as np
from . import io as io_mod
from . import spectrum as spec_mod
from . import tpi as tpi_mod
from . import plotting as plot_mod
from . import utils


class GradPipeline:
    def __init__(self, base_path="./GRAD-300/GRAD-300",
                 location=None,
                 basefreq=None, bandwidth=None, nchan=None,
                 file_pattern=None):
        self.base_path = base_path
        # defaults come from package __init__ if not provided
        from . import LOCATION, BASE_FREQ, BANDWIDTH, NCHAN
        self.location = location or LOCATION
        self.basefreq = basefreq or BASE_FREQ
        self.bandwidth = bandwidth or BANDWIDTH
        self.nchan = nchan or NCHAN
        self.file_pattern = file_pattern or r'(\d{8})-(\d{6})_([A-Z]+)-([A-Z0-9]+)-([A-Z0-9]+)_\d+#_\d+#\.fits'
        self.processed = []
        self.ignored = []
        self.rfi_detected = {}  # Store RFI timestamps by target

    def run(self, obs_date, target=None, window=7, polyorder=2,
            start_ignore=1, end_ignore=0):
        print(f"\n=== GRAD-300 {obs_date} {target or 'tout'} ===")
        files = io_mod.find_files(self.base_path, obs_date, self.file_pattern, target)
        for t in ['tpi', 'spectrum', 'image']:
            print(f"{t}s: {len(files[t])}")

        out_dir = os.path.join(self.base_path, obs_date, "output_nametarget")

        # Process TPI first to detect RFI
        self._process_tpi_files(files, obs_date, out_dir, window, polyorder, start_ignore, end_ignore)

        # ===== SPECTRES =====
        for info in files['spectrum']:
            print(f"\n📊 {info['target']} - {info['time']}")
            # Get RFI timestamps for this target if available
            rfi_for_spectrum = None
            if info['target'] in self.rfi_detected:
                # Combine RFI from all TPI observations of this target
                rfi_for_spectrum = {}
                for tpi_time, rfi_data in self.rfi_detected[info['target']].items():
                    for bbc_col, rfi_info in rfi_data.items():
                        if bbc_col not in rfi_for_spectrum:
                            rfi_for_spectrum[bbc_col] = []
                        rfi_for_spectrum[bbc_col].extend(rfi_info['timestamps'])

            fig = spec_mod.process_spectrum(info, self.basefreq, self.bandwidth, self.nchan, rfi_for_spectrum)
            if fig:
                path = os.path.join(out_dir, "plots", info['target'], "unprocessed")
                utils.ensure_dir(path)
                fig.savefig(os.path.join(path, f"{info['target']}_spectrum_{info['time']}.png"), dpi=150)
                import matplotlib.pyplot as plt
                plt.close(fig)
                self.processed.append(f"spectrum:{info['target']}:{info['time']}")
                if rfi_for_spectrum:
                    print(f"  🔴 Spectre avec marquage RFI ({len(rfi_for_spectrum)} BBC affectés)")
                else:
                    print("  ✅ Sauvegardé")
            else:
                self.ignored.append(f"spectrum:{info['target']}:{info['time']}")

            # Create enhanced RFI-highlighted spectrum if RFI detected
            if rfi_for_spectrum and len(rfi_for_spectrum) > 0:
                rfi_fig = spec_mod.create_rfi_highlighted_spectrum(
                    info, self.basefreq, self.bandwidth, self.nchan,
                    rfi_for_spectrum,
                    output_path=os.path.join(path, f"{info['target']}_spectrum_rfi_{info['time']}.png")
                )
                if rfi_fig:
                    import matplotlib.pyplot as plt
                    plt.close(rfi_fig)

                # Create cleaned spectrum with RFI removed
                clean_fig = spec_mod.create_clean_spectrum(
                    info, self.basefreq, self.bandwidth, self.nchan,
                    rfi_for_spectrum,
                    output_path=os.path.join(path, f"{info['target']}_spectrum_rfi_highlight_{info['time']}.png")
                )
                if clean_fig:
                    import matplotlib.pyplot as plt
                    plt.close(clean_fig)
                    print(f"  🔴 Spectre RFI mis en évidence sauvegardé")

        # ===== IMAGES =====
        for info in files['image']:
            print(f"\n🖼️  {info['target']} - {info['time']}")
            try:
                from astropy.io import fits
                with fits.open(info['path']) as hdul:
                    data = None
                    for hdu in hdul:
                        if hdu.data is not None and getattr(hdu.data, 'ndim', 0) >= 2:
                            data = hdu.data
                            break
                    if data is None:
                        self.ignored.append(f"image:{info['target']}:{info['time']}")
                        continue
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 8))
                    if data.ndim == 2:
                        im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto')
                        plt.colorbar(im, ax=ax, label='Intensité')
                    else:
                        ax.plot(data.flatten())
                    ax.set_title(f"Image {info['target']} - {info['time']}")
                    out_img = os.path.join(out_dir, "plots", info['target'], "unprocessed")
                    utils.ensure_dir(out_img)
                    fig.savefig(os.path.join(out_img, f"{info['target']}_image_{info['time']}.png"), dpi=150)
                    plt.close(fig)
                    self.processed.append(f"image:{info['target']}:{info['time']}")
                    print("  ✅ Sauvegardé")
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
                self.ignored.append(f"image:{info['target']}:{info['time']}")

        print(f"\n✅ Terminé: {len(self.processed)} traités, {len(self.ignored)} ignorés")
        return self.processed

    def _process_tpi_files(self, files, obs_date, out_dir, window, polyorder, start_ignore, end_ignore):
        """Process TPI files to detect RFI before processing spectra."""
        for info in files['tpi']:
            print(f"\n🎯 {info['target']} - {info['time']}")

            # raw plot
            try:
                from astropy.io import fits
                import matplotlib.pyplot as plt
                with fits.open(info['path']) as hdul:
                    data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bbc_cols = [c for c in data.dtype.names if 'BBC' in c][:8]
                    jd_col = data['JD']
                    for col in bbc_cols:
                        ax.plot(jd_col, data[col], label=col, alpha=0.7, linewidth=1)
                    ax.legend(loc='upper right', ncol=2)
                    ax.set_xlabel("Time (JD)")
                    ax.set_ylabel("Signal (ADU)")
                    ax.set_title(f"TPI brut {info['target']} - {info['time']}")
                    ax.grid(True, alpha=0.3)
                    out_raw = os.path.join(out_dir, "plots", info['target'], "unprocessed")
                    utils.ensure_dir(out_raw)
                    fig.savefig(os.path.join(out_raw, f"{info['target']}_tpi_{info['time']}.png"), dpi=150)
                    plt.close(fig)
            except Exception:
                pass

            result = tpi_mod.process_tpi(info, self.location, window, polyorder,
                                         start_ignore, end_ignore)
            if result and 'vit_ra' in result:
                import matplotlib.pyplot as plt
                bbc_cols = [c for c in result['clean'].dtype.names if 'BBC' in c][:8]
                jd_col = result['clean']['JD']
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes[0, 0].scatter(result['ra'], result['dec'], s=5, alpha=0.5)
                axes[0, 0].set_xlabel("RA (deg)")
                axes[0, 0].set_ylabel("Dec (deg)")
                axes[0, 0].set_title("RA/Dec")
                axes[0, 0].invert_xaxis()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 1].scatter(result['vit_ra'], result['vit_dec'], s=5, alpha=0.5, c='blue')
                axes[0, 1].set_xlabel("Vitesse RA (deg/s)")
                axes[0, 1].set_ylabel("Vitesse Dec (deg/s)")
                axes[0, 1].set_title("Vitesses")
                axes[0, 1].grid(True, alpha=0.3)
                if 'acc_ra' in result and 'acc_dec' in result:
                    sc = axes[1, 0].scatter(
                        result['acc_ra'], result['acc_dec'],
                        c=np.sqrt(result['vit_ra']**2 + result['vit_dec']**2),
                        s=5, cmap='viridis', alpha=0.7
                    )
                    axes[1, 0].set_xlabel("Accélération RA (deg/s²)")
                    axes[1, 0].set_ylabel("Accélération Dec (deg/s²)")
                    axes[1, 0].set_title("Accélérations (couleur = vitesse)")
                    axes[1, 0].grid(True, alpha=0.3)
                    plt.colorbar(sc, ax=axes[1, 0], label="Vitesse (deg/s)")
                else:
                    axes[1, 0].text(0.5, 0.5, "Accélérations non disponibles",
                                    ha='center', va='center', transform=axes[1, 0].transAxes)
                for col in bbc_cols:
                    axes[1, 1].plot(jd_col, result['clean'][col], label=col, alpha=0.7, linewidth=1)
                axes[1, 1].set_xlabel("Time (JD)")
                axes[1, 1].set_ylabel("Signal (ADU)")
                axes[1, 1].set_title(f"BBCs nettoyées - {info['target']}")
                axes[1, 1].legend(loc='upper right', ncol=2, fontsize='small')
                axes[1, 1].grid(True, alpha=0.3)
                fig.suptitle(f"{info['target']} - {info['time']} (window={window})")
                fig.tight_layout()
                out_proc = os.path.join(out_dir, "plots", info['target'], "processed")
                utils.ensure_dir(out_proc)
                fig.savefig(os.path.join(out_proc, f"{info['target']}_tpi_processed_{info['time']}.png"), dpi=150)
                plt.close(fig)

                # reconstruction
                print("  --- Reconstruction d'image ---")
                # Utiliser la résolution de l'image originale correspondante
                img_shape = io_mod.get_image_resolution(self.base_path, obs_date, info['target'], info['time'])
                if img_shape:
                    nx, ny = img_shape
                    print(f"  Résolution utilisée: {nx} x {ny}")
                else:
                    print(f"  ⚠️  Pas d'image trouvée pour {info['target']}")
                    continue

                # check for holes and lower resolution if needed
                maps = plot_mod.project_tpi_to_map(result, nx=nx, ny=ny, method='binned_statistic')
                occupied = np.sum(maps['hit_map'] > 0)
                total = nx * ny
                coverage = occupied / total
                print(f"  ✅ Résolution finale {info['target']}: {nx}x{ny} (couverture = {coverage*100:.1f}%, {occupied} pixels occupés)\n")

                maps_dir = os.path.join(out_dir, "maps", info['target'])
                utils.ensure_dir(maps_dir)
                methods = ['loop', 'histogram2d', 'binned_statistic']
                for method in methods:
                    maps_hit = plot_mod.project_tpi_to_map(result, nx=nx, ny=ny, method=method)
                    np.save(os.path.join(maps_dir, f"{info['target']}_hitmap_{method}_{info['time']}.npy"), maps_hit['hit_map'])
                    if bbc_cols:
                        bbc = bbc_cols[0]
                        maps_data = plot_mod.project_tpi_to_map(result, nx=nx, ny=ny,
                                                              method=method,
                                                              bbc_column=bbc,
                                                              statistic='mean')
                        np.save(os.path.join(maps_dir, f"{info['target']}_{bbc}_{method}_{info['time']}.npy"), maps_data['data_map'])

                maps_best = plot_mod.project_tpi_to_map(result, nx=nx, ny=ny, method='binned_statistic')
                plot_mod.plot_tpi_maps_comparison(maps_best, info['target'], info['time'], out_proc)

                # Estimate FWHM using first BBC data
                if bbc_cols:
                    bbc_for_fwhm = bbc_cols[0]
                    maps_fwhm = plot_mod.project_tpi_to_map(result, nx=nx, ny=ny,
                                                            method='binned_statistic',
                                                            bbc_column=bbc_for_fwhm,
                                                            statistic='mean')
                    # Check if we have valid data
                    valid_mask = ~np.isnan(maps_fwhm['data_map']) & (maps_fwhm['data_map'] > 0)
                    if np.sum(valid_mask) > 0:
                        try:
                            fwhm_result = plot_mod.plot_fwhm_estimation(
                                maps_fwhm['data_map'],
                                maps_fwhm['ra_center'],
                                maps_fwhm['dec_center'],
                                info['target'],
                                info['time'],
                                out_proc
                            )
                        except Exception as e:
                            print(f"  ⚠️  Erreur lors de l'estimation FWHM: {e}")

                if len(bbc_cols) >= 2:
                    plot_mod.compare_bbc_maps(result, info['target'], info['time'], out_proc, bbc_cols, nx=nx, ny=ny)

                # Detect RFI in this TPI
                rfi_detected = tpi_mod.detect_rfi_in_tpi(result, method='mad_relative', threshold=1.0)
                if rfi_detected:
                    if info['target'] not in self.rfi_detected:
                        self.rfi_detected[info['target']] = {}
                    self.rfi_detected[info['target']][info['time']] = rfi_detected
                    print(f"  🔴 RFI détecté dans {len(rfi_detected)} BBC(s): {[bbc for bbc in rfi_detected.keys()]}")
                    for bbc, rfi_info in rfi_detected.items():
                        print(f"    {bbc}: {rfi_info['count']} points ({rfi_info['percentage']:.1f}%), sévérité moyenne: {rfi_info['avg_severity']:.1f}")

                print("  ✅ Reconstruction d'image terminée")
                self.processed.append(f"tpi_map:{info['target']}:{info['time']}")
            else:
                self.ignored.append(f"tpi:{info['target']}:{info['time']}")


if __name__ == '__main__':
    pipe = GradPipeline()
    pipe.run('20260130', 'SUN', window=5)
