import argparse
import ase
from ase.io import read

import abtem
import cupy
import numpy as np
import matplotlib.pyplot as plt

# -- GPU block
import os
import dask
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ['CUPY_GPU_MEMORY_LIMIT'] = '100%'
dask.config.set({"num_workers": 1})
abtem.config.set({"device": "gpu", "fft": "fftw"})
abtem.config.set({"dask.chunk-size-gpu" : "512 MB"})
abtem.config.set({"cupy.fft-cache-size" : "0 MB"})

import time

def main(filename, ctf_cs, show_images=False):
    # -- start
    start_time = time.time()

    # -- input
    ctf_cs = float(ctf_cs)
    ctf_print = str(int(ctf_cs*1e6))
    atoms = read('data/'+filename+'.cfg')
    print (f'--- Working with sample {filename}\n')
    
    if show_images:
        abtem.show_atoms(
            atoms,
            plane="xz",  # show a view perpendicular to the 'xy' plane
            scale=0.4,   # scale atoms to 0.4 of their covalent radii; default is 0.75
            legend=True, # show a legend with the atomic symbols
        )
        plt.show()

    # -- phonons
    frozen_phonons = abtem.FrozenPhonons(atoms, 2, sigmas=0.1)

    # -- potential
    potential = abtem.Potential(
        frozen_phonons,
        gpts=896, # 768, 896, 1024, 1152, 1280 
        slice_thickness=2,
    )

    # -- smatrix
    s_matrix = abtem.SMatrix(
        potential=potential, 
        energy=200e3, 
        semiangle_cutoff=20,  
        interpolation=4, 
        store_on_host=True,
    )

    print ('System setup complete;')
    bf_cutoff = s_matrix.cutoff_angles[0]
    print (f'{bf_cutoff:.2f} BF semiangle cutoff (mrad)\n')

    # -- CTF
    Cs = ctf_cs * 1e10  # e.g. ctf_cs = 8e-6
    ctf = abtem.CTF(Cs=Cs, defocus="scherzer", energy=s_matrix.energy)
    print ('CTF setup complete;')
    print(f"Cs = {ctf_print} (microns)")
    print(f"defocus = {ctf.defocus:.2f} (Å)\n")

    if show_images:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        s_matrix.dummy_probes(plane="entrance", ctf=ctf).show(ax=ax1, title="entrance", power=1)
        s_matrix.dummy_probes(plane="exit", ctf=ctf).show(ax=ax2, title="exit", power=1)
        plt.show()

    # -- detectors
    detectors = abtem.FlexibleAnnularDetector()
    flexible_measurement = s_matrix.scan(detectors=detectors, ctf=ctf)
    print ('Detector setup complete;')

    # -- measurement
    print ('Init multislice PRISM...')
    flexible_measurement.compute() # only for small models
    print ('\n Measurement complete! \n')
    flexible_measurement.to_zarr('cache/'+filename+'_'+ctf_print+'.zarr')

    bf_measurement = flexible_measurement.integrate_radial(0, s_matrix.semiangle_cutoff)
    aadf_measurement = flexible_measurement.integrate_radial(45, 70) # change range manually

    if show_images:
        measurements = abtem.stack(
            [bf_measurement, aadf_measurement], ("BF", "AADF")
        )

        measurements.show(
            explode=True,
            figsize=(14, 5),
            cbar=True,
        )
        plt.show()

    noisy_measurements = bf_measurement.poisson_noise(dose_per_area=1e7) # finite dose (1e7 e-/A^2)
    filtered_bf_0p5 = noisy_measurements.gaussian_filter(0.5) # spatial coherence ~ probe size (0.5 A)
    filtered_bf_1p0 = noisy_measurements.gaussian_filter(1.0) # spatial coherence ~ probe size (1.0 A)

    if show_images:
        measurements = abtem.stack(
            [noisy_measurements, filtered_bf_0p5, filtered_bf_1p0], ("BF + PN", "PN + GN 0.5 A", "PN + GN 1.0 A")
        )

        measurements.show(
            explode=True,
            figsize=(14, 5),
            cbar=True,
        )
        plt.show()

    filtered_bf_1p0.show(cmap='binary_r')
    plt.axis('off')
    plt.savefig('outputs/'+filename+'_'+ctf_print+'.png', bbox_inches='tight', pad_inches=0, dpi=150)

    elapsed_time = time.time() - start_time

    print(f"--- {elapsed_time:.2f} seconds ---")
    # time.sleep(30)
    # -- end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abtem run with integrated GPU")
    parser.add_argument('filename', type=str, help='Path to the file.')
    parser.add_argument('ctf_cs', type=str, help='Spherical aberration (e.g. 8e-6).')
    parser.add_argument('--show_images', action='store_true', help='Flag to display images (default is False).')
    args = parser.parse_args()
    main(args.filename, args.ctf_cs, args.show_images)