import numpy as np
import h5py

# Load the pseudo labels from the .npy file
pseudo_labels = np.load('data/pseudo_labels.npy')

# Create an HDF5 file with SWMR support
with h5py.File('data/pseudo_labels.h5', 'w', libver='latest') as f:
    f.create_dataset('pseudo_labels', data=pseudo_labels)

print("Pseudo-labels successfully converted to HDF5 format with SWMR support.")
