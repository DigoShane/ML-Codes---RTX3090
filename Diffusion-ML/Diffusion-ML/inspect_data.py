#uncomment and run both atoms_list lines to check consistency of data sets:
#Make sure:
#No NaNs
#Distribution looks reasonable
#No absurd outliers (like 100 eV)

from ase.io import read # to read .xyz files
import numpy as np
import matplotlib.pyplot as plt

# Load BVSE training set
atoms_list = read("data/raw/nebBVSE122k/nebBVSE122k_train.xyz", index=":")
#atoms_list = read("data/raw/nebDFT2k/nebDFT2k_centroids.xyz", index=":")

print("Total structures:", len(atoms_list))

barriers = []
for atoms in atoms_list:
    barriers.append(atoms.info["em"])

barriers = np.array(barriers)

print("Min:", barriers.min())
print("Max:", barriers.max())
print("Mean:", barriers.mean())

plt.hist(barriers, bins=50)
plt.xlabel("Barrier (eV)")
plt.ylabel("Count")
plt.title("BVSE Barrier Distribution")
plt.show()
