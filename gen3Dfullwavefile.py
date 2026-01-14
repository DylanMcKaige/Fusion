"""
Generate fullwave .dat files for ERMES in 3D from ne.dat and topfile.json

Run this AFTER meshing and AFTER running in DEBUG mode in ERMES to generate the problem_name-1.dat file
"""
import json, re
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from math import *

# These 3 paths just need to be changed to your paths.

ne_data_path = 'YOUR_PATH_HERE' # ne.dat 
topfile_data_path = 'YOUR_PATH_HERE' # topfile in json format
msh_path = 'YOUR_PATH_HERE' # Inside .gid problem folder

ne_file = pd.read_csv(ne_data_path, sep=' ', header=None, skiprows=1) 
with open(topfile_data_path, 'r') as file: topfile_data = json.load(file) 

ne_data = ne_file.to_numpy(dtype=float).T  
ne_spline = UnivariateSpline(ne_data[0]**2, ne_data[1], k=5, s=0, ext=1) # Spline order could be varied depending on ne data

# Load cross-section fields (R,Z grids and 2D arrays)
A = {k: np.array(v) for k, v in topfile_data.items()}

Rg = np.asarray(A['R'])
Zg = np.asarray(A['Z'])
PsiRZ = np.asarray(A['pol_flux'])
BrRZ = np.asarray(A['Br'])
BtRZ = np.asarray(A['Bt'])
BzRZ = np.asarray(A['Bz'])

# Build splines on the cross-section, 
# the spline order likely can be varied for a more physical fit
pol_flux_spline = RectBivariateSpline(Rg, Zg, PsiRZ.T, kx=5, ky=5, s=0)
Br_spline = RectBivariateSpline(Rg, Zg, BrRZ.T,  kx=5, ky=5, s=0)
Bt_spline = RectBivariateSpline(Rg, Zg, BtRZ.T,  kx=5, ky=5, s=0)
Bz_spline = RectBivariateSpline(Rg, Zg, BzRZ.T,  kx=5, ky=5, s=0)

# Parse GiD mesh nodes:  No[ID] = p(x,y,z);
node_ids, xs, ys, zs = [], [], [], []
pat = re.compile(r"No\[(\d+)\]\s*=\s*p\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*;")

with open(msh_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            node_ids.append(int(m.group(1)))
            xs.append(float(m.group(2)))
            ys.append(float(m.group(3)))
            zs.append(float(m.group(4)))

node_ids = np.asarray(node_ids, dtype=np.int64)
x = np.asarray(xs, float)
y = np.asarray(ys, float)
z = np.asarray(zs, float)

if node_ids.size == 0:
    raise RuntimeError("No nodes parsed. Check the .dat/.msh format.")

# Map (x,y,z) -> (R, phi, Z) (cylindrical)
# x = R cosphi, y = Z, z = -R sinphi
Rnod = np.hypot(x, z)
Znod = y
phi = np.arctan2(-z, x) # minus to satisfy z = -R sinphi

# Evaluate Psi, Br, Bt, Bz at node (R,Z)
Psi = pol_flux_spline.ev(Rnod, Znod)
Br = Br_spline.ev(Rnod, Znod)
Bt = Bt_spline.ev(Rnod, Znod)
Bz = Bz_spline.ev(Rnod, Znod)
ne = ne_spline(Psi)

# Convert (Br,Bt,Bz) -> (Bx,By,Bz_cart) at local phi
# e_R=(cosphi,0,-sinphi), e_t=(-sinphi,0,-cosphi), e_Z=(0,1,0)
c = np.cos(phi)
s = np.sin(phi)
Bx = Br*c - Bt*s
By = Bz
Bz_cart = -Br*s - Bt*c

# Write outputs (sorted by NodeID)
order = np.argsort(node_ids)
nid_sorted = node_ids[order]

np.savetxt(
    "ne.dat",
    np.column_stack([nid_sorted, ne[order]*1e19]), # Scale here AFTER splining to minimize issues with the scale being too large
    fmt=["%d", "%.8e"]
)

np.savetxt(
    "mag.dat",
    np.column_stack([nid_sorted, Bx[order], By[order], Bz_cart[order]]),
    fmt=["%d", "%.8e", "%.8e", "%.8e"]
)

# Sanity print
print(f"Wrote {nid_sorted.size} nodes to ne.dat and mag.dat")